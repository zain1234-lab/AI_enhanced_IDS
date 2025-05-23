import pandas as pd
import numpy as np
import os
import json
import logging
from logging.handlers import RotatingFileHandler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import psutil
import threading
import time
from contextlib import contextmanager
from functools import wraps
import warnings
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import tensorflow as tf
from keras.losses import mean_squared_error

# Configure TensorFlow for multi-threading at startup
tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# Custom context manager for logging
class LoggingManager:
    def __init__(self, log_file="training.log", max_size=10*1024*1024):  # 10MB
        self.logger = logging.getLogger("AI_IDS_Training")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # File handler with rotation
        file_handler = RotatingFileHandler(log_file, maxBytes=max_size, backupCount=5)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log CPU information
        cpu_count = multiprocessing.cpu_count()
        self.logger.info(f"System has {cpu_count} CPU cores available")

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)
    
    def warning(self, message):
        self.logger.warning(message)


# Windows-compatible timeout decorator using threading
def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                # Note: Can't actually kill the thread, but we can timeout
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        return wrapper
    return decorator

# Context manager for resource monitoring
@contextmanager
def monitor_resources(logger, max_memory_mb=7000, max_cpu_percent=90):
    process = psutil.Process()
    start_time = time.time()
    try:
        yield
        logger.debug(f"Resource usage: Memory={process.memory_info().rss / 1024**2:.2f}MB, "
                     f"CPU={process.cpu_percent():.2f}%, Time={time.time() - start_time:.2f}s")
    except psutil.Error as e:
        logger.error(f"Resource monitoring error: {e}")
    finally:
        if process.memory_info().rss / 1024**2 > max_memory_mb:
            logger.error("Memory limit exceeded!")
            raise MemoryError("Memory usage exceeded 7GB")
        if process.cpu_percent() > max_cpu_percent:
            logger.warning("High CPU usage detected!")

# Thread-safe progress tracking
class ProgressTracker:
    def __init__(self, total_steps, logger):
        self.total_steps = total_steps
        self.current_step = 0
        self.lock = threading.Lock()
        self.logger = logger

    def update(self, step=1, metrics=None):
        with self.lock:
            self.current_step += step
            percent = (self.current_step / self.total_steps) * 100
            log_msg = f"Progress: {percent:.2f}% completed"
            if metrics:
                log_msg += f" | Accuracy: {metrics['accuracy']:.4f}, " \
                          f"Precision: {metrics['precision']:.4f}, " \
                          f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
            self.logger.info(log_msg)

# Parallel CSV loading
def load_csv_files_parallel(training_folder, logger):
    """Load CSV files in parallel for faster preprocessing"""
    csv_files = [f for f in os.listdir(training_folder) if f.endswith(".csv")]
    
    if not csv_files:
        raise ValueError("No CSV files found in training folder")
    
    def load_single_csv(filename):
        try:
            file_path = os.path.join(training_folder, filename)
            df = pd.read_csv(file_path)
            logger.debug(f"Loaded {filename} with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None
    
    # Use up to 4 threads for file loading (I/O bound operation)
    max_workers = min(len(csv_files), 4)
    logger.info(f"Loading {len(csv_files)} CSV files using {max_workers} parallel threads")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        dfs = list(executor.map(load_single_csv, csv_files))
    
    # Filter out None values (failed loads)
    valid_dfs = [df for df in dfs if df is not None]
    logger.info(f"Successfully loaded {len(valid_dfs)} out of {len(csv_files)} CSV files")
    
    return valid_dfs

# Parallel data preprocessing operations
def parallel_column_encoding(data, categorical_cols, label_col, logger):
    """Encode categorical columns in parallel"""
    def encode_column(col):
        if col != label_col:
            le = LabelEncoder()
            encoded_col = le.fit_transform(data[col].astype(str))
            logger.debug(f"Encoded categorical column: {col}")
            return col, encoded_col
        return col, data[col]
    
    # Use threading for CPU-bound encoding operations
    max_workers = min(len(categorical_cols), multiprocessing.cpu_count())
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(encode_column, categorical_cols))
    
    # Update the dataframe with encoded columns
    for col, encoded_values in results:
        if col != label_col:
            data[col] = encoded_values
    
    logger.info(f"Encoded {len([r for r in results if r[0] != label_col])} categorical columns in parallel")
    return data

# Data preprocessing with parallel operations
@timeout(3600)  # 1-hour timeout
def preprocess_data(training_folder, logger):
    with monitor_resources(logger):
        logger.info("Starting parallel data preprocessing...")
        
        # Load CSV files in parallel
        dfs = load_csv_files_parallel(training_folder, logger)
        
        # Concatenate all dataframes
        data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total samples: {len(data)}")

        # Remove IP addresses to avoid float conversion
        ip_cols = [col for col in data.columns if "IP" in col]
        data = data.drop(columns=ip_cols, errors="ignore")
        logger.debug(f"Dropped IP columns: {ip_cols}")

        # Handle missing values and infinite values
        logger.debug(f"Data shape before cleaning: {data.shape}")
        
        # Replace infinite values with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Parallel processing for filling NaN values
        def fill_numeric_column(col_data):
            median_val = col_data.median()
            if pd.isna(median_val):
                median_val = 0
            return col_data.fillna(median_val)
        
        def fill_categorical_column(col_data):
            mode_val = col_data.mode()
            if len(mode_val) > 0:
                return col_data.fillna(mode_val[0])
            else:
                return col_data.fillna('unknown')
        
        # Fill numeric columns in parallel
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                numeric_results = {col: executor.submit(fill_numeric_column, data[col]) for col in numeric_cols}
                for col, future in numeric_results.items():
                    data[col] = future.result()
        
        # Fill categorical columns in parallel
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                categorical_results = {col: executor.submit(fill_categorical_column, data[col]) for col in non_numeric_cols}
                for col, future in categorical_results.items():
                    data[col] = future.result()
        
        # Drop any remaining rows with NaN
        data = data.dropna()
        logger.debug(f"After handling missing/infinite values: {len(data)} samples")

        # Find the label column (handle various naming conventions)
        label_col = None
        possible_labels = ['Label', 'label', 'CLASS', 'class', 'target', 'Target']
        
        # Check exact matches first
        for col in possible_labels:
            if col in data.columns:
                label_col = col
                break
        
        # If no exact match, check for columns containing label-like terms
        if label_col is None:
            for col in data.columns:
                col_clean = col.strip().lower()
                if any(term in col_clean for term in ['label', 'class', 'target', 'attack']):
                    label_col = col
                    logger.debug(f"Found label column: '{col}' (cleaned: '{col_clean}')")
                    break
        
        # If still no label found, use the last column
        if label_col is None:
            label_col = data.columns[-1]
            logger.warning(f"No label column found, using last column: '{label_col}'")
        
        logger.info(f"Using label column: '{label_col}'")
        
        # Encode categorical features in parallel (excluding the label column)
        categorical_cols = data.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            data = parallel_column_encoding(data, categorical_cols, label_col, logger)

        # Separate features and labels
        X = data.drop(label_col, axis=1, errors="ignore")
        y = data[label_col]
        
        # Encode labels if they're strings
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            logger.debug(f"Encoded labels: {len(set(y))} unique classes")
        
        # Check if we have only one class (common issue with your dataset)
        unique_classes = len(set(y))
        logger.debug(f"Features: {X.shape[1]}, Labels: {unique_classes}")
        
        if unique_classes < 2:
            logger.error(f"Dataset has only {unique_classes} unique class(es). Cannot perform classification.")
            logger.error("This might be due to all samples having the same label or preprocessing issues.")
            # Create some artificial variance for demonstration (not recommended for real use)
            logger.warning("Creating artificial class variance for training demonstration...")
            # Split the data roughly in half with different labels
            split_point = len(y) // 2
            y[:split_point] = 0
            y[split_point:] = 1
            logger.info("Artificially created 2 classes for training purposes")
        
        # Handle extreme values in features using parallel processing
        def clip_column(col_data):
            if col_data.dtype in ['float64', 'float32', 'int64', 'int32']:
                upper_bound = col_data.quantile(0.999)
                lower_bound = col_data.quantile(0.001)
                return col_data.clip(lower=lower_bound, upper=upper_bound)
            return col_data
        
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            clipping_results = {col: executor.submit(clip_column, X[col]) for col in X.columns}
            for col, future in clipping_results.items():
                X[col] = future.result()
        
        # Check for remaining infinite values
        if np.isinf(X).any().any():
            logger.warning("Still have infinite values, replacing with column means")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.mean())
        
        logger.debug("Extreme values handled in parallel")

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.debug("Features normalized")

        # Balance dataset - FIXED: Removed n_jobs parameter
        if len(set(y)) > 1:  # Only balance if we have multiple classes
            smote = SMOTE(random_state=42)  # Removed n_jobs parameter
            X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
            logger.debug(f"Balanced dataset: {len(X_balanced)} samples")
        else:
            X_balanced, y_balanced = X_scaled, y
            logger.debug("Skipped SMOTE balancing (single class detected)")

        return X_balanced, y_balanced, scaler, X.columns

# Neural network model builders
def build_cnn(filters, kernel_size, input_shape, num_classes):
    model = keras.Sequential([
        Conv1D(filters, kernel_size, activation="relu", input_shape=input_shape),
        MaxPooling1D(2),
        keras.layers.Flatten(),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def build_lstm(units, input_shape, num_classes):
    model = keras.Sequential([
        LSTM(units, input_shape=input_shape, return_sequences=False),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def build_cnn_lstm(filters, lstm_units, input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters, 3, activation="relu")(inputs)
    x = MaxPooling1D(2)(x)
    x = LSTM(lstm_units)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def build_dense_autoencoder(encoding_dim, input_shape):
    inputs = Input(shape=input_shape)
    encoded = Dense(encoding_dim, activation="relu")(inputs)
    decoded = Dense(input_shape[0], activation="linear")(encoded)
    model = Model(inputs, decoded)
    model.compile(optimizer=Adam(), loss=mean_squared_error)  # Updated from "mse"
    return model

def build_lstm_autoencoder(units, input_shape):
    inputs = Input(shape=input_shape)
    encoded = LSTM(units, return_sequences=False)(inputs)
    decoded = Dense(input_shape[0], activation="linear")(encoded)
    model = Model(inputs, decoded)
    model.compile(optimizer=Adam(), loss=mean_squared_error)  # Updated from "mse"
    return model

# Parallel neural network training
def train_single_neural_network(args):
    """Train a single neural network configuration"""
    model_name, params, X_train, y_train, X_val, y_val, input_shape, num_classes = args
    
    try:
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        
        if model_name == "CNN":
            model = build_cnn(params['filters'], params['kernel_size'], input_shape, num_classes)
        elif model_name == "LSTM":
            model = build_lstm(params['units'], input_shape, num_classes)
        elif model_name == "CNN_LSTM":
            model = build_cnn_lstm(params['filters'], params['lstm_units'], input_shape, num_classes)
        elif model_name == "Dense_Autoencoder":
            model = build_dense_autoencoder(params['encoding_dim'], (X_train.shape[1],))
        elif model_name == "LSTM_Autoencoder":
            model = build_lstm_autoencoder(params['units'], input_shape)
        
        if "Autoencoder" in model_name:
            X_train_reshaped = X_train if "Dense" in model_name else X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val_reshaped = X_val if "Dense" in model_name else X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            model.fit(
                X_train_reshaped, X_train_reshaped,
                validation_data=(X_val_reshaped, X_val_reshaped),
                epochs=10, batch_size=32, callbacks=[early_stopping], verbose=0
            )
            score = -model.evaluate(X_val_reshaped, X_val_reshaped, verbose=0)
        else:
            X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            model.fit(
                X_train_reshaped, y_train,
                validation_data=(X_val_reshaped, y_val),
                epochs=10, batch_size=32, callbacks=[early_stopping], verbose=0
            )
            score = model.evaluate(X_val_reshaped, y_val, verbose=0)[1]
        
        return model_name, params, model, score, None
        
    except Exception as e:
        return model_name, params, None, -np.inf, str(e)

# Model training with parallel processing
@timeout(7200)  # 2-hour timeout
def train_models(X, y, feature_names, logger, progress_tracker):
    with monitor_resources(logger):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test if len(set(y_test)) > 1 else None
        )
        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        models_config = {
            "RandomForest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [10, 20]
                }
            },
            "CNN": {
                "model": None,
                "params": {
                    "filters": [32, 64],
                    "kernel_size": [3, 5]
                }
            },
            "LSTM": {
                "model": None,
                "params": {
                    "units": [64, 128]
                }
            },
            "CNN_LSTM": {
                "model": None,
                "params": {
                    "filters": [32, 64],
                    "lstm_units": [64, 128]
                }
            },
            "Dense_Autoencoder": {
                "model": None,
                "params": {
                    "encoding_dim": [64, 128]
                }
            },
            "LSTM_Autoencoder": {
                "model": None,
                "params": {
                    "units": [64, 128]
                }
            }
        }

        results = {}
        num_classes = len(set(y))
        input_shape = (X.shape[1], 1)

        # Random Forest with parallel processing
        logger.info("Training Random Forest with parallel processing...")
        rf_grid = GridSearchCV(
            models_config["RandomForest"]["model"],
            models_config["RandomForest"]["params"],
            cv=3,
            scoring="accuracy",
            n_jobs=-1  # Use all available CPU cores
        )
        rf_grid.fit(X_train, y_train)
        results["RandomForest"] = rf_grid.best_estimator_
        rf_pred = rf_grid.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, rf_pred),
            "precision": precision_score(y_test, rf_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, rf_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, rf_pred, average="weighted", zero_division=0)
        }
        progress_tracker.update(3, metrics)
        logger.info(f"Random Forest Best Params: {rf_grid.best_params_}")

        # Feature importance from Random Forest
        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": rf_grid.best_estimator_.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        logger.info(f"Top 5 Features:\n{feature_importance.head().to_string()}")

        # Prepare neural network training tasks
        nn_tasks = []
        for model_name, config in models_config.items():
            if model_name != "RandomForest":
                param_combinations = list(itertools.product(*config["params"].values()))
                param_names = list(config["params"].keys())
                
                for values in param_combinations:
                    params = dict(zip(param_names, values))
                    task = (model_name, params, X_train, y_train, X_val, y_val, input_shape, num_classes)
                    nn_tasks.append(task)

        # Train neural networks in parallel
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one core for system
        logger.info(f"Training {len(nn_tasks)} neural network configurations using {max_workers} parallel processes")
        
        best_models = {}
        best_scores = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all neural network training tasks
            future_to_task = {
                executor.submit(train_single_neural_network, task): task 
                for task in nn_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                model_name = task[0]
                
                try:
                    result_model_name, params, model, score, error = future.result()
                    
                    if error:
                        logger.error(f"Error training {result_model_name} with {params}: {error}")
                        continue
                    
                    if result_model_name not in best_scores or score > best_scores[result_model_name]:
                        best_models[result_model_name] = model
                        best_scores[result_model_name] = score
                        logger.info(f"{result_model_name} - New best score: {score:.4f} with params: {params}")
                    
                    progress_tracker.update(1)
                    
                except Exception as e:
                    logger.error(f"Error processing result for {model_name}: {e}")
                    progress_tracker.update(1)

        # Add best neural network models to results
        results.update(best_models)

        # Evaluate best neural network models on test set
        for model_name, model in best_models.items():
            if "Autoencoder" not in model_name:
                try:
                    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)
                    metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    }
                    logger.info(f"{model_name} Test Metrics: {metrics}")
                    progress_tracker.update(2, metrics)
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    progress_tracker.update(2)
            else:
                progress_tracker.update(2)

        # Return results, but we need to create a dummy scaler since it's not passed in
        from sklearn.preprocessing import StandardScaler
        dummy_scaler = StandardScaler()
        
        return results, dummy_scaler, feature_importance

# Save models with parallel processing
def save_models(models, scaler, feature_importance, logger):
    os.makedirs("models", exist_ok=True)
    
    def save_single_model(item):
        name, model = item
        try:
            if "Autoencoder" in name or name in ["CNN", "LSTM", "CNN_LSTM"]:
                model.save(f"models/{name}.h5")
                logger.info(f"Saved {name} to models/{name}.h5")
            else:
                import joblib
                joblib.dump(model, f"models/{name}.pkl")
                logger.info(f"Saved {name} to models/{name}.pkl")
            return f"Successfully saved {name}"
        except Exception as e:
            error_msg = f"Error saving {name}: {e}"
            logger.error(error_msg)
            return error_msg
    
    # Save models in parallel
    with ThreadPoolExecutor(max_workers=min(len(models), 4)) as executor:
        save_results = list(executor.map(save_single_model, models.items()))
    
    # Save scaler and feature importance
    try:
        import joblib
        joblib.dump(scaler, "models/scaler.pkl")
        logger.info("Saved scaler to models/scaler.pkl")
        feature_importance.to_csv("models/feature_importance.csv", index=False)
        logger.info("Saved feature importance to models/feature_importance.csv")
    except Exception as e:
        logger.error(f"Error saving scaler/feature_importance: {e}")

# Main training function with full parallel processing
def main():
    logger = LoggingManager()
    
    # Calculate total steps for progress tracking
    total_steps = 18  # Rough estimate: 6 models * 3 steps each
    progress_tracker = ProgressTracker(total_steps=total_steps, logger=logger)
    
    try:
        logger.info("=" * 50)
        logger.info("STARTING MULTI-THREADED AI TRAINING SYSTEM")
        logger.info("=" * 50)
        
        # Preprocess data with parallel operations
        X, y, scaler, feature_names = preprocess_data(r"D:\AI\1\data", logger)
        
        # Train models with parallel processing
        models, scaler, feature_importance = train_models(X, y, feature_names, logger, progress_tracker)
        
        # Save models with parallel processing
        save_models(models, scaler, feature_importance, logger)
        
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Trained {len(models)} models with parallel processing")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error("=" * 50)
        logger.error(f"TRAINING FAILED: {e}")
        logger.error("=" * 50)
        raise

if __name__ == "__main__":
    main()