"""
Self-improving model for AI-IDS
---------------------------
This module provides self-improving model capabilities for the AI-IDS system.
"""

import os
import sys
import time
import json
import pickle
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from enhanced_logging import get_logger, track_performance
from thread_safety import SafeThread, AtomicBoolean, synchronized
from resource_monitor import track_operation, limit_memory_usage
from security_utils import get_secure_serializer
from training import get_model_trainer, get_data_processor

# Get logger for this module
logger = get_logger("self_improving_model")

# Reuse model loaders from model_integration if possible, or redefine here
# This avoids code duplication if this module needs independent loading
MODEL_LOADERS = {
    ".pkl": lambda p: pickle.load(open(p, "rb")),
    ".joblib": lambda p: __import__("joblib").load(p),
    ".h5": lambda p: __import__("tensorflow").keras.models.load_model(p),
    # Add other loaders if needed
}

class ModelFeedbackCollector:
    """Class for collecting model feedback"""
    
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the model feedback collector"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        # Initialize feedback storage {model_name: [feedback_entries]}
        self.feedback = {}
        # Initialize lock
        self._lock = threading.RLock()
        self._initialized = True
        logger.info("Model feedback collector initialized")
    
    @synchronized
    def add_feedback(self, model_name, features, prediction, actual, confidence=None):
        """Add feedback for a model"""
        try:
            # Create feedback entry
            entry = {
                "timestamp": time.time(),
                "features": features,  # Consider converting features to list for JSON compatibility if saving
                "prediction": prediction,
                "actual": actual,
                "confidence": confidence
            }
            
            # Add to feedback
            if model_name not in self.feedback:
                self.feedback[model_name] = []
            
            self.feedback[model_name].append(entry)
            
            logger.debug(f"Added feedback for model {model_name}: {prediction} -> {actual}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding feedback for {model_name}: {e}", exc_info=True)
            return False
    
    @synchronized
    def get_feedback(self, model_name, limit=None):
        """Get feedback for a model"""
        try:
            feedback_list = self.feedback.get(model_name, [])
            if not feedback_list:
                # logger.debug(f"No feedback found for model {model_name}")
                return []
            
            # Return a copy to prevent external modification
            feedback_copy = list(feedback_list)
            
            # Apply limit
            if limit is not None and limit > 0:
                feedback_copy = feedback_copy[-limit:]
            
            # logger.debug(f"Retrieved {len(feedback_copy)} feedback entries for model {model_name}")
            return feedback_copy
        
        except Exception as e:
            logger.error(f"Error getting feedback for {model_name}: {e}", exc_info=True)
            return []
    
    @synchronized
    def clear_feedback(self, model_name):
        """Clear feedback for a model"""
        try:
            if model_name in self.feedback:
                count = len(self.feedback[model_name])
                self.feedback[model_name] = []
                logger.info(f"Cleared {count} feedback entries for model {model_name}")
                return True
            else:
                # logger.warning(f"No feedback to clear for model {model_name}")
                return False
        
        except Exception as e:
            logger.error(f"Error clearing feedback for {model_name}: {e}", exc_info=True)
            return False
    
    @synchronized
    def save_feedback(self, model_name, file_path):
        """Save feedback to file"""
        feedback_to_save = self.feedback.get(model_name)
        if not feedback_to_save:
            logger.warning(f"No feedback to save for model {model_name}")
            return False
            
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Ensure features are JSON serializable (convert numpy arrays)
            serializable_feedback = []
            for entry in feedback_to_save:
                new_entry = entry.copy()
                if isinstance(new_entry.get("features"), np.ndarray):
                    new_entry["features"] = new_entry["features"].tolist()
                serializable_feedback.append(new_entry)

            with open(file_path, "w") as f:
                json.dump(serializable_feedback, f, indent=2)
            
            logger.info(f"Saved {len(serializable_feedback)} feedback entries for model {model_name} to {file_path}")
            return True
        
        except TypeError as te:
            logger.error(f"Error saving feedback for {model_name} (Serialization Error): {te}. Features might not be JSON serializable.", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Error saving feedback for {model_name} to {file_path}: {e}", exc_info=True)
            return False
    
    @synchronized
    def load_feedback(self, model_name, file_path):
        """Load feedback from file"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Feedback file not found: {file_path}")
                return False
            
            with open(file_path, "r") as f:
                feedback = json.load(f)
            
            # Set feedback (overwrites existing)
            self.feedback[model_name] = feedback
            
            logger.info(f"Loaded {len(feedback)} feedback entries for model {model_name} from {file_path}")
            return True
        
        except json.JSONDecodeError as je:
            logger.error(f"Error loading feedback from {file_path}: Invalid JSON format. {je}")
            return False
        except Exception as e:
            logger.error(f"Error loading feedback for {model_name} from {file_path}: {e}", exc_info=True)
            return False

class ModelImprover:
    """Class for improving models based on feedback"""
    
    def __init__(self, model_name="default_model", model_dir="models"):
        """Initialize the model improver"""
        self.model_name = model_name  # Store the target model name
        self.model_dir = model_dir
        # Initialize components
        self.feedback_collector = ModelFeedbackCollector.get_instance()
        # Pass model_dir to trainer
        self.model_trainer = get_model_trainer(model_name, model_dir) 
        self.data_processor = get_data_processor()
        # Initialize lock for thread safety if needed for internal state
        self._lock = threading.RLock()
        logger.info(f"Model improver initialized for model '{model_name}' with model_dir '{model_dir}'")
    
    @track_performance("improve_model")
    @limit_memory_usage(max_mb=1000)  # Decorator might need adjustment
    def improve_model(self, model_name, current_model, model_dir, max_training_time=300):
        """Improve a model based on feedback."""
        logger.info(f"Attempting to improve model '{model_name}'...")
        try:
            # Get feedback for the specific model
            feedback = self.feedback_collector.get_feedback(model_name)
            
            if not feedback:
                logger.info(f"No feedback available for model '{model_name}'. Improvement skipped.")
                return current_model, False  # Return original model, no improvement
            
            # Create training data
            X = []
            y = []
            
            for entry in feedback:
                # Ensure features and actual label exist
                if entry.get("features") is not None and entry.get("actual") is not None:
                    X.append(entry["features"])
                    y.append(entry["actual"])
                else:
                    logger.warning(f"Skipping feedback entry due to missing features or actual label: {entry}")

            # Check if enough valid samples
            min_samples = 10  # Configurable threshold
            if len(X) < min_samples:
                logger.warning(f"Not enough valid feedback samples ({len(X)} < {min_samples}) for model '{model_name}'. Improvement skipped.")
                return current_model, False
            
            # Convert to numpy arrays
            try:
                X = np.array(X)
                y = np.array(y)
                logger.info(f"Prepared {len(X)} samples for retraining model '{model_name}'.")
            except Exception as e:
                logger.error(f"Error converting feedback to numpy arrays for model '{model_name}': {e}", exc_info=True)
                return current_model, False

            # Train model using the ModelTrainer component
            # Pass the current model for potential incremental training if supported
            improved_model = self.model_trainer.train_model(model_name, X, y, max_training_time, base_model=current_model)
            
            if improved_model is None:
                logger.error(f"Failed to improve model '{model_name}' during training.")
                return current_model, False
            
            # Evaluate model (optional, but recommended)
            eval_results = self._evaluate_model(improved_model, X, y)
            if eval_results:
                logger.info(f"Improved model '{model_name}' evaluation: {eval_results}")
            else:
                logger.warning(f"Failed to evaluate improved model '{model_name}'. Proceeding anyway.")

            # Save the improved model (overwrite or versioned)
            # Use the model_trainer's save method if it exists, otherwise use local save
            if hasattr(self.model_trainer, 'save_model'):
                success = self.model_trainer.save_model(improved_model, model_name)
            else:
                success = self._save_model_local(improved_model, model_name, model_dir)
            
            if not success:
                logger.error(f"Failed to save improved model '{model_name}'. Reverting to previous model.")
                return current_model, False
            
            # Save evaluation results (optional)
            if eval_results:
                results_path = os.path.join(model_dir, f"{model_name}_improvement_results.json")
                try:
                    with open(results_path, "w") as f:
                        json.dump(eval_results, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to save evaluation results for '{model_name}': {e}")
            
            logger.info(f"Successfully improved and saved model '{model_name}'.")
            
            # Clear feedback used for this improvement cycle
            self.feedback_collector.clear_feedback(model_name)
            
            return improved_model, True
        
        except Exception as e:
            logger.error(f"Error improving model '{model_name}': {e}", exc_info=True)
            return current_model, False  # Return original model on error
    
    def _evaluate_model(self, model, X, y):
        """Evaluate model performance."""
        try:
            # Ensure necessary libraries are imported locally if not globally
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Model prediction might need specific input format
            if hasattr(model, 'predict'):
                y_pred = model.predict(X)
            else:
                logger.warning("Model has no predict method for evaluation.")
                return None

            # Calculate metrics
            # Ensure y and y_pred are compatible (e.g., same shape, type)
            # Handle potential errors during metric calculation
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        except ImportError:
            logger.warning("Scikit-learn not available. Cannot evaluate improved model.")
            return None
        except Exception as e:
            logger.error(f"Error evaluating model: {e}", exc_info=True)
            return None
    
    def _save_model_local(self, model, model_name, model_dir):
        """Save model to file locally (used if trainer doesn't handle saving)."""
        # Determine the preferred extension (e.g., .pkl)
        # This should ideally match how the model was loaded or the trainer's format
        save_extension = ".pkl"  # Default to pickle
        model_path = os.path.join(model_dir, f"{model_name}{save_extension}")
        logger.info(f"Saving improved model '{model_name}' locally to {model_path}")
        try:
            os.makedirs(model_dir, exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            return True
        except Exception as e:
            logger.error(f"Error saving model '{model_name}' locally to {model_path}: {e}", exc_info=True)
            return False

class AdaptiveModel:
    """Class for adaptive model that improves over time"""
    
    def __init__(self, model_name, model_dir, feedback_dir=None):
        """Initialize the adaptive model"""
        self.model_name = model_name
        self.model_dir = model_dir
        self.feedback_dir = feedback_dir or os.path.join(model_dir, "feedback")
        os.makedirs(self.feedback_dir, exist_ok=True)  # Ensure feedback dir exists
        
        self._lock = threading.RLock()  # Lock for accessing/modifying self.model
        
        self.feedback_collector = ModelFeedbackCollector.get_instance()
        self.model_improver = ModelImprover(model_name, model_dir)
        
        self.model = None  # Initialize model as None
        self._load_model()  # Attempt to load the model during init
        
        # Improvement settings
        self.improvement_threshold = 10  # Min feedback entries
        self.improvement_interval = 3600  # Min time (seconds)
        self.last_improvement_time = 0
        
        self.stop_flag = AtomicBoolean(False)
        self.improvement_thread = None
        
        logger.info(f"Adaptive model initialized: {self.model_name}")
        # Load existing feedback if available
        self.feedback_collector.load_feedback(self.model_name, os.path.join(self.feedback_dir, f"{self.model_name}_feedback.json"))

    @synchronized  # Use the instance lock
    def _load_model(self):
        """Load the primary model file robustly, trying different extensions."""
        logger.info(f"Attempting to load model '{self.model_name}' from {self.model_dir}")
        loaded_successfully = False
        # Prioritize extensions if needed, e.g., try .h5 before .pkl
        potential_extensions = [".h5", ".pkl", ".joblib"]  # Add others as needed

        for ext in potential_extensions:
            model_path = os.path.join(self.model_dir, f"{self.model_name}{ext}")
            if os.path.exists(model_path):
                logger.info(f"Found model file: {model_path}")
                if ext in MODEL_LOADERS:
                    try:
                        loader_func = MODEL_LOADERS[ext]
                        self.model = loader_func(model_path)
                        logger.info(f"Successfully loaded model '{self.model_name}' from {model_path}")
                        loaded_successfully = True
                        break  # Stop after successful load
                    except Exception as e:
                        logger.error(f"Error loading model '{self.model_name}' from {model_path}: {e}", exc_info=True)
                        # Continue to try other extensions
                else:
                    logger.warning(f"Found model file {model_path} but no loader defined for extension '{ext}'.")
            # else: logger.debug(f"Model file not found: {model_path}")

        if not loaded_successfully:
            logger.error(f"Failed to load any model file for '{self.model_name}' in {self.model_dir}")
            self.model = None  # Ensure model is None if loading failed
        
        return self.model is not None  # Return status

    @track_performance("predict")
    def predict(self, features):
        """Make a prediction using the loaded model."""
        with self._lock:  # Ensure thread safety when accessing self.model
            if self.model is None:
                logger.error(f"Model '{self.model_name}' not loaded, cannot predict.")
                return "unknown", 0.0
            
            try:
                # Adapt prediction logic based on model type (e.g., sklearn, keras)
                if hasattr(self.model, "predict_proba"):  # Sklearn style
                    # Input shape might need adjustment (e.g., [features] or features.reshape(1, -1))
                    feature_array = np.array(features).reshape(1, -1)
                    y_proba = self.model.predict_proba(feature_array)[0]
                    prediction_idx = np.argmax(y_proba)
                    # Ensure classes_ attribute exists and prediction_idx is valid
                    if hasattr(self.model, 'classes_') and prediction_idx < len(self.model.classes_):
                        y_pred = self.model.classes_[prediction_idx]
                    else:
                        y_pred = str(prediction_idx)  # Fallback to index as string
                    confidence = np.max(y_proba)
                elif hasattr(self.model, "predict"):  # Keras/TF style or basic predict
                    # Input shape might need adjustment
                    feature_array = np.array(features).reshape(1, -1)  # Common case
                    pred_output = self.model.predict(feature_array)[0]
                    # Process pred_output - highly dependent on model output
                    # Example: Assume output is single confidence value for 'attack'
                    if isinstance(pred_output, (np.ndarray, list)) and len(pred_output) > 0:
                        confidence = float(pred_output[0])
                    else:
                        confidence = float(pred_output)  # Assume single scalar output
                    y_pred = "attack" if confidence > 0.5 else "normal"  # Example thresholding
                else:
                    logger.error(f"Model '{self.model_name}' has no recognized predict method.")
                    return "unknown", 0.0
                
                return y_pred, float(confidence)
            
            except Exception as e:
                logger.error(f"Error during prediction with model '{self.model_name}': {e}", exc_info=True)
                return "unknown", 0.0
    
    @track_performance("predict_with_feedback")
    def predict_with_feedback(self, features, actual=None):
        """Make a prediction and collect feedback."""
        prediction, confidence = self.predict(features)
        
        if actual is not None:
            # Add feedback asynchronously?
            self.feedback_collector.add_feedback(
                self.model_name, features, prediction, actual, confidence
            )
            # Trigger improvement check (could be async)
            self._check_improvement()
        
        return prediction, confidence
    
    def _check_improvement(self):
        """Check if model improvement is needed and trigger it."""
        # This check should be quick and non-blocking if called synchronously
        # The actual improvement runs in a separate thread or process
        feedback_count = len(self.feedback_collector.get_feedback(self.model_name))
        current_time = time.time()

        if feedback_count >= self.improvement_threshold and \
           current_time - self.last_improvement_time >= self.improvement_interval:
            
            logger.info(f"Improvement conditions met for model '{self.model_name}'. Triggering improvement.")
            # Ensure only one improvement runs at a time if needed
            # Run improvement in background thread
            improvement_task = threading.Thread(
                target=self._run_improvement_task,
                daemon=True
            )
            improvement_task.start()
            # Update time immediately to prevent rapid re-triggering
            self.last_improvement_time = current_time 
        # else: logger.debug("Improvement conditions not met.")

    def _run_improvement_task(self):
        """The actual task of improving the model."""
        with self._lock:  # Lock to prevent prediction while model is being replaced
            logger.info(f"Starting improvement task for model '{self.model_name}'.")
            current_model = self.model  # Get current model safely
            
        # Run the potentially long improvement process outside the main lock
        improved_model, success = self.model_improver.improve_model(
            self.model_name, current_model, self.model_dir
        )
        
        if success:
            with self._lock:  # Lock again to update the model reference
                self.model = improved_model
                logger.info(f"Successfully updated adaptive model '{self.model_name}' after improvement.")
                # Optionally save the updated model immediately
                # self._save_model_local(self.model, self.model_name, self.model_dir)
        else:
            logger.error(f"Improvement task failed for model '{self.model_name}'. Model remains unchanged.")

    def start_improvement_thread(self):
        """Start a thread for periodic checks for model improvement."""
        if self.improvement_thread and self.improvement_thread.is_alive():
            logger.warning(f"Improvement thread for '{self.model_name}' already running.")
            return False
            
        try:
            self.stop_flag.set(False)
            self.improvement_thread = SafeThread(
                target=self._improvement_loop,
                name=f"ImprovementLoop-{self.model_name}"
            )
            self.improvement_thread.daemon = True
            self.improvement_thread.start()
            logger.info(f"Started improvement loop thread for model {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Error starting improvement loop thread for '{self.model_name}': {e}", exc_info=True)
            return False
    
    def stop_improvement_thread(self):
        """Stop the improvement loop thread."""
        if not self.improvement_thread or not self.improvement_thread.is_alive():
            logger.info(f"Improvement loop thread for '{self.model_name}' is not running.")
            return True
            
        try:
            self.stop_flag.set(True)
            self.improvement_thread.join(timeout=5)  # Wait briefly for thread to exit
            if self.improvement_thread.is_alive():
                logger.warning(f"Improvement loop thread for '{self.model_name}' did not stop gracefully.")
            else:
                logger.info(f"Stopped improvement loop thread for model {self.model_name}")
            self.improvement_thread = None
            return True
        except Exception as e:
            logger.error(f"Error stopping improvement loop thread for '{self.model_name}': {e}", exc_info=True)
            return False
    
    def _improvement_loop(self):
        """The loop running in the background thread to check for improvements."""
        logger.info(f"Improvement loop started for '{self.model_name}'.")
        while not self.stop_flag.get():
            try:
                self._check_improvement()
                # Sleep for a configurable interval before next check
                check_interval = 60  # Check every 60 seconds
                for _ in range(check_interval):
                    if self.stop_flag.get(): 
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in improvement loop for '{self.model_name}': {e}", exc_info=True)
                # Avoid busy-looping on error
                time.sleep(60)
        logger.info(f"Improvement loop stopped for '{self.model_name}'.")

# --- Factory Functions --- 

# Singleton instance for Feedback Collector
_feedback_collector_instance = None
_feedback_collector_lock = threading.Lock()

def get_feedback_collector():
    """Get singleton instance of ModelFeedbackCollector"""
    global _feedback_collector_instance
    if _feedback_collector_instance is None:
        with _feedback_collector_lock:
            if _feedback_collector_instance is None:
                _feedback_collector_instance = ModelFeedbackCollector()
    return _feedback_collector_instance

# Adaptive models are usually specific, maybe not singleton
# Store instances in a dictionary if multiple adaptive models are needed
_adaptive_models = {}
_adaptive_models_lock = threading.Lock()

def get_adaptive_model(model_name, model_dir="models", feedback_dir=None):
    """Get or create an instance of AdaptiveModel."""
    with _adaptive_models_lock:
        if model_name not in _adaptive_models:
            logger.info(f"Creating new AdaptiveModel instance for '{model_name}'")
            _adaptive_models[model_name] = AdaptiveModel(model_name, model_dir, feedback_dir)
            # Optionally start the improvement thread automatically
            # _adaptive_models[model_name].start_improvement_thread()
        return _adaptive_models[model_name]

# Online Learning Manager (Optional High-Level Interface)
class OnlineLearningManager:
    """Manages online learning and model adaptation across multiple models."""
    def __init__(self, models_dir="models", profiles_dir="profiles"):
        self.models_dir = models_dir
        self.profiles_dir = profiles_dir
        self.feedback_collector = get_feedback_collector()
        # Potentially manage multiple adaptive models
        self.adaptive_models = {}  # Store instances managed here
        self._lock = threading.RLock()
        logger.info("Online Learning Manager initialized.")

    @synchronized
    def register_adaptive_model(self, model_name):
        """Ensure an adaptive model instance exists and is managed."""
        if model_name not in self.adaptive_models:
            model_instance = get_adaptive_model(model_name, self.models_dir)
            if model_instance:
                self.adaptive_models[model_name] = model_instance
                # Start its improvement thread if not already started
                if not model_instance.improvement_thread or not model_instance.improvement_thread.is_alive():
                    model_instance.start_improvement_thread()
                logger.info(f"Registered and started adaptive model: {model_name}")
            else:
                logger.error(f"Failed to get or create adaptive model instance for {model_name}")

    def process_packet_feedback(self, model_name, features, prediction, actual, confidence=None):
        """Provide feedback for a specific model."""
        # Ensure the model is registered for adaptation
        self.register_adaptive_model(model_name)
        # Add feedback using the collector
        self.feedback_collector.add_feedback(model_name, features, prediction, actual, confidence)
        # The adaptive model's internal loop will handle improvement checks

    def stop_all_improvement_threads(self):
        """Stop all managed improvement threads."""
        logger.info("Stopping all adaptive model improvement threads...")
        with _adaptive_models_lock:  # Use the lock from get_adaptive_model factory
            for model_name, model_instance in _adaptive_models.items():
                logger.debug(f"Stopping thread for {model_name}...")
                model_instance.stop_improvement_thread()
        logger.info("All adaptive model improvement threads stopped.")

    # Add methods for managing profiles, thresholds etc. if needed
    def get_adaptation_stats(self):
        # Example stats - needs more implementation based on requirements
        stats = {
            "adaptation_count": 0,  # How many times models were retrained
            "active_profile": "default",  # Placeholder
            "current_threshold": 0.5  # Placeholder
        }
        # Could query individual adaptive models for their stats
        return stats

_online_learning_manager_instance = None
_online_learning_manager_lock = threading.Lock()

def get_online_learning_manager(models_dir="models", profiles_dir="profiles"):
    """Get singleton instance of OnlineLearningManager."""
    global _online_learning_manager_instance
    if _online_learning_manager_instance is None:
        with _online_learning_manager_lock:
            if _online_learning_manager_instance is None:
                _online_learning_manager_instance = OnlineLearningManager(models_dir, profiles_dir)
    return _online_learning_manager_instance