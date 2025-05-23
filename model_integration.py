"""
Model integration module for AI-IDS
-------------------------------
This module provides model integration capabilities for the AI-IDS system.
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
from advanced_feature_engineering import get_feature_extractor
from self_improving_model import get_adaptive_model

# Get logger for this module
logger = get_logger("model_integration")

# Supported model extensions and their loading functions/libraries
# Add more as needed (e.g., PyTorch .pt/.pth)
MODEL_LOADERS = {
    '.pkl': lambda p: pickle.load(open(p, 'rb')),
    '.joblib': lambda p: __import__('joblib').load(p),
    '.h5': lambda p: __import__('tensorflow').keras.models.load_model(p),
    # '.json': lambda p: json.load(open(p, 'r')), # JSON usually stores config, not full model
    # Add loaders for .pt, .pth, .onnx etc. if needed
}

class ModelEnsemble:
    """Class for ensemble of multiple models"""
    
    def __init__(self, models_dir):
        """Initialize the model ensemble"""
        # Set models directory
        self.models_dir = models_dir
        logger.info(f"ModelEnsemble initialized with models_dir: {self.models_dir}")
        
        # Initialize models dictionary {model_name: {model_info}}
        self.models = {}
        
        # Initialize lock for thread safety
        self._lock = threading.RLock()
        
        # Initialize feature extractor
        self.feature_extractor = get_feature_extractor()
        
        logger.info("Model ensemble initialized")
    
    @synchronized
    def load_models(self, models_dir=None):
        """Load models from the specified directory."""
        if models_dir is None:
            models_dir = self.models_dir
        
        # Use the provided models_dir, ensure it's an absolute path if needed
        # The user noted D:\AI\1\models, but the code should use relative paths like 'models'
        # unless an absolute path is explicitly passed via args.
        # We'll assume models_dir is the correct path to use.
        logger.info(f"Attempting to load models from directory: {models_dir}")

        if not models_dir or not os.path.isdir(models_dir):
            logger.error(f"Models directory does not exist or is not a directory: {models_dir}")
            return False
            
        loaded_count = 0
        processed_basenames = set() # Keep track of model basenames already loaded

        # List files and prioritize certain extensions if needed
        # For now, process whatever is found
        try:
            files_in_dir = os.listdir(models_dir)
        except OSError as e:
            logger.error(f"Error listing files in models directory {models_dir}: {e}")
            return False

        for filename in files_in_dir:
            basename, ext = os.path.splitext(filename)
            
            # Skip if we already loaded a model with this basename
            if basename in processed_basenames:
                logger.debug(f"Skipping {filename}, model '{basename}' already processed.")
                continue

            # Check if the extension is supported
            if ext in MODEL_LOADERS:
                model_path = os.path.join(models_dir, filename)
                logger.info(f"Attempting to load model '{basename}' from {filename}...")
                try:
                    # Load the model using the appropriate function
                    loader_func = MODEL_LOADERS[ext]
                    model_obj = loader_func(model_path)
                    
                    # Store the loaded model
                    self.models[basename] = {
                        "model": model_obj,
                        "type": "loaded",
                        "extension": ext,
                        "weight": 1.0,
                        "path": model_path
                    }
                    
                    loaded_count += 1
                    processed_basenames.add(basename)
                    logger.info(f"Successfully loaded model: {basename} from {filename}")

                except pickle.UnpicklingError as pe:
                    logger.error(f"Failed to load pickle model {filename}: {pe}. File might be corrupted or incompatible.")
                    # Optionally try adaptive fallback
                    # if self.add_model(basename, "adaptive"): processed_basenames.add(basename)
                except ImportError as ie:
                     logger.error(f"Failed to load model {filename}: Missing library {ie}. Please install required dependencies.")
                except Exception as e:
                    logger.error(f"Failed to load model {filename}: {e}", exc_info=True)
                    # Optionally try adaptive fallback
                    # if self.add_model(basename, "adaptive"): processed_basenames.add(basename)
            
            # Handle scaler separately if named conventionally (e.g., scaler.pkl)
            elif basename == 'scaler' and ext in ['.pkl', '.joblib']:
                 model_path = os.path.join(models_dir, filename)
                 logger.info(f"Attempting to load scaler from {filename}...")
                 try:
                     if ext == '.pkl':
                         with open(model_path, 'rb') as f:
                             scaler_obj = pickle.load(f)
                     elif ext == '.joblib':
                         import joblib
                         scaler_obj = joblib.load(model_path)
                     
                     # Store the scaler (maybe separately or tagged)
                     self.models[basename] = {
                         "model": scaler_obj,
                         "type": "scaler",
                         "extension": ext,
                         "weight": 0, # Not used for prediction ensemble
                         "path": model_path
                     }
                     loaded_count += 1
                     processed_basenames.add(basename)
                     logger.info(f"Successfully loaded scaler: {basename} from {filename}")
                 except Exception as e:
                     logger.error(f"Failed to load scaler {filename}: {e}", exc_info=True)

            else:
                logger.debug(f"Skipping file with unsupported extension or unknown type: {filename}")

        # Check for models mentioned in logs but not loaded (e.g., RandomForest adaptive)
        # The logs suggest RandomForest might be intended as an adaptive model
        if 'RandomForest' not in processed_basenames:
             logger.info("Attempting to add RandomForest as an adaptive model based on logs.")
             if self.add_model('RandomForest', "adaptive"): 
                 loaded_count += 1 # Count adaptive models if added successfully
                 processed_basenames.add('RandomForest')

        if loaded_count > 0:
            logger.info(f"Finished loading models. Total loaded/added: {loaded_count} from {models_dir}")
            return True
        else:
            logger.warning(f"No models were successfully loaded from {models_dir}")
            return False
            
    
    @synchronized
    def add_model(self, model_name, model_type):
        """Add a model to the ensemble"""
        try:
            # Check if model already exists
            if model_name in self.models:
                logger.warning(f"Model '{model_name}' already exists in ensemble. Skipping add.")
                return False
            
            # Create model
            if model_type == "adaptive":
                # Ensure models_dir is passed correctly
                model = get_adaptive_model(model_name, self.models_dir)
                if model is None:
                    logger.error(f"Failed to initialize adaptive model '{model_name}'")
                    return False
            else:
                logger.error(f"Unsupported model type for dynamic addition: {model_type}")
                return False
            
            # Add model
            self.models[model_name] = {
                "model": model,
                "type": model_type,
                "weight": 1.0, # Default weight
                "path": None # No specific file path for adaptive models initially
            }
            
            logger.info(f"Added model to ensemble: {model_name} ({model_type})")
            return True
        
        except Exception as e:
            logger.error(f"Error adding model '{model_name}': {e}", exc_info=True)
            return False
    
    @synchronized
    def remove_model(self, model_name):
        """Remove a model from the ensemble"""
        try:
            # Check if model exists
            if model_name not in self.models:
                logger.warning(f"Model not found for removal: {model_name}")
                return False
            
            # Remove model
            del self.models[model_name]
            
            logger.info(f"Removed model from ensemble: {model_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error removing model '{model_name}': {e}", exc_info=True)
            return False
    
    @synchronized
    def set_model_weight(self, model_name, weight):
        """Set the weight of a model"""
        try:
            # Check if model exists
            if model_name not in self.models:
                logger.warning(f"Model not found for setting weight: {model_name}")
                return False
            
            # Set weight
            self.models[model_name]["weight"] = max(0.0, min(1.0, float(weight)))
            
            logger.info(f"Set model weight: {model_name} = {self.models[model_name]['weight']}")
            return True
        
        except Exception as e:
            logger.error(f"Error setting model weight for '{model_name}': {e}", exc_info=True)
            return False
    
    @synchronized # Ensure thread safety for accessing self.models
    def get_model_names(self):
        """Return a list of names of the loaded models."""
        return list(self.models.keys())

    @synchronized
    def get_model_info(self, model_name):
        """Get information about a specific model."""
        if model_name in self.models:
            # Return a copy of the info, excluding the actual model object
            info = self.models[model_name].copy()
            info.pop('model', None) # Remove the potentially large model object
            return info
        else:
            logger.warning(f"Requested info for non-existent model: {model_name}")
            return None

    @track_performance("predict")
    def predict(self, features):
        """Make a prediction using the ensemble"""
        # This method needs access to self.models, ensure synchronization if modifying
        with self._lock: # Use the instance lock for reading self.models
            if not self.models:
                logger.warning("Prediction requested but no models are loaded in the ensemble.")
                # Return a default prediction or raise an error
                return "unknown", 0.0 # Default prediction
            
            # Make predictions
            predictions = {}
            total_weight = 0.0
            
            for model_name, model_info in self.models.items():
                # Skip non-predictive models like scalers
                if model_info.get("type") == "scaler":
                    continue
                    
                model = model_info.get("model")
                weight = model_info.get("weight", 1.0)
                
                if model is None or weight <= 0:
                    continue

                try:
                    # Assuming model has a predict method returning (prediction, confidence)
                    # Adapt this if models have different prediction signatures
                    if hasattr(model, 'predict_proba'): # Scikit-learn style
                        proba = model.predict_proba(features)[0] # Assuming binary or multi-class
                        # Determine prediction based on proba (e.g., highest proba class)
                        # This needs classes_ attribute or similar mapping
                        # Simplified: assume binary normal/attack, confidence is attack proba
                        # This part needs refinement based on actual model outputs
                        if len(proba) > 1:
                             confidence = proba[1] # Confidence of being class '1' (attack)
                             prediction = "attack" if confidence > 0.5 else "normal" # Example threshold
                        else:
                             confidence = proba[0]
                             prediction = "unknown" # Or map based on single class output
                    elif hasattr(model, 'predict'): # Keras/TF style or simple predict
                        pred_output = model.predict(features)
                        # Process pred_output to get prediction label and confidence
                        # This is highly dependent on the model's output format
                        # Simplified example: assume output is confidence of attack
                        if isinstance(pred_output, (np.ndarray, list)):
                            confidence = float(pred_output[0]) # Assuming single output
                        else:
                            confidence = float(pred_output)
                        prediction = "attack" if confidence > 0.5 else "normal" # Example threshold
                    else:
                        logger.warning(f"Model '{model_name}' has no standard predict method. Skipping.")
                        continue

                    # Store weighted prediction score
                    # Simple averaging based on confidence * weight
                    if prediction not in predictions:
                        predictions[prediction] = 0.0
                    predictions[prediction] += confidence * weight
                    total_weight += weight

                except Exception as e:
                    logger.error(f"Error during prediction with model '{model_name}': {e}", exc_info=True)
                    continue # Skip this model on error

            # Determine final ensemble prediction
            if not predictions or total_weight == 0:
                logger.warning("No valid predictions from ensemble models.")
                return "unknown", 0.0

            # Example: Weighted average or majority vote
            # Weighted average approach (simplified): highest weighted score wins
            best_prediction = max(predictions, key=predictions.get)
            # Calculate a combined confidence (e.g., normalized score)
            best_score = predictions[best_prediction]
            combined_confidence = best_score / total_weight if total_weight > 0 else 0.0
            
            # Ensure confidence is between 0 and 1
            combined_confidence = max(0.0, min(1.0, combined_confidence))

            return best_prediction, combined_confidence
        
    @track_performance("predict_with_feedback")
    def predict_with_feedback(self, features, actual=None):
        """Make a prediction and provide feedback to adaptive models"""
        # Make prediction first
        prediction, confidence = self.predict(features)
        
        # Provide feedback if actual label is given
        if actual is not None:
            with self._lock: # Lock for iterating/modifying models if feedback updates them
                for model_name, model_info in self.models.items():
                    model = model_info.get("model")
                    # Check if model is adaptive and supports feedback
                    if model_info.get("type") == "adaptive" and hasattr(model, "predict_with_feedback"):
                        try:
                            model.predict_with_feedback(features, actual)
                        except Exception as e:
                            logger.error(f"Error providing feedback to adaptive model '{model_name}': {e}", exc_info=True)
        
        return prediction, confidence

class DetectionEngine:
    """Class for detecting threats using models"""
    
    def __init__(self, models_dir):
        """Initialize the detection engine"""
        # Set models directory
        self.models_dir = models_dir
        logger.info(f"DetectionEngine initializing with models_dir: {self.models_dir}")
        
        # Initialize model ensemble - pass models_dir
        self.model_ensemble = ModelEnsemble(self.models_dir)
        # Load models immediately upon initialization
        self.model_ensemble.load_models()
        
        # Initialize feature extractor
        self.feature_extractor = get_feature_extractor()
        
        # Initialize detection history (consider using a deque for efficiency)
        self.detection_history = [] 
        self.max_history = 1000
        
        # Initialize lock for thread safety on history
        self._history_lock = threading.RLock()
        
        logger.info("Detection engine initialized")
    
    @track_performance("detect")
    def detect(self, packet):
        """Detect threats in a packet"""
        try:
            # Extract features
            # Ensure packet is in the expected format (e.g., dictionary)
            features = self.feature_extractor.extract_features(packet)
            
            if features is None:
                # logger.debug("Failed to extract features from packet.") # Can be noisy
                return None
            
            # Preprocess features (e.g., scaling, encoding)
            # This step might depend on the models used (e.g., scaler)
            scaler = self.model_ensemble.models.get('scaler', {}).get('model')
            if scaler and hasattr(scaler, 'transform'):
                 # Reshape features if necessary for the scaler
                 # Assuming features is a 1D array/list and scaler expects 2D
                 try:
                     feature_array = np.array(features).reshape(1, -1)
                     processed_features = scaler.transform(feature_array)
                 except Exception as scale_e:
                     logger.error(f"Error scaling features: {scale_e}")
                     processed_features = features # Fallback to unscaled features?
            else:
                 processed_features = features # No scaler found or needed
            
            if processed_features is None:
                logger.error("Failed to preprocess features")
                return None
            
            # Make prediction using the ensemble
            prediction, confidence = self.model_ensemble.predict(processed_features)
            
            # Create detection result
            result = {
                "timestamp": time.time(),
                # "packet": packet, # Avoid storing full packet in history unless needed
                "src_ip": packet.get('src_ip', 'N/A'), # Example: store key fields
                "dst_ip": packet.get('dst_ip', 'N/A'),
                "protocol": packet.get('protocol', 'N/A'),
                # "features": features, # Avoid storing features unless needed
                "prediction": prediction,
                "confidence": float(confidence), # Ensure float
                "is_alert": prediction != "normal" # Define what constitutes an alert
            }
            
            # Add to history
            self._add_to_history(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error detecting threats: {e}", exc_info=True)
            return None
    
    def _add_to_history(self, result):
        """Add a detection result to history (thread-safe)."""
        with self._history_lock:
            try:
                self.detection_history.append(result)
                # Trim history using slicing (more efficient than repeated pop)
                if len(self.detection_history) > self.max_history:
                    self.detection_history = self.detection_history[-self.max_history:]
                return True
            except Exception as e:
                logger.error(f"Error adding to history: {e}", exc_info=True)
                return False
    
    def get_detection_history(self, limit=None):
        """Get detection history (thread-safe)."""
        with self._history_lock:
            try:
                # Return a copy to prevent modification issues outside the lock
                history = list(self.detection_history)
                if limit is not None and limit > 0:
                    history = history[-limit:]
                return history
            except Exception as e:
                logger.error(f"Error getting detection history: {e}", exc_info=True)
                return []
    
    def clear_detection_history(self):
        """Clear detection history (thread-safe)."""
        with self._history_lock:
            try:
                self.detection_history = []
                logger.info("Cleared detection history")
                return True
            except Exception as e:
                logger.error(f"Error clearing detection history: {e}", exc_info=True)
                return False
    
    @track_performance("provide_feedback")
    def provide_feedback(self, detection_index, actual):
        """Provide feedback for a detection based on its index in the history."""
        # Note: Using index might be fragile if history is cleared/modified concurrently
        # Consider using a unique ID per detection if feedback is critical
        with self._history_lock: # Lock history while accessing by index
            try:
                if detection_index < 0 or detection_index >= len(self.detection_history):
                    logger.warning(f"Detection index out of bounds: {detection_index}")
                    return False
                
                # Get detection (we stored limited info, might not have full features)
                detection = self.detection_history[detection_index]
                
                # Feedback requires features. If not stored, we can't provide feedback this way.
                # This design assumes features ARE available or feedback uses different mechanism.
                # For now, assume features are somehow retrievable or feedback is handled elsewhere.
                logger.warning("Feedback by index requires features stored in history, which is currently disabled.")
                # Placeholder for actual feedback logic if features were available:
                # features = detection.get("features") 
                # if features:
                #     processed_features = self.feature_extractor.preprocess_features(features)
                #     if processed_features:
                #         self.model_ensemble.predict_with_feedback(processed_features, actual)
                #         detection["actual"] = actual
                #         detection["feedback_time"] = time.time()
                #         logger.info(f"Provided feedback for detection {detection_index}: {actual}")
                #         return True
                # return False # Features not available
                return False # Temporarily disabled due to missing features in history
            
            except Exception as e:
                logger.error(f"Error providing feedback for index {detection_index}: {e}", exc_info=True)
                return False

class ModelIntegration:
    """Class for integrating models with packet processing"""
    
    def __init__(self, models_dir):
        """Initialize the model integration"""
        self.models_dir = models_dir
        logger.info(f"ModelIntegration initializing with models_dir: {self.models_dir}")
        self.detection_engine = DetectionEngine(self.models_dir)
        self.feature_extractor = get_feature_extractor()
        self._lock = threading.RLock() # Lock for stats if needed
        self.packet_count = 0
        self.alert_count = 0
        logger.info("Model integration initialized")
    
    @track_performance("process_packet")
    def process_packet(self, packet):
        """Process a packet and return detection results"""
        try:
            result = self.detection_engine.detect(packet)
            if result:
                with self._lock:
                    self.packet_count += 1
                    if result.get("is_alert", False):
                        self.alert_count += 1
            return result
        except Exception as e:
            logger.error(f"Error processing packet: {e}", exc_info=True)
            return None
    
    @synchronized
    def get_stats(self):
        """Get processing statistics"""
        # This provides simple counts. More detailed stats could come from history.
        return {
            "total_packets_processed": self.packet_count,
            "total_alerts_generated": self.alert_count,
            "history_length": len(self.detection_engine.get_detection_history()) # Get current history size
        }

# --- Factory Functions --- 

_model_ensemble_instance = None
_model_ensemble_lock = threading.Lock()

def get_model_ensemble(models_dir="models"):
    """Get singleton instance of ModelEnsemble"""
    global _model_ensemble_instance
    if _model_ensemble_instance is None:
        with _model_ensemble_lock:
            if _model_ensemble_instance is None:
                _model_ensemble_instance = ModelEnsemble(models_dir)
    return _model_ensemble_instance

_detection_engine_instance = None
_detection_engine_lock = threading.Lock()

def get_detection_engine(models_dir="models"):
    """Get singleton instance of DetectionEngine"""
    global _detection_engine_instance
    if _detection_engine_instance is None:
        with _detection_engine_lock:
            if _detection_engine_instance is None:
                _detection_engine_instance = DetectionEngine(models_dir)
    return _detection_engine_instance

_model_integration_instance = None
_model_integration_lock = threading.Lock()

def get_model_integration(models_dir="models"):
    """Get singleton instance of ModelIntegration"""
    global _model_integration_instance
    if _model_integration_instance is None:
        with _model_integration_lock:
            if _model_integration_instance is None:
                _model_integration_instance = ModelIntegration(models_dir)
    return _model_integration_instance