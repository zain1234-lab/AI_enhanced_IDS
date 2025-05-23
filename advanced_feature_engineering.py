"""
Advanced feature engineering for AI-IDS
-------------------------------------
This module provides robust feature extraction and preprocessing for network traffic analysis.
"""

import os
import sys
import time
import json
import ipaddress
import numpy as np
import pandas as pd
import logging
import threading
from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from enhanced_logging import get_logger, track_performance
from thread_safety import SafeDict, AtomicBoolean, synchronized
from resource_monitor import track_operation, limit_memory_usage

# Get logger for this module
logger = get_logger('advanced_feature_engineering')

class FeatureExtractor:
    """Class for extracting features from network packets"""
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance with thread safety"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        """Initialize the feature extractor"""
        # Initialize flow tracking
        self.flows = {}
        self.flow_timeout = 60  # seconds
        
        # Initialize feature definitions
        self.feature_definitions = {
            'basic': [
                'src_ip_num',
                'dst_ip_num',
                'src_port',
                'dst_port',
                'protocol_tcp',
                'protocol_udp',
                'protocol_icmp',
                'packet_size',
                'has_payload'
            ],
            'flow': [
                'flow_duration',
                'flow_packets',
                'flow_bytes',
                'flow_rate_pps',
                'flow_rate_bps',
                'flow_packet_size_mean',
                'flow_packet_size_std',
                'flow_packet_size_min',
                'flow_packet_size_max'
            ],
            'time': [
                'time_hour',
                'time_minute',
                'time_second',
                'time_day_of_week',
                'time_is_weekend'
            ]
        }
        
        # Initialize feature types
        self.feature_types = {
            'numeric': [
                'src_port',
                'dst_port',
                'packet_size',
                'flow_duration',
                'flow_packets',
                'flow_bytes',
                'flow_rate_pps',
                'flow_rate_bps',
                'flow_packet_size_mean',
                'flow_packet_size_std',
                'flow_packet_size_min',
                'flow_packet_size_max',
                'time_hour',
                'time_minute',
                'time_second',
                'time_day_of_week'
            ],
            'categorical': [
                'protocol_tcp',
                'protocol_udp',
                'protocol_icmp',
                'has_payload',
                'time_is_weekend'
            ],
            'ip_address': [
                'src_ip_num',
                'dst_ip_num'
            ]
        }
        
        # Initialize feature scalers
        self.scalers = {}
        
        # Initialize feature encoders
        self.encoders = {}
        
        # Initialize locks
        self._flow_lock = threading.RLock()
        
        logger.info("Feature extractor initialized")
    
    @track_performance("extract_features")
    def extract_features(self, packet):
        """Extract features from a packet"""
        try:
            # Extract basic features
            features = self._extract_basic_features(packet)
            
            # Extract flow features
            flow_features = self._extract_flow_features(packet)
            features.update(flow_features)
            
            # Extract time features
            time_features = self._extract_time_features(packet)
            features.update(time_features)
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def _extract_basic_features(self, packet):
        """Extract basic features from a packet"""
        try:
            features = {}
            
            # Extract IP addresses
            if 'src_ip' in packet:
                try:
                    ip = ipaddress.ip_address(packet['src_ip'])
                    features['src_ip_num'] = int(ip)
                except:
                    features['src_ip_num'] = 0
            else:
                features['src_ip_num'] = 0
            
            if 'dst_ip' in packet:
                try:
                    ip = ipaddress.ip_address(packet['dst_ip'])
                    features['dst_ip_num'] = int(ip)
                except:
                    features['dst_ip_num'] = 0
            else:
                features['dst_ip_num'] = 0
            
            # Extract ports
            features['src_port'] = packet.get('src_port', 0)
            features['dst_port'] = packet.get('dst_port', 0)
            
            # Extract protocol
            protocol = packet.get('protocol', '').lower()
            features['protocol_tcp'] = 1 if protocol == 'tcp' else 0
            features['protocol_udp'] = 1 if protocol == 'udp' else 0
            features['protocol_icmp'] = 1 if protocol == 'icmp' else 0
            
            # Extract packet size
            features['packet_size'] = packet.get('packet_size', 0)
            
            # Extract payload information
            features['has_payload'] = 1 if packet.get('data') else 0
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting basic features: {e}")
            return {}
    
    def _extract_flow_features(self, packet):
        """Extract flow features from a packet"""
        try:
            features = {}
            
            # Create flow key
            flow_key = self._create_flow_key(packet)
            
            # Skip if no flow key
            if not flow_key:
                return features
            
            with self._flow_lock:
                # Get current time
                current_time = time.time()
                
                # Clean up expired flows
                self._cleanup_flows(current_time)
                
                # Create or update flow
                if flow_key not in self.flows:
                    # Create new flow
                    self.flows[flow_key] = {
                        'start_time': current_time,
                        'last_time': current_time,
                        'packets': 1,
                        'bytes': packet.get('packet_size', 0),
                        'packet_sizes': [packet.get('packet_size', 0)]
                    }
                else:
                    # Update existing flow
                    flow = self.flows[flow_key]
                    flow['last_time'] = current_time
                    flow['packets'] += 1
                    flow['bytes'] += packet.get('packet_size', 0)
                    flow['packet_sizes'].append(packet.get('packet_size', 0))
                
                # Get flow
                flow = self.flows[flow_key]
                
                # Calculate flow features
                flow_duration = flow['last_time'] - flow['start_time']
                flow_duration = max(0.001, flow_duration)  # Avoid division by zero
                
                features['flow_duration'] = flow_duration
                features['flow_packets'] = flow['packets']
                features['flow_bytes'] = flow['bytes']
                features['flow_rate_pps'] = flow['packets'] / flow_duration
                features['flow_rate_bps'] = flow['bytes'] / flow_duration
                
                # Calculate packet size statistics
                packet_sizes = flow['packet_sizes']
                features['flow_packet_size_mean'] = np.mean(packet_sizes)
                features['flow_packet_size_std'] = np.std(packet_sizes) if len(packet_sizes) > 1 else 0
                features['flow_packet_size_min'] = np.min(packet_sizes)
                features['flow_packet_size_max'] = np.max(packet_sizes)
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting flow features: {e}")
            return {}
    
    def _extract_time_features(self, packet):
        """Extract time features from a packet"""
        try:
            features = {}
            
            # Get timestamp
            timestamp = packet.get('timestamp', time.time())
            
            # Convert to datetime
            dt = datetime.fromtimestamp(timestamp)
            
            # Extract time features
            features['time_hour'] = dt.hour
            features['time_minute'] = dt.minute
            features['time_second'] = dt.second
            features['time_day_of_week'] = dt.weekday()
            features['time_is_weekend'] = 1 if dt.weekday() >= 5 else 0
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting time features: {e}")
            return {}
    
    def _create_flow_key(self, packet):
        """Create a flow key from a packet"""
        try:
            # Check if packet has required fields
            if 'src_ip' not in packet or 'dst_ip' not in packet:
                return None
            
            # Get fields
            src_ip = packet.get('src_ip')
            dst_ip = packet.get('dst_ip')
            src_port = packet.get('src_port', 0)
            dst_port = packet.get('dst_port', 0)
            protocol = packet.get('protocol', '').lower()
            
            # Create bidirectional flow key
            if f"{src_ip}:{src_port}" < f"{dst_ip}:{dst_port}":
                return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
            else:
                return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        
        except Exception as e:
            logger.error(f"Error creating flow key: {e}")
            return None
    
    def _cleanup_flows(self, current_time):
        """Clean up expired flows"""
        try:
            # Find expired flows
            expired_keys = []
            
            for key, flow in self.flows.items():
                if current_time - flow['last_time'] > self.flow_timeout:
                    expired_keys.append(key)
            
            # Remove expired flows
            for key in expired_keys:
                del self.flows[key]
            
            return True
        
        except Exception as e:
            logger.error(f"Error cleaning up flows: {e}")
            return False
    
    @track_performance("preprocess_features")
    def preprocess_features(self, features_dict):
        """Preprocess features for model input"""
        try:
            # Convert to DataFrame
            features_df = pd.DataFrame([features_dict])
            
            # Preprocess features
            return self._preprocess_dataframe(features_df)
        
        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            return None
    
    @track_performance("preprocess_dataframe")
    def preprocess_dataframe(self, df):
        """Preprocess a DataFrame of features"""
        try:
            # Make a copy of the DataFrame
            df = df.copy()
            
            # Preprocess features
            return self._preprocess_dataframe(df)
        
        except Exception as e:
            logger.error(f"Error preprocessing DataFrame: {e}")
            return None
    
    def _preprocess_dataframe(self, df):
        """Preprocess a DataFrame of features"""
        try:
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Handle IP addresses
            df = self._handle_ip_addresses(df)
            
            # Handle categorical features
            df = self._handle_categorical_features(df)
            
            # Scale numeric features
            df = self._scale_numeric_features(df)
            
            return df
        
        except Exception as e:
            logger.error(f"Error in _preprocess_dataframe: {e}")
            return None
    
    def _handle_missing_values(self, df):
        """Handle missing values in a DataFrame"""
        try:
            # Create imputer if not exists
            if 'imputer' not in self.scalers:
                self.scalers['imputer'] = SimpleImputer(strategy='mean')
                
                # Fit imputer on numeric columns
                numeric_cols = [col for col in df.columns if col in self.feature_types['numeric']]
                if numeric_cols:
                    self.scalers['imputer'].fit(df[numeric_cols])
            
            # Impute missing values in numeric columns
            numeric_cols = [col for col in df.columns if col in self.feature_types['numeric']]
            if numeric_cols:
                df[numeric_cols] = self.scalers['imputer'].transform(df[numeric_cols])
            
            # Fill missing values in categorical columns
            categorical_cols = [col for col in df.columns if col in self.feature_types['categorical']]
            if categorical_cols:
                df[categorical_cols] = df[categorical_cols].fillna(0)
            
            # Fill missing values in IP address columns
            ip_cols = [col for col in df.columns if col in self.feature_types['ip_address']]
            if ip_cols:
                df[ip_cols] = df[ip_cols].fillna(0)
            
            return df
        
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return df
    
    def _handle_ip_addresses(self, df):
        """Handle IP address features in a DataFrame"""
        try:
            # Scale IP addresses
            ip_cols = [col for col in df.columns if col in self.feature_types['ip_address']]
            
            if ip_cols:
                # Create scaler if not exists
                if 'ip_scaler' not in self.scalers:
                    self.scalers['ip_scaler'] = MinMaxScaler()
                    self.scalers['ip_scaler'].fit(df[ip_cols])
                
                # Scale IP addresses
                df[ip_cols] = self.scalers['ip_scaler'].transform(df[ip_cols])
            
            return df
        
        except Exception as e:
            logger.error(f"Error handling IP addresses: {e}")
            return df
    
    def _handle_categorical_features(self, df):
        """Handle categorical features in a DataFrame"""
        try:
            # Get categorical columns
            categorical_cols = [col for col in df.columns if col in self.feature_types['categorical']]
            
            if categorical_cols:
                # Create encoder if not exists
                if 'categorical_encoder' not in self.encoders:
                    self.encoders['categorical_encoder'] = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    self.encoders['categorical_encoder'].fit(df[categorical_cols])
                
                # Encode categorical features
                encoded = self.encoders['categorical_encoder'].transform(df[categorical_cols])
                
                # Get feature names
                feature_names = []
                for i, col in enumerate(categorical_cols):
                    for j in range(len(self.encoders['categorical_encoder'].categories_[i])):
                        feature_names.append(f"{col}_{j}")
                
                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                
                # Drop original categorical columns
                df = df.drop(columns=categorical_cols)
                
                # Concatenate encoded features
                df = pd.concat([df, encoded_df], axis=1)
            
            return df
        
        except Exception as e:
            logger.error(f"Error handling categorical features: {e}")
            return df
    
    def _scale_numeric_features(self, df):
        """Scale numeric features in a DataFrame"""
        try:
            # Get numeric columns
            numeric_cols = [col for col in df.columns if col in self.feature_types['numeric']]
            
            if numeric_cols:
                # Create scaler if not exists
                if 'numeric_scaler' not in self.scalers:
                    self.scalers['numeric_scaler'] = StandardScaler()
                    self.scalers['numeric_scaler'].fit(df[numeric_cols])
                
                # Scale numeric features
                df[numeric_cols] = self.scalers['numeric_scaler'].transform(df[numeric_cols])
            
            return df
        
        except Exception as e:
            logger.error(f"Error scaling numeric features: {e}")
            return df
    
    def fit_preprocessor(self, df):
        """Fit preprocessor on a DataFrame"""
        try:
            # Reset scalers and encoders
            self.scalers = {}
            self.encoders = {}
            
            # Fit imputer
            numeric_cols = [col for col in df.columns if col in self.feature_types['numeric']]
            if numeric_cols:
                self.scalers['imputer'] = SimpleImputer(strategy='mean')
                self.scalers['imputer'].fit(df[numeric_cols])
            
            # Fit IP address scaler
            ip_cols = [col for col in df.columns if col in self.feature_types['ip_address']]
            if ip_cols:
                self.scalers['ip_scaler'] = MinMaxScaler()
                self.scalers['ip_scaler'].fit(df[ip_cols])
            
            # Fit categorical encoder
            categorical_cols = [col for col in df.columns if col in self.feature_types['categorical']]
            if categorical_cols:
                self.encoders['categorical_encoder'] = OneHotEncoder(sparse=False, handle_unknown='ignore')
                self.encoders['categorical_encoder'].fit(df[categorical_cols])
            
            # Fit numeric scaler
            if numeric_cols:
                self.scalers['numeric_scaler'] = StandardScaler()
                self.scalers['numeric_scaler'].fit(df[numeric_cols])
            
            logger.info(f"Fitted preprocessor on DataFrame with {len(df)} rows")
            return True
        
        except Exception as e:
            logger.error(f"Error fitting preprocessor: {e}")
            return False
    
    def save_preprocessor(self, file_path):
        """Save preprocessor to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save preprocessor
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'scalers': self.scalers,
                    'encoders': self.encoders,
                    'feature_definitions': self.feature_definitions,
                    'feature_types': self.feature_types
                }, f)
            
            logger.info(f"Saved preprocessor to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving preprocessor: {e}")
            return False
    
    def load_preprocessor(self, file_path):
        """Load preprocessor from file"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Preprocessor file not found: {file_path}")
                return False
            
            # Load preprocessor
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
                self.scalers = data['scalers']
                self.encoders = data['encoders']
                self.feature_definitions = data['feature_definitions']
                self.feature_types = data['feature_types']
            
            logger.info(f"Loaded preprocessor from {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")
            return False

class FeatureSelector:
    """Class for selecting features"""
    
    def __init__(self):
        """Initialize the feature selector"""
        # Initialize feature importance
        self.feature_importance = {}
        
        # Initialize selected features
        self.selected_features = []
        
        logger.info("Feature selector initialized")
    
    def select_features(self, X, y, method='importance', n_features=None):
        """Select features from a DataFrame"""
        try:
            # Select features based on method
            if method == 'importance':
                return self._select_by_importance(X, y, n_features)
            elif method == 'correlation':
                return self._select_by_correlation(X, y, n_features)
            elif method == 'variance':
                return self._select_by_variance(X, n_features)
            else:
                logger.error(f"Unknown feature selection method: {method}")
                return X.columns.tolist()
        
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return X.columns.tolist()
    
    def _select_by_importance(self, X, y, n_features=None):
        """Select features by importance"""
        try:
            # Train a model to get feature importance
            from sklearn.ensemble import RandomForestClassifier
            
            # Create model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Fit model
            model.fit(X, y)
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Create feature importance dictionary
            self.feature_importance = {col: importance[i] for i, col in enumerate(X.columns)}
            
            # Sort features by importance
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Select top n features
            if n_features is not None:
                sorted_features = sorted_features[:n_features]
            
            # Get selected feature names
            self.selected_features = [feature for feature, _ in sorted_features]
            
            return self.selected_features
        
        except Exception as e:
            logger.error(f"Error selecting features by importance: {e}")
            return X.columns.tolist()
    
    def _select_by_correlation(self, X, y, n_features=None):
        """Select features by correlation with target"""
        try:
            # Calculate correlation with target
            df = X.copy()
            df['target'] = y
            
            # Calculate correlation
            correlation = df.corr()['target'].abs()
            
            # Remove target
            correlation = correlation.drop('target')
            
            # Sort features by correlation
            sorted_features = correlation.sort_values(ascending=False)
            
            # Create feature importance dictionary
            self.feature_importance = sorted_features.to_dict()
            
            # Select top n features
            if n_features is not None:
                sorted_features = sorted_features[:n_features]
            
            # Get selected feature names
            self.selected_features = sorted_features.index.tolist()
            
            return self.selected_features
        
        except Exception as e:
            logger.error(f"Error selecting features by correlation: {e}")
            return X.columns.tolist()
    
    def _select_by_variance(self, X, n_features=None):
        """Select features by variance"""
        try:
            # Calculate variance
            variance = X.var()
            
            # Sort features by variance
            sorted_features = variance.sort_values(ascending=False)
            
            # Create feature importance dictionary
            self.feature_importance = sorted_features.to_dict()
            
            # Select top n features
            if n_features is not None:
                sorted_features = sorted_features[:n_features]
            
            # Get selected feature names
            self.selected_features = sorted_features.index.tolist()
            
            return self.selected_features
        
        except Exception as e:
            logger.error(f"Error selecting features by variance: {e}")
            return X.columns.tolist()
    
    def get_feature_importance(self):
        """Get feature importance"""
        return self.feature_importance
    
    def get_selected_features(self):
        """Get selected features"""
        return self.selected_features

class FeatureGenerator:
    """Class for generating new features"""
    
    def __init__(self):
        """Initialize the feature generator"""
        # Initialize feature definitions
        self.feature_definitions = {}
        
        logger.info("Feature generator initialized")
    
    def generate_features(self, df):
        """Generate new features from a DataFrame"""
        try:
            # Make a copy of the DataFrame
            df = df.copy()
            
            # Generate polynomial features
            df = self._generate_polynomial_features(df)
            
            # Generate interaction features
            df = self._generate_interaction_features(df)
            
            # Generate time-based features
            df = self._generate_time_features(df)
            
            return df
        
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            return df
    
    def _generate_polynomial_features(self, df):
        """Generate polynomial features"""
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Generate squared features
            for col in numeric_cols:
                df[f"{col}_squared"] = df[col] ** 2
            
            # Generate cubic features
            for col in numeric_cols:
                df[f"{col}_cubed"] = df[col] ** 3
            
            return df
        
        except Exception as e:
            logger.error(f"Error generating polynomial features: {e}")
            return df
    
    def _generate_interaction_features(self, df):
        """Generate interaction features"""
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # Generate interaction features
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
            
            return df
        
        except Exception as e:
            logger.error(f"Error generating interaction features: {e}")
            return df
    
    def _generate_time_features(self, df):
        """Generate time-based features"""
        try:
            # Check if time columns exist
            time_cols = [col for col in df.columns if col.startswith('time_')]
            
            if not time_cols:
                return df
            
            # Generate time of day feature
            if 'time_hour' in df.columns:
                df['time_of_day'] = df['time_hour'].apply(lambda x: 0 if 0 <= x < 6 else (1 if 6 <= x < 12 else (2 if 12 <= x < 18 else 3)))
            
            # Generate time period feature
            if 'time_hour' in df.columns:
                df['time_period'] = df['time_hour'].apply(lambda x: 0 if 0 <= x < 8 else (1 if 8 <= x < 16 else 2))
            
            return df
        
        except Exception as e:
            logger.error(f"Error generating time features: {e}")
            return df

# Singleton getter
def get_feature_extractor():
    """Get feature extractor singleton"""
    return FeatureExtractor.get_instance()

def get_feature_selector():
    """Get feature selector instance"""
    return FeatureSelector()

def get_feature_generator():
    """Get feature generator instance"""
    return FeatureGenerator()

# Initialize feature extractor
feature_extractor = get_feature_extractor()
