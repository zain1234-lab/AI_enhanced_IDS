"""
Enhanced logging system for AI-IDS
---------------------------------
This module provides thread-safe logging with context tracking and performance monitoring.
"""

import os
import sys
import time
import json
import logging
import threading
import functools
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Global context storage with thread safety
_context_storage = threading.local()

# Global logger cache with lock
_logger_cache = {}
_logger_lock = threading.RLock()

class LoggingManager:
    """Class for managing logging configuration and handlers"""
    
    def __init__(self, log_dir=None):
        """Initialize the logging manager"""
        # Set log directory
        self.log_dir = log_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set default log level
        self.default_log_level = logging.INFO
        
        # Set default log format
        self.default_log_format = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
        
        # Set default date format
        self.default_date_format = '%Y-%m-%d %H:%M:%S'
        
        # Set default max bytes
        self.default_max_bytes = 10 * 1024 * 1024  # 10 MB
        
        # Set default backup count
        self.default_backup_count = 5
        
        # Initialize root logger
        self._configure_root_logger()
        
        # Initialize performance metrics
        self.performance_metrics = {}
        self._metrics_lock = threading.RLock()
        
        logging.info("Logging manager initialized")
    
    def _configure_root_logger(self):
        """Configure root logger"""
        # Get root logger
        root_logger = logging.getLogger()
        
        # Set log level
        root_logger.setLevel(self.default_log_level)
        
        # Remove existing handlers
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.default_log_level)
        
        # Create formatter
        formatter = logging.Formatter(self.default_log_format, self.default_date_format)
        
        # Set formatter
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        root_logger.addHandler(console_handler)
    
    def configure_logger(self, name, log_level=None, log_file=None):
        """Configure a logger with the given name"""
        # Get logger
        logger = logging.getLogger(name)
        
        # Set log level
        logger.setLevel(log_level or self.default_log_level)
        
        # Remove existing handlers
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level or self.default_log_level)
        
        # Create formatter
        formatter = logging.Formatter(self.default_log_format, self.default_date_format)
        
        # Set formatter
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Create file handler if log file is specified
        if log_file:
            # Create log file path
            log_file_path = os.path.join(self.log_dir, log_file)
            
            # Create file handler
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=self.default_max_bytes,
                backupCount=self.default_backup_count
            )
            file_handler.setLevel(log_level or self.default_log_level)
            
            # Set formatter
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
        
        return logger
    
    def set_log_level(self, name, log_level):
        """Set log level for a logger"""
        # Get logger
        logger = logging.getLogger(name)
        
        # Set log level
        logger.setLevel(log_level)
        
        # Set log level for handlers
        for handler in logger.handlers:
            handler.setLevel(log_level)
        
        return True
    
    def add_performance_metric(self, name, duration):
        """Add a performance metric"""
        with self._metrics_lock:
            if name not in self.performance_metrics:
                self.performance_metrics[name] = {
                    'count': 0,
                    'total_duration': 0,
                    'min_duration': float('inf'),
                    'max_duration': 0
                }
            
            # Update metrics
            self.performance_metrics[name]['count'] += 1
            self.performance_metrics[name]['total_duration'] += duration
            self.performance_metrics[name]['min_duration'] = min(
                self.performance_metrics[name]['min_duration'],
                duration
            )
            self.performance_metrics[name]['max_duration'] = max(
                self.performance_metrics[name]['max_duration'],
                duration
            )
        
        return True
    
    def get_performance_metrics(self):
        """Get performance metrics"""
        with self._metrics_lock:
            metrics = {}
            
            for name, data in self.performance_metrics.items():
                metrics[name] = {
                    'count': data['count'],
                    'total_duration': data['total_duration'],
                    'min_duration': data['min_duration'],
                    'max_duration': data['max_duration'],
                    'avg_duration': data['total_duration'] / data['count'] if data['count'] > 0 else 0
                }
            
            return metrics
    
    def reset_performance_metrics(self):
        """Reset performance metrics"""
        with self._metrics_lock:
            self.performance_metrics = {}
        
        return True
    
    def save_performance_metrics(self, file_path=None):
        """Save performance metrics to file"""
        try:
            # Generate file path if not provided
            if not file_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_path = os.path.join(self.log_dir, f'performance_metrics_{timestamp}.json')
            
            # Get metrics
            metrics = self.get_performance_metrics()
            
            # Save metrics
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logging.info(f"Saved performance metrics to {file_path}")
            return True
        
        except Exception as e:
            logging.error(f"Error saving performance metrics: {e}")
            return False

def configure_logging(log_dir=None, log_level=None):
    """Configure logging with the given directory and level"""
    # Create log directory if it doesn't exist
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger()
    
    # Set log level
    logger.setLevel(log_level or logging.INFO)
    
    # Check if handlers already exist
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level or logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )
        
        # Set formatter
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Create file handler if log directory is specified
        if log_dir:
            # Create log file path
            log_file = os.path.join(log_dir, 'app.log')
            
            # Create file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5
            )
            file_handler.setLevel(log_level or logging.INFO)
            
            # Set formatter
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
    
    return logger

def get_logger(name):
    """Get a logger with the given name"""
    with _logger_lock:
        # Check if logger exists in cache
        if name in _logger_cache:
            return _logger_cache[name]
        
        # Create logger
        logger = logging.getLogger(name)
        
        # Add to cache
        _logger_cache[name] = logger
        
        return logger

def set_context(key, value):
    """Set a context value for the current thread"""
    if not hasattr(_context_storage, 'context'):
        _context_storage.context = {}
    
    _context_storage.context[key] = value

def get_context(key, default=None):
    """Get a context value for the current thread"""
    if not hasattr(_context_storage, 'context'):
        _context_storage.context = {}
    
    return _context_storage.context.get(key, default)

def clear_context():
    """Clear all context values for the current thread"""
    if hasattr(_context_storage, 'context'):
        _context_storage.context = {}

def get_all_context():
    """Get all context values for the current thread"""
    if not hasattr(_context_storage, 'context'):
        _context_storage.context = {}
    
    return _context_storage.context.copy()

def with_context(context_name):
    """Decorator to add context to a function"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set context
            set_context('function', f"{func.__module__}.{func.__name__}")
            set_context('context', context_name)
            
            try:
                # Call function
                return func(*args, **kwargs)
            
            finally:
                # Clear context
                clear_context()
        
        return wrapper
    
    return decorator

def track_performance(operation_name):
    """Decorator to track performance of a function"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            logger = get_logger(func.__module__)
            
            # Get start time
            start_time = time.time()
            
            try:
                # Call function
                return func(*args, **kwargs)
            
            finally:
                # Calculate duration
                duration = time.time() - start_time
                
                # Log performance
                logger.debug(f"Performance: {operation_name} took {duration:.6f} seconds")
                
                # Add performance metric
                try:
                    # Get logging manager
                    logging_manager = getattr(sys.modules[__name__], '_logging_manager', None)
                    
                    # Add metric if manager exists
                    if logging_manager:
                        logging_manager.add_performance_metric(operation_name, duration)
                
                except Exception as e:
                    logger.error(f"Error adding performance metric: {e}")
        
        return wrapper
    
    return decorator

def log_exception(logger=None):
    """Decorator to log exceptions from a function"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            try:
                # Call function
                return func(*args, **kwargs)
            
            except Exception as e:
                # Get context
                context = get_all_context()
                
                # Log exception with context
                if context:
                    logger.error(f"Exception in {func.__name__}: {e} (context: {context})")
                else:
                    logger.error(f"Exception in {func.__name__}: {e}")
                
                # Log traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                
                # Re-raise exception
                raise
        
        return wrapper
    
    return decorator

# Initialize logging manager
_logging_manager = LoggingManager()

# Configure root logger
configure_logging()
