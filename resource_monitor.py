"""
Resource monitoring and management module for AI-IDS
-------------------------------------------------
This module provides resource monitoring and management capabilities.
"""

import os
import sys
import time
import json
import psutil
import threading
import logging
import functools
from datetime import datetime
from enhanced_logging import get_logger

# Get logger for this module
logger = get_logger('resource_monitor')

class ResourceMonitor:
    """Class for monitoring system resources"""
    
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
        """Initialize the resource monitor"""
        # Initialize resource limits
        self.cpu_limit = 0.8  # 80%
        self.memory_limit = 0.8  # 80%
        self.memory_limit_mb = 1024  # 1 GB
        
        # Initialize operation tracking
        self.operations = {}
        
        # Initialize locks
        self._operations_lock = threading.RLock()
        
        logger.info("Resource monitor initialized")
    
    def get_resource_usage(self):
        """Get current resource usage"""
        try:
            # Get process
            process = psutil.Process(os.getpid())
            
            # Get process resource usage
            process_cpu_percent = process.cpu_percent(interval=0.1) / 100.0
            process_memory_info = process.memory_info()
            process_memory_mb = process_memory_info.rss / (1024 * 1024)
            process_memory_percent = process.memory_percent() / 100.0
            process_threads = len(process.threads())
            process_open_files = len(process.open_files())
            
            # Get system resource usage
            system_cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            system_memory = psutil.virtual_memory()
            system_memory_percent = system_memory.percent / 100.0
            system_memory_available_mb = system_memory.available / (1024 * 1024)
            
            # Return resource usage
            return {
                'timestamp': time.time(),
                'process': {
                    'pid': os.getpid(),
                    'cpu_percent': process_cpu_percent,
                    'memory_mb': process_memory_mb,
                    'memory_percent': process_memory_percent,
                    'threads': process_threads,
                    'open_files': process_open_files
                },
                'system': {
                    'cpu_percent': system_cpu_percent,
                    'memory_percent': system_memory_percent,
                    'memory_available_mb': system_memory_available_mb
                },
                'limits': {
                    'cpu_percent': self.cpu_limit,
                    'memory_percent': self.memory_limit,
                    'memory_mb': self.memory_limit_mb
                }
            }
        
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {
                'timestamp': time.time(),
                'process': {
                    'pid': os.getpid(),
                    'cpu_percent': 0.0,
                    'memory_mb': 0.0,
                    'memory_percent': 0.0,
                    'threads': 0,
                    'open_files': 0
                },
                'system': {
                    'cpu_percent': 0.0,
                    'memory_percent': 0.0,
                    'memory_available_mb': 0.0
                },
                'limits': {
                    'cpu_percent': self.cpu_limit,
                    'memory_percent': self.memory_limit,
                    'memory_mb': self.memory_limit_mb
                }
            }
    
    def check_resource_limits(self):
        """Check if resource usage exceeds limits"""
        try:
            # Get resource usage
            usage = self.get_resource_usage()
            
            # Check CPU limit
            if usage['process']['cpu_percent'] > self.cpu_limit:
                logger.warning(f"Process CPU usage ({usage['process']['cpu_percent']:.2%}) exceeds limit ({self.cpu_limit:.2%})")
                return False
            
            # Check memory limit (percent)
            if usage['process']['memory_percent'] > self.memory_limit:
                logger.warning(f"Process memory usage ({usage['process']['memory_percent']:.2%}) exceeds limit ({self.memory_limit:.2%})")
                return False
            
            # Check memory limit (MB)
            if usage['process']['memory_mb'] > self.memory_limit_mb:
                logger.warning(f"Process memory usage ({usage['process']['memory_mb']:.2f} MB) exceeds limit ({self.memory_limit_mb:.2f} MB)")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking resource limits: {e}")
            return False
    
    def set_resource_limits(self, cpu_limit=None, memory_limit=None, memory_limit_mb=None):
        """Set resource limits"""
        try:
            # Set CPU limit
            if cpu_limit is not None:
                self.cpu_limit = max(0.1, min(1.0, cpu_limit))
            
            # Set memory limit (percent)
            if memory_limit is not None:
                self.memory_limit = max(0.1, min(1.0, memory_limit))
            
            # Set memory limit (MB)
            if memory_limit_mb is not None:
                self.memory_limit_mb = max(64, memory_limit_mb)
            
            logger.info(f"Resource limits set: CPU={self.cpu_limit:.2%}, Memory={self.memory_limit:.2%}, Memory MB={self.memory_limit_mb:.2f}")
            return True
        
        except Exception as e:
            logger.error(f"Error setting resource limits: {e}")
            return False
    
    def start_operation(self, operation_name):
        """Start tracking an operation"""
        try:
            with self._operations_lock:
                # Create operation
                operation = {
                    'name': operation_name,
                    'start_time': time.time(),
                    'end_time': None,
                    'duration': None,
                    'completed': False,
                    'resource_usage': self.get_resource_usage()
                }
                
                # Add to operations
                self.operations[operation_name] = operation
                
                logger.debug(f"Started operation: {operation_name}")
                return True
        
        except Exception as e:
            logger.error(f"Error starting operation: {e}")
            return False
    
    def end_operation(self, operation_name):
        """End tracking an operation"""
        try:
            with self._operations_lock:
                # Check if operation exists
                if operation_name not in self.operations:
                    logger.warning(f"Operation not found: {operation_name}")
                    return False
                
                # Get operation
                operation = self.operations[operation_name]
                
                # Set end time
                operation['end_time'] = time.time()
                
                # Calculate duration
                operation['duration'] = operation['end_time'] - operation['start_time']
                
                # Set completed flag
                operation['completed'] = True
                
                # Get final resource usage
                operation['final_resource_usage'] = self.get_resource_usage()
                
                logger.debug(f"Ended operation: {operation_name} (duration: {operation['duration']:.6f} seconds)")
                
                # Remove operation
                del self.operations[operation_name]
                
                return True
        
        except Exception as e:
            logger.error(f"Error ending operation: {e}")
            return False
    
    def get_operation(self, operation_name):
        """Get operation information"""
        try:
            with self._operations_lock:
                # Check if operation exists
                if operation_name not in self.operations:
                    logger.warning(f"Operation not found: {operation_name}")
                    return None
                
                # Get operation
                operation = self.operations[operation_name]
                
                # Update resource usage
                operation['current_resource_usage'] = self.get_resource_usage()
                
                # Calculate current duration
                if operation['end_time'] is None:
                    operation['current_duration'] = time.time() - operation['start_time']
                
                return operation
        
        except Exception as e:
            logger.error(f"Error getting operation: {e}")
            return None
    
    def get_all_operations(self):
        """Get all operations"""
        try:
            with self._operations_lock:
                operations = {}
                
                for name, operation in self.operations.items():
                    # Update resource usage
                    operation['current_resource_usage'] = self.get_resource_usage()
                    
                    # Calculate current duration
                    if operation['end_time'] is None:
                        operation['current_duration'] = time.time() - operation['start_time']
                    
                    operations[name] = operation
                
                return operations
        
        except Exception as e:
            logger.error(f"Error getting all operations: {e}")
            return {}
    
    def check_memory_usage(self, max_mb=None):
        """Check if memory usage exceeds limit and attempt to reduce if needed"""
        try:
            # Get resource usage
            usage = self.get_resource_usage()
            
            # Get memory usage
            memory_mb = usage['process']['memory_mb']
            
            # Use provided limit or default
            limit_mb = max_mb or self.memory_limit_mb
            
            # Check if memory usage exceeds limit
            if memory_mb > limit_mb:
                logger.warning(f"High memory usage: {memory_mb:.2f} MB > {limit_mb:.2f} MB")
                
                # Attempt to reduce memory usage
                self._reduce_memory_usage()
                
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return False
    
    def _reduce_memory_usage(self):
        """Attempt to reduce memory usage"""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Memory reduction attempted")
            return True
        
        except Exception as e:
            logger.error(f"Error reducing memory usage: {e}")
            return False

class OperationTracker:
    """Context manager for tracking operations"""
    
    def __init__(self, operation_name):
        """Initialize the operation tracker"""
        self.operation_name = operation_name
        self.monitor = get_resource_monitor()
    
    def __enter__(self):
        """Start tracking operation"""
        self.monitor.start_operation(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End tracking operation"""
        self.monitor.end_operation(self.operation_name)
        return False

class MemoryLimiter:
    """Context manager for limiting memory usage"""
    
    def __init__(self, max_mb=None):
        """Initialize the memory limiter"""
        self.max_mb = max_mb
        self.monitor = get_resource_monitor()
    
    def __enter__(self):
        """Check memory usage on enter"""
        self.monitor.check_memory_usage(self.max_mb)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Check memory usage on exit"""
        self.monitor.check_memory_usage(self.max_mb)
        return False

# Singleton getter
def get_resource_monitor():
    """Get resource monitor singleton"""
    return ResourceMonitor.get_instance()

# Decorator for tracking operation performance
def track_operation(operation_name):
    """Decorator for tracking operation performance"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_resource_monitor()
            monitor.start_operation(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                monitor.end_operation(operation_name)
        return wrapper
    return decorator

# Decorator for limiting memory usage
def limit_memory_usage(max_mb=None):
    """Decorator for limiting memory usage"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_resource_monitor()
            monitor.check_memory_usage(max_mb)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                monitor.check_memory_usage(max_mb)
        return wrapper
    return decorator

# Initialize resource monitor
resource_monitor = get_resource_monitor()
