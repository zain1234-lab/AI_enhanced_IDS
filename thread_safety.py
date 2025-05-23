"""
Thread safety utilities for AI-IDS
------------------------------
This module provides thread safety utilities for the AI-IDS system.
"""

import os
import sys
import time
import threading
import functools
import logging
from enhanced_logging import get_logger

# Get logger for this module
logger = get_logger("thread_safety")

class AtomicBoolean:
    """Thread-safe boolean value"""
    
    def __init__(self, initial_value=False):
        """Initialize the atomic boolean"""
        self._value = initial_value
        self._lock = threading.RLock()
        # Add event for compatibility with threading.Event
        self._event = threading.Event()
        if initial_value:
            self._event.set()
    
    def get(self):
        """Get the current value"""
        with self._lock:
            return self._value
    
    def set(self, value):
        """Set the value"""
        with self._lock:
            self._value = bool(value)
            if self._value:
                self._event.set()
            else:
                self._event.clear()
            return self._value
    
    def compare_and_set(self, expected, update):
        """Compare and set the value"""
        with self._lock:
            if self._value == expected:
                self._value = bool(update)
                if self._value:
                    self._event.set()
                else:
                    self._event.clear()
                return True
            return False
            
    # Add compatibility with threading.Event
    def is_set(self):
        """Check if value is True (for compatibility with threading.Event)"""
        return self._event.is_set()
        
    def wait(self, timeout=None):
        """Wait for value to become True (for compatibility with threading.Event)"""
        return self._event.wait(timeout)
        
    def clear(self):
        """Set value to False (for compatibility with threading.Event)"""
        self.set(False)

class SafeDict:
    """Thread-safe dictionary"""
    
    def __init__(self, initial_dict=None):
        """Initialize the safe dictionary"""
        self._dict = initial_dict or {}
        self._lock = threading.RLock()
    
    def get(self, key, default=None):
        """Get a value from the dictionary"""
        with self._lock:
            return self._dict.get(key, default)
    
    def set(self, key, value):
        """Set a value in the dictionary"""
        with self._lock:
            self._dict[key] = value
            return value
    
    def delete(self, key):
        """Delete a key from the dictionary"""
        with self._lock:
            if key in self._dict:
                del self._dict[key]
                return True
            return False
    
    def keys(self):
        """Get all keys in the dictionary"""
        with self._lock:
            return list(self._dict.keys())
    
    def values(self):
        """Get all values in the dictionary"""
        with self._lock:
            return list(self._dict.values())
    
    def items(self):
        """Get all items in the dictionary"""
        with self._lock:
            return list(self._dict.items())
    
    def clear(self):
        """Clear the dictionary"""
        with self._lock:
            self._dict.clear()
    
    def __getitem__(self, key):
        """Get a value from the dictionary"""
        with self._lock:
            return self._dict[key]
    
    def __setitem__(self, key, value):
        """Set a value in the dictionary"""
        with self._lock:
            self._dict[key] = value
    
    def __delitem__(self, key):
        """Delete a key from the dictionary"""
        with self._lock:
            del self._dict[key]
    
    def __contains__(self, key):
        """Check if key is in the dictionary"""
        with self._lock:
            return key in self._dict
    
    def __len__(self):
        """Get the length of the dictionary"""
        with self._lock:
            return len(self._dict)
    
    def __str__(self):
        """Get string representation of the dictionary"""
        with self._lock:
            return str(self._dict)
    
    def __repr__(self):
        """Get representation of the dictionary"""
        with self._lock:
            return repr(self._dict)

class SafeThread(threading.Thread):
    """Thread with additional safety features"""
    
    def __init__(self, *args, **kwargs):
        """Initialize the safe thread"""
        super().__init__(*args, **kwargs)
        # Use standard threading.Event for full compatibility
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._exception = None
    
    def run(self):
        """Run the thread"""
        try:
            self._started.set()
            super().run()
        except Exception as e:
            self._exception = e
            logger.error(f"Exception in thread {self.name}: {e}")
        finally:
            self._stopped.set()
    
    def is_started(self):
        """Check if thread has started"""
        return self._started.is_set()
    
    def is_stopped(self):
        """Check if thread has stopped"""
        return self._stopped.is_set()
    
    def get_exception(self):
        """Get any exception that occurred"""
        return self._exception
    
    def wait_for_start(self, timeout=None):
        """Wait for thread to start"""
        return self._started.wait(timeout)
    
    def wait_for_stop(self, timeout=None):
        """Wait for thread to stop"""
        return self._stopped.wait(timeout)

class ThreadPool:
    """Thread pool for managing multiple threads"""
    
    def __init__(self, max_threads=10):
        """Initialize the thread pool"""
        self.max_threads = max_threads
        self.threads = []
        self._lock = threading.RLock()
    
    def submit(self, target, args=(), kwargs=None, name=None):
        """Submit a task to the thread pool"""
        with self._lock:
            # Clean up completed threads
            self._cleanup()
            
            # Check if pool is full
            if len(self.threads) >= self.max_threads:
                return None
            
            # Create thread
            thread = SafeThread(
                target=target,
                args=args,
                kwargs=kwargs or {},
                name=name or f"ThreadPool-{len(self.threads)}"
            )
            thread.daemon = True
            
            # Add to pool
            self.threads.append(thread)
            
            # Start thread
            thread.start()
            
            return thread
    
    def _cleanup(self):
        """Clean up completed threads"""
        self.threads = [t for t in self.threads if t.is_alive()]
    
    def shutdown(self, wait=True):
        """Shutdown the thread pool"""
        with self._lock:
            # Wait for threads to complete
            if wait:
                for thread in self.threads:
                    thread.join()
            
            # Clear threads
            self.threads = []

def synchronized(func):
    """Decorator for synchronizing method access"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Ensure _lock exists
        if not hasattr(self, '_lock'):
            self._lock = threading.RLock()
        with self._lock:
            return func(self, *args, **kwargs)
    return wrapper

def with_retry(max_attempts=3, delay=1.0):
    """Decorator for retrying operations"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Retry {attempt+1}/{max_attempts} for {func.__name__}: {e}")
                    
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            
            if last_exception:
                raise last_exception
        
        return wrapper
    
    return decorator

def with_timeout(timeout=60.0):
    """Decorator for adding timeout to operations"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = SafeThread(target=target)
            thread.daemon = True
            thread.start()
            
            thread.join(timeout)
            
            if thread.is_alive():
                raise TimeoutError(f"Operation timed out after {timeout} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    
    return decorator
