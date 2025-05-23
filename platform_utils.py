"""
Platform-specific utilities for AI-IDS
----------------------------------
This module provides platform-specific utilities for the AI-IDS system.
"""

import os
import sys
import platform
import threading
import logging
import tempfile
import shutil
from enhanced_logging import get_logger

# Get logger for this module
logger = get_logger("platform_utils")

class PlatformDetector:
    """Class for detecting platform information"""
    
    def __init__(self):
        """Initialize the platform detector"""
        # Get platform information
        self.system = platform.system().lower()
        self.release = platform.release()
        self.machine = platform.machine()
        self.os_name = self.system
        self.os_version = self.release
        self.python_version = platform.python_version()
        self.is_container = self._check_container()
        
        # Log platform information
        logger.info(f"Platform detected: {self.system} {self.release} ({self.machine})")
        
        if self.is_container:
            logger.info("Running in container environment")
    
    def is_windows(self):
        """Check if platform is Windows"""
        return self.system == "windows"
    
    def is_linux(self):
        """Check if platform is Linux"""
        return self.system == "linux"
    
    def is_mac(self):
        """Check if platform is macOS"""
        return self.system == "darwin"
    
    def _check_container(self):
        """Check if running in a container"""
        try:
            # Check for container indicators
            if os.path.exists("/.dockerenv"):
                return True
            
            if os.path.exists("/run/.containerenv"):
                return True
            
            # Check cgroup
            try:
                with open("/proc/1/cgroup", "r") as f:
                    content = f.read()
                    if "docker" in content or "lxc" in content:
                        return True
            except:
                pass
            
            return False
        
        except Exception as e:
            logger.error(f"Error checking container: {e}")
            return False
    
    def get_platform_info(self):
        """Get platform information"""
        return {
            "system": self.system,
            "release": self.release,
            "machine": self.machine,
            "is_container": self.is_container,
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }
    
    def get_app_data_dir(self, app_name):
        """Get platform-specific application data directory"""
        if self.is_windows():
            return os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), app_name)
        elif self.is_mac():
            return os.path.join(os.path.expanduser("~"), "Library", "Application Support", app_name)
        else:  # Linux and others
            return os.path.join(os.path.expanduser("~"), f".{app_name.lower()}")

class NetworkInterfaceManager:
    """Class for managing network interfaces"""
    
    def __init__(self):
        """Initialize the network interface manager"""
        self.interfaces = []
        self._update_interfaces()
        logger.info("Network interface manager initialized")
    
    def _update_interfaces(self):
        """Update list of network interfaces"""
        try:
            import netifaces
            self.interfaces = netifaces.interfaces()
        except ImportError:
            logger.warning("netifaces module not available, using fallback")
            # Fallback for common interface names
            if platform.system().lower() == "windows":
                self.interfaces = ["Ethernet", "Wi-Fi", "Local Area Connection"]
            else:
                self.interfaces = ["eth0", "wlan0", "lo"]
        except Exception as e:
            logger.error(f"Error updating interfaces: {e}")
            self.interfaces = []
    
    def get_interface_names(self):
        """Get list of interface names"""
        return self.interfaces
    
    def get_interface_details(self, interface_name):
        """Get details for a specific interface"""
        try:
            import netifaces
            addresses = netifaces.ifaddresses(interface_name)
            return {
                "ipv4": addresses.get(netifaces.AF_INET, []),
                "ipv6": addresses.get(netifaces.AF_INET6, [])
            }
        except ImportError:
            logger.warning("netifaces module not available")
            return {}
        except Exception as e:
            logger.error(f"Error getting interface details for {interface_name}: {e}")
            return {}

class MemoryMonitor:
    """Class for monitoring system memory usage"""
    
    def __init__(self):
        """Initialize the memory monitor"""
        try:
            import psutil
            self.psutil = psutil
            logger.info("Memory monitor initialized")
        except ImportError:
            logger.error("psutil module not available")
            self.psutil = None
    
    def get_memory_usage(self):
        """Get current memory usage as a percentage"""
        if self.psutil:
            try:
                return self.psutil.virtual_memory().percent
            except Exception as e:
                logger.error(f"Error getting memory usage: {e}")
                return None
        return None
    
    def get_available_memory(self):
        """Get available memory in GB"""
        if self.psutil:
            try:
                return self.psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
            except Exception as e:
                logger.error(f"Error getting available memory: {e}")
                return None
        return None
    
    def get_total_memory(self):
        """Get total memory in GB"""
        if self.psutil:
            try:
                return self.psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
            except Exception as e:
                logger.error(f"Error getting total memory: {e}")
                return None
        return None
    
    def get_memory_stats(self):
        """Get comprehensive memory statistics"""
        if self.psutil:
            try:
                memory = self.psutil.virtual_memory()
                return {
                    "total_mb": memory.total / (1024 * 1024),
                    "available_mb": memory.available / (1024 * 1024),
                    "current_mb": memory.used / (1024 * 1024),
                    "usage_percent": memory.percent
                }
            except Exception as e:
                logger.error(f"Error getting memory stats: {e}")
                return {
                    "total_mb": 0,
                    "available_mb": 0,
                    "current_mb": 0,
                    "usage_percent": 0
                }
        return {
            "total_mb": 0,
            "available_mb": 0,
            "current_mb": 0,
            "usage_percent": 0
        }
    
    def reduce_memory_usage(self):
        """Attempt to reduce memory usage"""
        try:
            import gc
            gc.collect()
            logger.info("Performed garbage collection")
            return True
        except Exception as e:
            logger.error(f"Error reducing memory usage: {e}")
            return False

class PlatformManager:
    """Class for managing platform-specific operations"""
    
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
        """Initialize the platform manager"""
        # Initialize platform detector
        self.detector = PlatformDetector()
        
        # Initialize platform-specific directories
        self.temp_dir = tempfile.gettempdir()
        self.home_dir = os.path.expanduser("~")
        self.config_dir = self._get_config_dir()
        self.cache_dir = self._get_cache_dir()
        self.log_dir = self._get_log_dir()
        
        # Create directories
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info("Created platform-specific directories")
        
        # Initialize resource manager
        self._init_resource_manager()
        
        logger.info("Resource manager initialized")
        
        # Start resource monitoring
        self._start_resource_monitoring()
        
        logger.info("Resource monitoring started")
    
    def _get_config_dir(self):
        """Get platform-specific config directory"""
        if self.detector.is_windows():
            return os.path.join(os.environ.get("APPDATA", self.home_dir), "AI-IDS")
        elif self.detector.is_mac():
            return os.path.join(self.home_dir, "Library", "Application Support", "AI-IDS")
        else:  # Linux and others
            return os.path.join(self.home_dir, ".config", "ai-ids")
    
    def _get_cache_dir(self):
        """Get platform-specific cache directory"""
        if self.detector.is_windows():
            return os.path.join(os.environ.get("LOCALAPPDATA", self.home_dir), "AI-IDS", "Cache")
        elif self.detector.is_mac():
            return os.path.join(self.home_dir, "Library", "Caches", "AI-IDS")
        else:  # Linux and others
            return os.path.join(self.home_dir, ".cache", "ai-ids")
    
    def _get_log_dir(self):
        """Get platform-specific log directory"""
        if self.detector.is_windows():
            return os.path.join(os.environ.get("LOCALAPPDATA", self.home_dir), "AI-IDS", "Logs")
        elif self.detector.is_mac():
            return os.path.join(self.home_dir, "Library", "Logs", "AI-IDS")
        else:  # Linux and others
            return os.path.join(self.home_dir, ".local", "share", "ai-ids", "logs")
    
    def _init_resource_manager(self):
        """Initialize platform-specific resource manager"""
        # This is a placeholder for platform-specific resource management
        # In a real implementation, this would initialize different resource managers
        # based on the platform
        pass
    
    def _start_resource_monitoring(self):
        """Start platform-specific resource monitoring"""
        # This is a placeholder for platform-specific resource monitoring
        # In a real implementation, this would start different monitoring processes
        # based on the platform
        pass
    
    def get_network_interfaces(self):
        """Get available network interfaces"""
        try:
            import netifaces
            
            # Get interfaces
            interfaces = netifaces.interfaces()
            
            # Get interface details
            result = {}
            
            for interface in interfaces:
                try:
                    # Get addresses
                    addresses = netifaces.ifaddresses(interface)
                    
                    # Get IPv4 addresses
                    ipv4 = addresses.get(netifaces.AF_INET, [])
                    
                    # Get IPv6 addresses
                    ipv6 = addresses.get(netifaces.AF_INET6, [])
                    
                    # Add to result
                    result[interface] = {
                        "ipv4": ipv4,
                        "ipv6": ipv6
                    }
                
                except Exception as e:
                    logger.error(f"Error getting interface details for {interface}: {e}")
            
            return result
        
        except ImportError:
            logger.error("netifaces module not available")
            return {}
        
        except Exception as e:
            logger.error(f"Error getting network interfaces: {e}")
            return {}
    
    def get_system_info(self):
        """Get system information"""
        try:
            import psutil
            
            # Get CPU info
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory info
            memory = psutil.virtual_memory()
            
            # Get disk info
            disk = psutil.disk_usage("/")
            
            # Get network info
            network_interfaces = self.get_network_interfaces()
            
            # Return system info
            return {
                "platform": self.detector.get_platform_info(),
                "cpu": {
                    "count": cpu_count,
                    "count_logical": cpu_count_logical,
                    "percent": cpu_percent
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                },
                "network": {
                    "interfaces": network_interfaces
                }
            }
        
        except ImportError:
            logger.error("psutil module not available")
            return {
                "platform": self.detector.get_platform_info()
            }
        
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {
                "platform": self.detector.get_platform_info(),
                "error": str(e)
            }
    
    def cleanup(self):
        """Clean up platform-specific resources"""
        try:
            # Clean up cache directory
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            
            # Create cache directory
            os.makedirs(self.cache_dir, exist_ok=True)
            
            logger.info("Cleaned up platform-specific resources")
            return True
        
        except Exception as e:
            logger.error(f"Error cleaning up platform-specific resources: {e}")
            return False

# Singleton instances
_platform_detector = None
_network_interface_manager = None
_memory_monitor = None
_platform_manager = None

def get_platform_detector():
    """Get platform detector instance"""
    global _platform_detector
    if _platform_detector is None:
        _platform_detector = PlatformDetector()
    return _platform_detector

def get_network_interface_manager():
    """Get network interface manager instance"""
    global _network_interface_manager
    if _network_interface_manager is None:
        _network_interface_manager = NetworkInterfaceManager()
    return _network_interface_manager

def get_memory_monitor():
    """Get memory monitor instance"""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor

def get_platform_manager():
    """Get platform manager instance"""
    global _platform_manager
    if _platform_manager is None:
        _platform_manager = PlatformManager.get_instance()
    return _platform_manager

# Initialize platform manager
platform_manager = get_platform_manager()