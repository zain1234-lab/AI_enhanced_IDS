"""
Packet sniffing module for AI-IDS
------------------------------
This module provides packet capture and analysis capabilities.
"""

import os
import sys
import time
import json
import socket
import struct
import threading
import logging
import ipaddress
from datetime import datetime
from enhanced_logging import get_logger, track_performance
from thread_safety import SafeThread, AtomicBoolean, synchronized
from resource_monitor import track_operation, limit_memory_usage
from security_utils import get_input_validator
from platform_utils import get_platform_detector

# Get logger for this module
logger = get_logger("packet_sniffing")

class PacketSniffer:
    """Class for capturing and analyzing network packets"""
    
    def __init__(self, interface=None, output_dir=None):
        """Initialize the packet sniffer"""
        # Set interface
        self.interface = interface
        
        # Set output directory
        self.output_dir = output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "packets")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize packet buffer
        self.packet_buffer = []
        self.max_buffer_size = 1000
        
        # Initialize packet count
        self.packet_count = 0
        
        # Initialize capture socket
        self.socket = None
        
        # Initialize capture thread
        self.capture_thread = None
        self.stop_flag = AtomicBoolean(False)
        
        # Initialize packet processors
        self.packet_processors = []
        
        # Initialize locks
        self._buffer_lock = threading.RLock()
        self._socket_lock = threading.RLock()
        
        # Initialize platform detector
        self.platform_detector = get_platform_detector()
        
        # Initialize input validator
        self.input_validator = get_input_validator()
        
        logger.info("Packet sniffer initialized")
    
    @synchronized
    def add_packet_processor(self, processor):
        """Add a packet processor"""
        try:
            # Add processor to list
            self.packet_processors.append(processor)
            
            logger.info(f"Added packet processor: {processor.__class__.__name__}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding packet processor: {e}")
            return False
    
    @synchronized
    def remove_packet_processor(self, processor):
        """Remove a packet processor"""
        try:
            # Remove processor from list
            if processor in self.packet_processors:
                self.packet_processors.remove(processor)
                
                logger.info(f"Removed packet processor: {processor.__class__.__name__}")
                return True
            
            logger.warning(f"Packet processor not found: {processor.__class__.__name__}")
            return False
        
        except Exception as e:
            logger.error(f"Error removing packet processor: {e}")
            return False
    
    @track_performance("start_capture")
    def start_capture(self, interface=None):
        """Start packet capture"""
        try:
            # Set interface if provided
            if interface:
                self.interface = interface
            
            # Check if interface is set
            if not self.interface:
                logger.error("Interface not set")
                return False
            
            # Check if already capturing
            if self.capture_thread and self.capture_thread.is_alive():
                logger.warning("Already capturing packets")
                return False
            
            # Reset stop flag
            self.stop_flag.set(False)
            
            # Create capture thread
            self.capture_thread = SafeThread(
                target=self._capture_loop,
                name="PacketCaptureThread"
            )
            self.capture_thread.daemon = True
            
            # Start capture thread
            self.capture_thread.start()
            
            # Wait for thread to start
            if not self.capture_thread.wait_for_start(timeout=5.0):
                logger.error("Capture thread failed to start")
                return False
            
            logger.info(f"Started packet capture on interface {self.interface}")
            return True
        
        except Exception as e:
            logger.error(f"Error starting packet capture: {e}")
            return False
    
    @track_performance("stop_capture")
    def stop_capture(self):
        """Stop packet capture"""
        try:
            # Check if capturing
            if not self.capture_thread or not self.capture_thread.is_alive():
                logger.warning("Not capturing packets")
                return False
            
            # Set stop flag
            self.stop_flag.set(True)
            
            # Wait for thread to stop
            self.capture_thread.join(timeout=5.0)
            
            # Check if thread stopped
            if self.capture_thread.is_alive():
                logger.warning("Capture thread did not stop gracefully")
                return False
            
            # Reset capture thread
            self.capture_thread = None
            
            logger.info("Stopped packet capture")
            return True
        
        except Exception as e:
            logger.error(f"Error stopping packet capture: {e}")
            return False
    
    def _capture_loop(self):
        """Packet capture loop"""
        try:
            # Create raw socket
            with self._socket_lock:
                try:
                    # Create socket based on platform
                    if self.platform_detector.is_windows():
                        # Windows requires administrator privileges
                        self.socket = self._create_windows_socket()
                    else:
                        # Linux/Unix socket
                        self.socket = self._create_unix_socket()
                    
                    if not self.socket:
                        logger.error("Failed to create capture socket")
                        return
                except Exception as e:
                    logger.error(f"Error creating capture socket: {e}")
                    return
            
            # Signal that thread has started
            if isinstance(threading.current_thread(), SafeThread):
                threading.current_thread()._started_event.set()
            
            # Capture packets
            while not self.stop_flag.get():
                try:
                    # Receive packet
                    packet_data, addr = self.socket.recvfrom(65535)
                    
                    # Process packet
                    self._process_packet(packet_data, addr)
                    
                except socket.timeout:
                    # Socket timeout, continue
                    continue
                
                except Exception as e:
                    logger.error(f"Error capturing packet: {e}")
                    time.sleep(0.1)
            
            # Close socket
            with self._socket_lock:
                if self.socket:
                    try:
                        self.socket.close()
                        self.socket = None
                    except Exception as e:
                        logger.error(f"Error closing capture socket: {e}")
        
        except Exception as e:
            logger.error(f"Error in capture loop: {e}")
        
        finally:
            # Close socket if still open
            with self._socket_lock:
                if self.socket:
                    try:
                        self.socket.close()
                        self.socket = None
                    except:
                        pass
    
    def _create_windows_socket(self):
        """Create a Windows capture socket"""
        try:
            # Import Windows-specific modules
            import win32file
            import win32event
            import win32con
            
            # Create raw socket
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
            
            # Bind to interface
            s.bind((self.interface, 0))
            
            # Set socket options
            s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
            s.ioctl(socket.SIO_RCVALL, socket.RCVALL_ON)
            
            # Set timeout
            s.settimeout(1.0)
            
            return s
        
        except Exception as e:
            logger.error(f"Error creating Windows capture socket: {e}")
            return None
    
    def _create_unix_socket(self):
        """Create a Unix capture socket"""
        try:
            # Create raw socket
            s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
            
            # Set timeout
            s.settimeout(1.0)
            
            return s
        
        except Exception as e:
            logger.error(f"Error creating Unix capture socket: {e}")
            return None
    
    @track_performance("process_packet")
    def _process_packet(self, packet_data, addr):
        """Process a captured packet"""
        try:
            # Parse packet
            packet = self._parse_packet(packet_data, addr)
            
            if not packet:
                return
            
            # Increment packet count
            self.packet_count += 1
            
            # Add to buffer
            with self._buffer_lock:
                self.packet_buffer.append(packet)
                
                # Trim buffer if needed
                if len(self.packet_buffer) > self.max_buffer_size:
                    self.packet_buffer = self.packet_buffer[-self.max_buffer_size:]
            
            # Process packet with processors
            for processor in self.packet_processors:
                try:
                    processor.process_packet(packet)
                except Exception as e:
                    logger.error(f"Error in packet processor {processor.__class__.__name__}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
    
    def _parse_packet(self, packet_data, addr):
        """Parse a packet"""
        try:
            # Create packet dictionary
            packet = {
                "timestamp": time.time(),
                "data": packet_data,
                "length": len(packet_data)
            }
            
            # Parse Ethernet header
            if self.platform_detector.is_windows():
                # Windows captures at IP layer
                self._parse_ip_header(packet, packet_data, 0)
            else:
                # Unix captures at Ethernet layer
                self._parse_ethernet_header(packet, packet_data)
            
            return packet
        
        except Exception as e:
            logger.error(f"Error parsing packet: {e}")
            return None
    
    def _parse_ethernet_header(self, packet, data):
        """Parse Ethernet header"""
        try:
            # Parse Ethernet header
            eth_header = struct.unpack("!6s6sH", data[:14])
            
            # Get source and destination MAC
            packet["eth_src_mac"] = self._format_mac(data[6:12])
            packet["eth_dst_mac"] = self._format_mac(data[:6])
            
            # Get EtherType
            packet["eth_type"] = eth_header[2]
            
            # Parse IP header if EtherType is IP
            if packet["eth_type"] == 0x0800:
                self._parse_ip_header(packet, data, 14)
        
        except Exception as e:
            logger.error(f"Error parsing Ethernet header: {e}")
    
    def _parse_ip_header(self, packet, data, offset):
        """Parse IP header"""
        try:
            # Parse IP header
            ip_header = struct.unpack("!BBHHHBBH4s4s", data[offset:offset+20])
            
            # Get IP version
            version_ihl = ip_header[0]
            packet["ip_version"] = version_ihl >> 4
            
            # Get header length
            ihl = version_ihl & 0xF
            ip_header_length = ihl * 4
            
            # Get protocol
            packet["ip_protocol"] = ip_header[6]
            
            # Get source and destination IP
            packet["src_ip"] = socket.inet_ntoa(ip_header[8])
            packet["dst_ip"] = socket.inet_ntoa(ip_header[9])
            
            # Parse TCP or UDP header
            if packet["ip_protocol"] == 6:  # TCP
                self._parse_tcp_header(packet, data, offset + ip_header_length)
            elif packet["ip_protocol"] == 17:  # UDP
                self._parse_udp_header(packet, data, offset + ip_header_length)
        
        except Exception as e:
            logger.error(f"Error parsing IP header: {e}")
    
    def _parse_tcp_header(self, packet, data, offset):
        """Parse TCP header"""
        try:
            # Parse TCP header
            tcp_header = struct.unpack("!HHLLBBHHH", data[offset:offset+20])
            
            # Get source and destination port
            packet["src_port"] = tcp_header[0]
            packet["dst_port"] = tcp_header[1]
            
            # Get sequence and acknowledgment numbers
            packet["tcp_seq"] = tcp_header[2]
            packet["tcp_ack"] = tcp_header[3]
            
            # Get flags
            packet["tcp_flags"] = tcp_header[5]
            packet["tcp_fin"] = (tcp_header[5] & 0x01) != 0
            packet["tcp_syn"] = (tcp_header[5] & 0x02) != 0
            packet["tcp_rst"] = (tcp_header[5] & 0x04) != 0
            packet["tcp_psh"] = (tcp_header[5] & 0x08) != 0
            packet["tcp_ack_flag"] = (tcp_header[5] & 0x10) != 0
            packet["tcp_urg"] = (tcp_header[5] & 0x20) != 0
            
            # Get header length
            data_offset = tcp_header[4] >> 4
            tcp_header_length = data_offset * 4
            
            # Set protocol
            packet["protocol"] = "tcp"
            
            # Get payload
            payload_offset = offset + tcp_header_length
            packet["payload"] = data[payload_offset:]
            packet["payload_length"] = len(packet["payload"])
        
        except Exception as e:
            logger.error(f"Error parsing TCP header: {e}")
    
    def _parse_udp_header(self, packet, data, offset):
        """Parse UDP header"""
        try:
            # Parse UDP header
            udp_header = struct.unpack("!HHHH", data[offset:offset+8])
            
            # Get source and destination port
            packet["src_port"] = udp_header[0]
            packet["dst_port"] = udp_header[1]
            
            # Get length
            packet["udp_length"] = udp_header[2]
            
            # Set protocol
            packet["protocol"] = "udp"
            
            # Get payload
            payload_offset = offset + 8
            packet["payload"] = data[payload_offset:]
            packet["payload_length"] = len(packet["payload"])
        
        except Exception as e:
            logger.error(f"Error parsing UDP header: {e}")
    
    def _format_mac(self, mac_bytes):
        """Format MAC address"""
        try:
            return ":".join(f"{b:02x}" for b in mac_bytes)
        
        except Exception as e:
            logger.error(f"Error formatting MAC address: {e}")
            return ""
    
    @track_performance("get_packets")
    def get_packets(self, count=None):
        """Get captured packets"""
        try:
            with self._buffer_lock:
                if count:
                    return self.packet_buffer[-count:]
                else:
                    return self.packet_buffer.copy()
        
        except Exception as e:
            logger.error(f"Error getting packets: {e}")
            return []
    
    @track_performance("save_packets")
    def save_packets(self, filename=None):
        """Save captured packets to file"""
        try:
            # Create filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"packets_{timestamp}.json"
            
            # Create filepath
            filepath = os.path.join(self.output_dir, filename)
            
            # Get packets
            packets = self.get_packets()
            
            # Convert packets to serializable format
            serializable_packets = []
            
            for packet in packets:
                # Create serializable packet
                serializable_packet = packet.copy()
                
                # Remove non-serializable fields
                if "data" in serializable_packet:
                    serializable_packet["data"] = None
                
                if "payload" in serializable_packet:
                    serializable_packet["payload"] = None
                
                # Add to list
                serializable_packets.append(serializable_packet)
            
            # Save packets
            with open(filepath, "w") as f:
                json.dump(serializable_packets, f, indent=2)
            
            logger.info(f"Saved {len(serializable_packets)} packets to {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Error saving packets: {e}")
            return None
    
    @track_performance("load_packets")
    def load_packets(self, filepath):
        """Load packets from file"""
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.error(f"Packet file not found: {filepath}")
                return False
            
            # Load packets
            with open(filepath, "r") as f:
                packets = json.load(f)
            
            # Add to buffer
            with self._buffer_lock:
                self.packet_buffer.extend(packets)
                
                # Trim buffer if needed
                if len(self.packet_buffer) > self.max_buffer_size:
                    self.packet_buffer = self.packet_buffer[-self.max_buffer_size:]
            
            logger.info(f"Loaded {len(packets)} packets from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading packets: {e}")
            return False
    
    @track_performance("clear_packets")
    def clear_packets(self):
        """Clear captured packets"""
        try:
            with self._buffer_lock:
                self.packet_buffer = []
                self.packet_count = 0
            
            logger.info("Cleared packet buffer")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing packets: {e}")
            return False
    
    @track_performance("get_packet_stats")
    def get_packet_stats(self):
        """Get packet statistics"""
        try:
            # Get packets
            packets = self.get_packets()
            
            # Initialize statistics
            stats = {
                "total_packets": len(packets),
                "protocols": {},
                "src_ips": {},
                "dst_ips": {},
                "src_ports": {},
                "dst_ports": {}
            }
            
            # Calculate statistics
            for packet in packets:
                # Count protocols
                protocol = packet.get("protocol")
                if protocol:
                    stats["protocols"][protocol] = stats["protocols"].get(protocol, 0) + 1
                
                # Count source IPs
                src_ip = packet.get("src_ip")
                if src_ip:
                    stats["src_ips"][src_ip] = stats["src_ips"].get(src_ip, 0) + 1
                
                # Count destination IPs
                dst_ip = packet.get("dst_ip")
                if dst_ip:
                    stats["dst_ips"][dst_ip] = stats["dst_ips"].get(dst_ip, 0) + 1
                
                # Count source ports
                src_port = packet.get("src_port")
                if src_port:
                    stats["src_ports"][src_port] = stats["src_ports"].get(src_port, 0) + 1
                
                # Count destination ports
                dst_port = packet.get("dst_port")
                if dst_port:
                    stats["dst_ports"][dst_port] = stats["dst_ports"].get(dst_port, 0) + 1
            
            # Sort statistics
            stats["protocols"] = dict(sorted(stats["protocols"].items(), key=lambda x: x[1], reverse=True))
            stats["src_ips"] = dict(sorted(stats["src_ips"].items(), key=lambda x: x[1], reverse=True))
            stats["dst_ips"] = dict(sorted(stats["dst_ips"].items(), key=lambda x: x[1], reverse=True))
            stats["src_ports"] = dict(sorted(stats["src_ports"].items(), key=lambda x: x[1], reverse=True))
            stats["dst_ports"] = dict(sorted(stats["dst_ports"].items(), key=lambda x: x[1], reverse=True))
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting packet statistics: {e}")
            return None

class PacketProcessor:
    """Base class for packet processors"""
    
    def __init__(self):
        """Initialize the packet processor"""
        logger.info(f"Packet processor initialized: {self.__class__.__name__}")
    
    def process_packet(self, packet):
        """Process a packet"""
        raise NotImplementedError("Subclasses must implement process_packet method")

class PacketFilter(PacketProcessor):
    """Class for filtering packets"""
    
    def __init__(self, filter_rules=None):
        """Initialize the packet filter"""
        super().__init__()
        
        # Set filter rules
        self.filter_rules = filter_rules or []
        
        # Initialize filtered packets
        self.filtered_packets = []
        self.max_filtered_packets = 1000
        
        # Initialize locks
        self._filter_lock = threading.RLock()
    
    @synchronized
    def add_filter_rule(self, rule):
        """Add a filter rule"""
        try:
            # Add rule to list
            self.filter_rules.append(rule)
            
            logger.info(f"Added filter rule: {rule}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding filter rule: {e}")
            return False
    
    @synchronized
    def remove_filter_rule(self, rule):
        """Remove a filter rule"""
        try:
            # Remove rule from list
            if rule in self.filter_rules:
                self.filter_rules.remove(rule)
                
                logger.info(f"Removed filter rule: {rule}")
                return True
            
            logger.warning(f"Filter rule not found: {rule}")
            return False
        
        except Exception as e:
            logger.error(f"Error removing filter rule: {e}")
            return False
    
    @synchronized
    def clear_filter_rules(self):
        """Clear all filter rules"""
        try:
            # Clear rules
            self.filter_rules = []
            
            logger.info("Cleared filter rules")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing filter rules: {e}")
            return False
    
    @track_performance("process_packet")
    def process_packet(self, packet):
        """Process a packet"""
        try:
            # Check if packet matches any filter rule
            if self._matches_filter(packet):
                # Add to filtered packets
                with self._filter_lock:
                    self.filtered_packets.append(packet)
                    
                    # Trim filtered packets if needed
                    if len(self.filtered_packets) > self.max_filtered_packets:
                        self.filtered_packets = self.filtered_packets[-self.max_filtered_packets:]
        
        except Exception as e:
            logger.error(f"Error processing packet in filter: {e}")
    
    def _matches_filter(self, packet):
        """Check if packet matches filter rules"""
        try:
            # If no rules, match all packets
            if not self.filter_rules:
                return True
            
            # Check each rule
            for rule in self.filter_rules:
                # Check rule type
                if rule["type"] == "ip":
                    # Check IP address
                    field = rule["field"]
                    value = rule["value"]
                    
                    if field in packet and packet[field] == value:
                        return True
                
                elif rule["type"] == "port":
                    # Check port
                    field = rule["field"]
                    value = rule["value"]
                    
                    if field in packet and packet[field] == value:
                        return True
                
                elif rule["type"] == "protocol":
                    # Check protocol
                    value = rule["value"]
                    
                    if "protocol" in packet and packet["protocol"] == value:
                        return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error matching filter: {e}")
            return False
    
    @track_performance("get_filtered_packets")
    def get_filtered_packets(self, count=None):
        """Get filtered packets"""
        try:
            with self._filter_lock:
                if count:
                    return self.filtered_packets[-count:]
                else:
                    return self.filtered_packets.copy()
        
        except Exception as e:
            logger.error(f"Error getting filtered packets: {e}")
            return []
    
    @track_performance("clear_filtered_packets")
    def clear_filtered_packets(self):
        """Clear filtered packets"""
        try:
            with self._filter_lock:
                self.filtered_packets = []
            
            logger.info("Cleared filtered packets")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing filtered packets: {e}")
            return False

class PacketAnalyzer(PacketProcessor):
    """Class for analyzing packets"""
    
    def __init__(self):
        """Initialize the packet analyzer"""
        super().__init__()
        
        # Initialize analysis results
        self.analysis_results = {}
        
        # Initialize locks
        self._analysis_lock = threading.RLock()
    
    @track_performance("process_packet")
    def process_packet(self, packet):
        """Process a packet"""
        try:
            # Analyze packet
            self._analyze_packet(packet)
        
        except Exception as e:
            logger.error(f"Error processing packet in analyzer: {e}")
    
    def _analyze_packet(self, packet):
        """Analyze a packet"""
        try:
            # Get protocol
            protocol = packet.get("protocol")
            
            if not protocol:
                return
            
            # Analyze based on protocol
            if protocol == "tcp":
                self._analyze_tcp_packet(packet)
            elif protocol == "udp":
                self._analyze_udp_packet(packet)
        
        except Exception as e:
            logger.error(f"Error analyzing packet: {e}")
    
    def _analyze_tcp_packet(self, packet):
        """Analyze a TCP packet"""
        try:
            # Check for SYN scan
            if packet.get("tcp_syn") and not packet.get("tcp_ack_flag"):
                # Get source IP
                src_ip = packet.get("src_ip")
                
                if not src_ip:
                    return
                
                # Update SYN scan count
                with self._analysis_lock:
                    if "syn_scan" not in self.analysis_results:
                        self.analysis_results["syn_scan"] = {}
                    
                    if src_ip not in self.analysis_results["syn_scan"]:
                        self.analysis_results["syn_scan"][src_ip] = 0
                    
                    self.analysis_results["syn_scan"][src_ip] += 1
            
            # Check for port scan
            src_ip = packet.get("src_ip")
            dst_ip = packet.get("dst_ip")
            dst_port = packet.get("dst_port")
            
            if src_ip and dst_ip and dst_port:
                # Update port scan count
                with self._analysis_lock:
                    if "port_scan" not in self.analysis_results:
                        self.analysis_results["port_scan"] = {}
                    
                    if src_ip not in self.analysis_results["port_scan"]:
                        self.analysis_results["port_scan"][src_ip] = {}
                    
                    if dst_ip not in self.analysis_results["port_scan"][src_ip]:
                        self.analysis_results["port_scan"][src_ip][dst_ip] = set()
                    
                    self.analysis_results["port_scan"][src_ip][dst_ip].add(dst_port)
        
        except Exception as e:
            logger.error(f"Error analyzing TCP packet: {e}")
    
    def _analyze_udp_packet(self, packet):
        """Analyze a UDP packet"""
        try:
            # Check for UDP scan
            src_ip = packet.get("src_ip")
            dst_ip = packet.get("dst_ip")
            dst_port = packet.get("dst_port")
            
            if src_ip and dst_ip and dst_port:
                # Update UDP scan count
                with self._analysis_lock:
                    if "udp_scan" not in self.analysis_results:
                        self.analysis_results["udp_scan"] = {}
                    
                    if src_ip not in self.analysis_results["udp_scan"]:
                        self.analysis_results["udp_scan"][src_ip] = {}
                    
                    if dst_ip not in self.analysis_results["udp_scan"][src_ip]:
                        self.analysis_results["udp_scan"][src_ip][dst_ip] = set()
                    
                    self.analysis_results["udp_scan"][src_ip][dst_ip].add(dst_port)
        
        except Exception as e:
            logger.error(f"Error analyzing UDP packet: {e}")
    
    @track_performance("get_analysis_results")
    def get_analysis_results(self):
        """Get analysis results"""
        try:
            with self._analysis_lock:
                # Convert sets to lists for JSON serialization
                results = {}
                
                for key, value in self.analysis_results.items():
                    if key == "syn_scan":
                        results[key] = value.copy()
                    elif key in ["port_scan", "udp_scan"]:
                        results[key] = {}
                        
                        for src_ip, dst_ips in value.items():
                            results[key][src_ip] = {}
                            
                            for dst_ip, ports in dst_ips.items():
                                results[key][src_ip][dst_ip] = list(ports)
                
                return results
        
        except Exception as e:
            logger.error(f"Error getting analysis results: {e}")
            return {}
    
    @track_performance("clear_analysis_results")
    def clear_analysis_results(self):
        """Clear analysis results"""
        try:
            with self._analysis_lock:
                self.analysis_results = {}
            
            logger.info("Cleared analysis results")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing analysis results: {e}")
            return False

# Singleton getter
def get_packet_sniffer(interface=None, output_dir=None):
    """Get packet sniffer instance"""
    return PacketSniffer(interface, output_dir)

def get_packet_filter(filter_rules=None):
    """Get packet filter instance"""
    return PacketFilter(filter_rules)

def get_packet_analyzer():
    """Get packet analyzer instance"""
    return PacketAnalyzer()

# Initialize packet sniffer
packet_sniffer = get_packet_sniffer()

class PacketProcessor:
    """Packet processor for model integration"""
    
    def __init__(self, detection_engine=None):
        self.detection_engine = detection_engine
        self.logger = get_logger("packet_processor")
    
    def process_packet(self, packet_dict):
        """Process packet with detection engine"""
        try:
            if self.detection_engine:
                return self.detection_engine.detect(packet_dict)
            return None
        except Exception as e:
            self.logger.error(f"Error processing packet: {e}")
            return None

# Global packet processor instance
_packet_processor = None

def get_packet_processor():
    """Get packet processor instance"""
    global _packet_processor
    if _packet_processor is None:
        _packet_processor = PacketProcessor()
    return _packet_processor

def set_model_integration(detection_engine):
    """Set model integration for packet processing"""
    processor = get_packet_processor()
    processor.detection_engine = detection_engine
    return processor
