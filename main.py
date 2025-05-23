"""
Main script for AI-enhanced Intrusion Detection System
-----------------------------------------------------
This module serves as the entry point for the application,
integrating all components and providing the web interface.
"""

import os
import sys
import time
import threading
import logging
import argparse
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from enhanced_logging import get_logger, LoggingManager
from platform_utils import get_platform_detector, get_network_interface_manager, get_memory_monitor
from advanced_feature_engineering import get_feature_extractor
from self_improving_model import get_online_learning_manager
from packet_sniffing import get_packet_sniffer, get_packet_processor, set_model_integration
from model_integration import get_model_integration, get_model_ensemble, get_detection_engine
from gui import create_app

# Get logger for this module
logger = get_logger('main')

# Get platform utilities
platform_detector = get_platform_detector()
network_manager = get_network_interface_manager()
memory_monitor = get_memory_monitor()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AI-Enhanced Intrusion Detection System')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the web interface on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the web interface on')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--interface', type=str, default=None,
                        help='Network interface to monitor')
    parser.add_argument('--models-dir', type=str, default=None,
                        help='Directory containing trained models')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for log files')
    parser.add_argument('--cli', action='store_true',
                        help='Run in CLI mode instead of GUI mode')
    
    return parser.parse_args()

def setup_directories(args):
    """Set up necessary directories"""
    # Get platform-specific app data directory
    app_data_dir = platform_detector.get_app_data_dir('ai_ids')
    
    # Set up output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created directory: {output_dir}")
    
    # Set up models directory
    models_dir = args.models_dir
    if not models_dir:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Created directory: {models_dir}")
    
    # Set up logs directory
    log_dir = args.log_dir
    if not log_dir:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger.info(f"Created directory: {log_dir}")
    
    # Set up data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Created directory: {data_dir}")
    
    # Set up profiles directory
    profiles_dir = os.path.join(app_data_dir, 'profiles')
    os.makedirs(profiles_dir, exist_ok=True)
    
    return {
        'app_data_dir': app_data_dir,
        'output_dir': output_dir,
        'models_dir': models_dir,
        'log_dir': log_dir,
        'profiles_dir': profiles_dir,
        'data_dir': data_dir
    }

def initialize_components(directories, args):
    """Initialize all system components"""
    # Initialize logging
    logging_manager = LoggingManager(log_dir=directories['log_dir'])
    logger.info("Logging manager initialized")
    
    # Initialize packet sniffer
    packet_sniffer = get_packet_sniffer()
    if args.interface:
        packet_sniffer.interface = args.interface
    
    # Initialize feature extractor
    feature_extractor = get_feature_extractor()
    
    # Initialize model integration
    model_ensemble = get_model_ensemble()
    detection_engine = get_detection_engine()
    
    # Load models
    model_loaded = model_ensemble.load_models(directories['models_dir'])
    
    # Initialize online learning
    online_learning = get_online_learning_manager()
    online_learning.models_dir = directories['models_dir']
    online_learning.profiles_dir = directories['profiles_dir']
    
    # Initialize packet processor with model integration
    packet_processor = set_model_integration(detection_engine)
    
    # Return initialized components
    return {
        'logging_manager': logging_manager,
        'packet_sniffer': packet_sniffer,
        'feature_extractor': feature_extractor,
        'model_ensemble': model_ensemble,
        'detection_engine': detection_engine,
        'online_learning': online_learning,
        'packet_processor': packet_processor
    }

def setup_packet_processing(components):
    """Set up packet processing pipeline"""
    packet_sniffer = components['packet_sniffer']
    packet_processor = components['packet_processor']
    feature_extractor = components['feature_extractor']
    detection_engine = components['detection_engine']
    online_learning = components['online_learning']
    
    # Define packet callback
    def packet_callback(packet_dict):
        try:
            # Extract features
            features = feature_extractor.extract_features(packet_dict)
            
            # Detect intrusion
            result = detection_engine.detect(features)
            
            # Process packet
            processed_result = packet_processor.process_packet(packet_dict)
            
            # Update online learning
            if result:
                online_learning.process_packet(
                    packet_dict, 
                    features, 
                    result.get('is_attack', False),
                    result.get('confidence', 0.0)
                )
            
            return result
        except Exception as e:
            logger.error(f"Error in packet callback: {e}")
            return None
    
    # Set packet callback
    packet_sniffer.callback = packet_callback
    
    logger.info("Packet processing pipeline set up")

def start_background_tasks(components):
    """Start background tasks"""
    # Memory monitoring task
    def memory_monitor_task():
        while True:
            try:
                # Check memory usage
                memory_stats = memory_monitor.get_memory_stats()
                
                # Log if usage is high
                if memory_stats['usage_percent'] > 80:
                    logger.warning(f"High memory usage: {memory_stats['current_mb']:.2f} MB ({memory_stats['usage_percent']:.2f}%)")
                    
                    # Try to reduce memory usage
                    memory_monitor.reduce_memory_usage()
                
                # Sleep for a while
                time.sleep(60)
            
            except Exception as e:
                logger.error(f"Error in memory monitor task: {e}")
                time.sleep(60)
    
    # Model adaptation task
    def model_adaptation_task():
        while True:
            try:
                # Get online learning manager
                online_learning = components['online_learning']
                
                # Get adaptation stats
                stats = online_learning.get_adaptation_stats()
                
                # Log stats periodically
                logger.info(f"Adaptation stats: {stats['adaptation_count']} adaptations, "
                           f"active profile: {stats['active_profile']}, "
                           f"threshold: {stats['current_threshold']:.4f}")
                
                # Sleep for a while
                time.sleep(300)
            
            except Exception as e:
                logger.error(f"Error in model adaptation task: {e}")
                time.sleep(300)
    
    # Start memory monitor thread
    memory_thread = threading.Thread(target=memory_monitor_task)
    memory_thread.daemon = True
    memory_thread.start()
    
    # Start model adaptation thread
    adaptation_thread = threading.Thread(target=model_adaptation_task)
    adaptation_thread.daemon = True
    adaptation_thread.start()
    
    logger.info("Background tasks started")

def start_packet_capture(components, interface=None):
    """Start packet capture"""
    packet_sniffer = components['packet_sniffer']
    
    # Set interface if provided
    if interface:
        packet_sniffer.interface = interface
    
    # Check if interface is set
    if not packet_sniffer.interface:
        # Try to find a default interface
        interfaces = network_manager.get_interface_names()
        if interfaces:
            packet_sniffer.interface = interfaces[0]
            logger.info(f"Using default interface: {packet_sniffer.interface}")
        else:
            logger.warning("No interface specified, packet capture not started")
            return False
    
    # Start packet capture
    success = packet_sniffer.start_capture()
    
    if success:
        logger.info(f"Started packet capture on interface: {packet_sniffer.interface}")
    else:
        logger.error(f"Failed to start packet capture on interface: {packet_sniffer.interface}")
    
    return success

def run_cli_mode(components, args, directories):
    """Run in CLI mode"""
    logger.info("Running in CLI mode")
    
    # Start packet capture if interface is specified
    if args.interface:
        start_packet_capture(components, args.interface)
    
    # Print available interfaces
    interfaces = network_manager.get_interface_names()
    print(f"Available interfaces: {', '.join(interfaces)}")
    
    # Print loaded models
    model_ensemble = components['model_ensemble']
    model_names = model_ensemble.get_model_names()
    print(f"Loaded models: {', '.join(model_names)}")
    
    # Print help
    print("\nCommands:")
    print("  help - Show this help")
    print("  start <interface> - Start packet capture on interface")
    print("  stop - Stop packet capture")
    print("  stats - Show detection stats")
    print("  models - Show loaded models")
    print("  exit - Exit the program")
    
    # Main loop
    while True:
        try:
            # Get command
            command = input("\nEnter command: ").strip()
            
            # Process command
            if command == "help":
                print("\nCommands:")
                print("  help - Show this help")
                print("  start <interface> - Start packet capture on interface")
                print("  stop - Stop packet capture")
                print("  stats - Show detection stats")
                print("  models - Show loaded models")
                print("  exit - Exit the program")
            
            elif command.startswith("start"):
                # Parse interface
                parts = command.split()
                if len(parts) > 1:
                    interface = parts[1]
                    start_packet_capture(components, interface)
                else:
                    print("Error: Interface not specified")
            
            elif command == "stop":
                # Stop packet capture
                packet_sniffer = components['packet_sniffer']
                packet_sniffer.stop_capture()
                print("Packet capture stopped")
            
            elif command == "stats":
                # Show detection stats
                detection_engine = components['detection_engine']
                stats = detection_engine.get_stats()
                print("\nDetection Stats:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            
            elif command == "models":
                # Show loaded models
                model_ensemble = components['model_ensemble']
                model_names = model_ensemble.get_model_names()
                print(f"\nLoaded models: {', '.join(model_names)}")
                
                # Show model info
                for name in model_names:
                    info = model_ensemble.get_model_info(name)
                    if info:
                        print(f"\n  {name}:")
                        for key, value in info.items():
                            print(f"    {key}: {value}")
            
            elif command == "exit":
                # Exit the program
                print("Exiting...")
                break
            
            else:
                print(f"Unknown command: {command}")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        except Exception as e:
            print(f"Error: {e}")
    
    # Stop packet capture
    packet_sniffer = components['packet_sniffer']
    packet_sniffer.stop_capture()

def create_fallback_app():
    """Create a fallback Flask app if gui module fails"""
    try:
        from flask import Flask, jsonify, render_template_string
        
        app = Flask(__name__)
        
        @app.route('/')
        def index():
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI-Enhanced IDS</title>
            </head>
            <body>
                <h1>AI-Enhanced Intrusion Detection System</h1>
                <p>System is running in fallback mode.</p>
                <p>The full GUI interface is not available.</p>
            </body>
            </html>
            ''')
        
        @app.route('/api/status')
        def status():
            return jsonify({'status': 'running', 'mode': 'fallback'})
        
        return app
    except ImportError:
        logger.error("Flask not available, cannot create fallback app")
        return None

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up directories
    directories = setup_directories(args)
    
    # Initialize components
    components = initialize_components(directories, args)
    
    # Set up packet processing
    setup_packet_processing(components)
    
    # Start background tasks
    start_background_tasks(components)
    
    # Log startup information
    logger.info(f"Platform: {platform_detector.os_name} {platform_detector.os_version}")
    logger.info(f"Python version: {platform_detector.python_version}")
    logger.info(f"Network interfaces: {', '.join(network_manager.get_interface_names())}")
    
    # Run in CLI or GUI mode
    if args.cli:
        run_cli_mode(components, args, directories)
    else:
        try:
            # Create Flask app
            app_result = create_app(components, directories)
            
            # Handle different return types from create_app
            if isinstance(app_result, tuple):
                # If create_app returns a tuple, extract the app from it
                app = app_result[0]
                logger.info("GUI module returned tuple, using first element as app")
            else:
                # Assume it's the Flask app directly
                app = app_result
            
            # Verify we have a valid Flask app
            if app is None or not hasattr(app, 'run'):
                logger.warning("Invalid app object returned from GUI module, creating fallback")
                app = create_fallback_app()
            
            if app is not None:
                # Log startup information
                logger.info(f"Starting web interface on {args.host}:{args.port}")
                
                # Run the app
                try:
                    app.run(host=args.host, port=args.port, debug=args.debug)
                except Exception as e:
                    logger.error(f"Error running Flask app: {e}")
                    logger.info("Falling back to CLI mode")
                    run_cli_mode(components, args, directories)
            else:
                logger.error("Could not create web interface, falling back to CLI mode")
                run_cli_mode(components, args, directories)
                
        except Exception as e:
            logger.error(f"Error creating GUI: {e}")
            logger.info("Falling back to CLI mode")
            run_cli_mode(components, args, directories)

if __name__ == '__main__':
    main()