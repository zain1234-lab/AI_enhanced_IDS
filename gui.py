"""
Web GUI module for AI-IDS using Flask and SocketIO.
-----------------------------------------------------
This module provides the web interface for the AI-Enhanced Intrusion Detection System.
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from enhanced_logging import get_logger # Use the centralized logger
from thread_safety import SafeDict, synchronized
from platform_utils import get_network_interface_manager

# Use the logger configured by LoggingManager in main.py
logger = get_logger('gui')

# --- State Management --- 
# Simple dictionary to hold shared state for the web app instance
# This replaces the more complex GUIStateManager used by Tkinter
app_state = SafeDict({
    'selected_interface': None,
    'is_running': False,
    'packet_count': 0,
    'alert_count': 0,
    # Add other state variables as needed
})

# --- Flask App and SocketIO Setup --- 

# Initialize Flask app and SocketIO
app = Flask(__name__, template_folder='templates', static_folder='static') # Assuming templates/static are in the root
socketio = SocketIO(app, async_mode='threading') # Use threading for background tasks

# Store components and directories passed from main.py
app_components = {}
app_directories = {}

def create_app(components, directories):
    """Creates and configures the Flask application and SocketIO instance."""
    global app_components, app_directories
    app_components = components
    app_directories = directories

    # Update state with initial component info if needed
    # e.g., app_state.set_state('models_dir', directories.get('models_dir'))

    logger.info("Flask app created and configured.")
    return app, socketio

# --- Flask Routes --- 

@app.route('/')
def index_route():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/packets')
def packets():
    return render_template('packets.html')

@app.route('/alerts')
def alerts():
    return render_template('alerts.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

# Serve static files (CSS, JS, Images)
# Adjust paths if your static files are organized differently
@app.route('/static/<path:subpath>')
def serve_static(subpath):
    # Determine the correct base directory for static files
    # Assuming 'static' folder is at the same level as gui.py or main.py
    static_dir = os.path.join(os.path.dirname(__file__), 'static') 
    # If static files are elsewhere, adjust this path accordingly
    # Example: static_dir = os.path.join(app_directories.get('base_dir', '.'), 'static')
    logger.debug(f"Serving static file from: {static_dir} / {subpath}")
    return send_from_directory(static_dir, subpath)


# --- API Endpoints --- 

@app.route('/api/interfaces', methods=['GET', 'POST'])
def handle_interfaces():
    network_manager = get_network_interface_manager()
    if request.method == 'GET':
        try:
            interfaces = network_manager.get_interface_names()
            logger.info(f"Fetched interfaces: {interfaces}")
            return jsonify({'interfaces': interfaces})
        except Exception as e:
            logger.error(f"Error fetching interfaces: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
            
    elif request.method == 'POST':
        try:
            data = request.get_json()
            selected_interface = data.get('interface')
            if not selected_interface:
                return jsonify({'success': False, 'message': 'No interface provided'}), 400
            
            # Validate interface?
            interfaces = network_manager.get_interface_names()
            if selected_interface not in interfaces:
                 return jsonify({'success': False, 'message': f'Invalid interface: {selected_interface}'}), 400

            app_state.set('selected_interface', selected_interface)
            logger.info(f"Interface selected via API: {selected_interface}")
            # Notify clients via SocketIO (optional, if needed immediately)
            socketio.emit('interface_update', {'interface': selected_interface})
            return jsonify({'success': True, 'message': f'Interface set to {selected_interface}'})
        except Exception as e:
            logger.error(f"Error setting interface: {e}", exc_info=True)
            return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        if 'model_ensemble' in app_components:
            model_names = app_components['model_ensemble'].get_model_names()
            logger.info(f"Fetched models: {model_names}")
            return jsonify({'models': model_names})
        else:
             logger.warning("Model ensemble not found in components.")
             return jsonify({'models': []})
    except Exception as e:
        logger.error(f"Error fetching models: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    try:
        status_data = {
            'is_running': app_state.get('is_running', False),
            'selected_interface': app_state.get('selected_interface'),
            'packet_count': app_state.get('packet_count', 0),
            'alert_count': app_state.get('alert_count', 0),
            # Add other relevant status info
        }
        return jsonify(status_data)
    except Exception as e:
        logger.error(f"Error fetching status: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs', methods=['GET'])
def get_logs():
    # Basic implementation: read the last N lines from the main log file
    # Assumes LoggingManager uses a file handler
    try:
        log_file_path = None
        if 'logging_manager' in app_components:
             # Attempt to get the log file path from the manager if possible
             # This might require adding a method to LoggingManager
             # For now, assume a default path based on directories
             log_dir = app_directories.get('log_dir', 'logs')
             # Assuming the main log file is named 'main.log' or similar
             # Need to confirm the actual filename used by LoggingManager
             potential_log_file = os.path.join(log_dir, 'main.log') # Adjust filename if needed
             if os.path.exists(potential_log_file):
                 log_file_path = potential_log_file
             else:
                 # Fallback or alternative log file name?
                 logger.warning(f"Main log file not found at {potential_log_file}")
                 # Check for gui.log as a fallback? (Though basicConfig was removed)
                 gui_log_file = os.path.join(log_dir, 'gui.log')
                 if os.path.exists(gui_log_file):
                     log_file_path = gui_log_file
                     logger.info("Reading from gui.log as fallback.")
                 else:
                     logger.error("Could not determine log file path.")
                     return jsonify({'logs': 'Error: Log file path not found.'}), 500
        else:
            logger.error("LoggingManager not found in components.")
            return jsonify({'logs': 'Error: LoggingManager not configured.'}), 500

        if log_file_path:
            lines_to_fetch = request.args.get('lines', 100, type=int)
            logs = []
            with open(log_file_path, 'r') as f:
                # Read lines efficiently (especially for large files)
                # This is a simple approach, might need optimization
                all_lines = f.readlines()
                logs = all_lines[-lines_to_fetch:]
            logger.info(f"Fetched last {len(logs)} lines from {log_file_path}")
            return jsonify({'logs': ''.join(logs)})
        else:
             # Should have been handled above, but as a safeguard
             return jsonify({'logs': 'Error: Log file path could not be determined.'}), 500

    except Exception as e:
        logger.error(f"Error fetching logs: {e}", exc_info=True)
        return jsonify({'logs': f'Error fetching logs: {e}'}), 500

# --- SocketIO Event Handlers --- 

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    # Send initial status on connect
    emit('status_update', {
        'is_running': app_state.get('is_running', False),
        'selected_interface': app_state.get('selected_interface')
    })

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('request_initial_data')
def handle_initial_data_request():
    """Send current state data when requested by a client (e.g., after page load)."""
    logger.debug(f"Received request_initial_data from {request.sid}")
    emit('status_update', {
        'is_running': app_state.get('is_running', False),
        'selected_interface': app_state.get('selected_interface')
    })
    emit('stats_update', {
        'packet_count': app_state.get('packet_count', 0),
        'alert_count': app_state.get('alert_count', 0)
        # Add other stats if tracked
    })
    # Optionally send recent alerts/packets if stored

@socketio.on('interface_selected')
def handle_interface_selection(data):
    """Handle interface selection event from client (alternative to POST API)."""
    selected_interface = data.get('interface')
    if selected_interface:
        # Optional: Validate interface name against available ones
        network_manager = get_network_interface_manager()
        interfaces = network_manager.get_interface_names()
        if selected_interface in interfaces:
            app_state.set('selected_interface', selected_interface)
            logger.info(f"Interface selected via SocketIO: {selected_interface}")
            emit('interface_update', {'interface': selected_interface}, broadcast=True)
        else:
            logger.warning(f"Invalid interface selected via SocketIO: {selected_interface}")
            # Optionally emit an error back to the specific client
            emit('error_message', {'message': f'Invalid interface: {selected_interface}'})
    else:
        logger.warning("Received interface_selected event with no interface data.")

@socketio.on('start_capture')
def handle_start_capture():
    logger.info("Received start_capture request")

    if app_state.get('is_running'):
        logger.warning("Capture is already running.")
        socketio.emit('stats_update', {'packet_count': 0, 'alert_count': 0})
        return

    selected_interface = app_state.get('selected_interface')
    if not selected_interface:
        logger.error("Start requested but no interface selected.")
        socketio.emit('error_message', {'message': 'Cannot start: No interface selected.'})
        socketio.emit('status_update', {'is_running': False})
        return

    packet_sniffer = app_components.get('packet_sniffer')
    if not packet_sniffer:
        logger.error("Packet sniffer component not found.")
        socketio.emit('error_message', {'message': 'Internal error: Packet sniffer not available.'})
        return

    app_state.set('packet_count', 0)
    app_state.set('alert_count', 0)
    socketio.emit('stats_update', {'packet_count': 0, 'alert_count': 0})

    try:
        logger.info(f"Attempting to start capture on interface: {selected_interface}")

        if hasattr(packet_sniffer, 'set_socketio_instance'):
            packet_sniffer.set_socketio_instance(socketio, app_state)
        elif 'packet_processor' in app_components and hasattr(app_components['packet_processor'], 'set_socketio_instance'):
            app_components['packet_processor'].set_socketio_instance(socketio, app_state)
        else:
            logger.warning("Could not set socketio instance for data emission in backend.")

        capture_thread = threading.Thread(
            target=packet_sniffer.start_capture,
            kwargs={'interface': selected_interface},
            daemon=True
        )
        capture_thread.start()
        time.sleep(1)

        app_state.set('is_running', True)
        logger.info(f"Capture started successfully on {selected_interface} (assumed).")
        socketio.emit('status_update', {
            'is_running': True,
            'selected_interface': selected_interface,
            'message': f'Capture started on {selected_interface}'
        })

    except Exception as e:
        logger.error(f"Failed to start packet capture: {e}", exc_info=True)
        app_state.set('is_running', False)
        socketio.emit('error_message', {'message': f'Failed to start capture: {e}'})
        socketio.emit('status_update', {'is_running': False})

@socketio.on('stop_capture')
def handle_stop_capture():
    logger.info("Received stop_capture request")
    if not app_state.get('is_running'):
        logger.warning("Capture is not running.")
        emit('status_update', {'is_running': False, 'message': 'Capture already stopped.'})
        return

    packet_sniffer = app_components.get('packet_sniffer')
    if not packet_sniffer:
        logger.error("Packet sniffer component not found.")
        emit('error_message', {'message': 'Internal error: Packet sniffer not available.'})
        return

    try:
        packet_sniffer.stop_capture()
        app_state.set('is_running', False)
        logger.info("Capture stopped successfully.")
        emit('status_update', {'is_running': False, 'message': 'Capture stopped.'}, broadcast=True)
    except Exception as e:
        logger.error(f"Failed to stop packet capture: {e}", exc_info=True)
        # Keep state as running? Or force stop?
        # app_state.set_state('is_running', False) # Assume stop worked despite error?
        emit('error_message', {'message': f'Error stopping capture: {e}'})
        # Optionally re-emit running status if stop failed critically
        # emit('status_update', {'is_running': app_state.get_state('is_running')}, broadcast=True)

# --- Helper Functions for Backend Emission --- 
# These functions can be called from other modules (like packet processing)
# Requires passing the 'socketio' object and 'app_state' dict to them.

def emit_packet_data(packet_info):
    """Emit packet data to clients."""
    try:
        # Increment packet count
        current_count = app_state.update_and_get('packet_count', lambda x: (x or 0) + 1)
        
        # Emit limited packet info and updated count
        socketio.emit('new_packet', {
            'info': packet_info, # Send selected fields, not the raw packet
            'packet_count': current_count
        })
    except Exception as e:
        logger.error(f"Error emitting packet data: {e}", exc_info=True)

def emit_alert(alert_details):
    """Emit alert data to clients."""
    try:
        # Increment alert count
        current_count = app_state.update_and_get('alert_count', lambda x: (x or 0) + 1)
        
        # Emit alert details and updated count
        socketio.emit('new_alert', {
            'alert': alert_details,
            'alert_count': current_count
        })
    except Exception as e:
        logger.error(f"Error emitting alert data: {e}", exc_info=True)

def emit_stats_update(stats_data):
    """Emit statistics updates to clients."""
    try:
        # Update central state if necessary (e.g., packet/alert counts are primary)
        # stats_data might include other metrics
        packet_count = app_state.get('packet_count', 0)
        alert_count = app_state.get('alert_count', 0)
        full_stats = {
            'packet_count': packet_count,
            'alert_count': alert_count,
            **stats_data # Merge other stats
        }
        socketio.emit('stats_update', full_stats)
    except Exception as e:
        logger.error(f"Error emitting stats update: {e}", exc_info=True)


