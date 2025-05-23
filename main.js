// Main JavaScript for AI-Enhanced Intrusion Detection System

// Ensure Socket.IO library is loaded (usually in index.html)
// const socket = io(); // Assuming io() is available globally

document.addEventListener("DOMContentLoaded", () => {
    // Check if socket is defined, if not, maybe log an error or wait
    if (typeof io === "undefined") {
        console.error("Socket.IO library not found. Ensure it is loaded in your HTML.");
        return;
    }
    const socket = io();

    // --- Global Elements (present on multiple pages) ---
    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");
    const packetCountSpan = document.getElementById("packet-count"); // e.g., in navbar or dashboard
    const alertCountSpan = document.getElementById("alert-count");   // e.g., in navbar or dashboard

    // --- Page Specific Elements --- 
    // Settings Page
    const startBtn = document.getElementById("start-btn");
    const stopBtn = document.getElementById("stop-btn");
    const interfaceSelect = document.getElementById("interface-select");
    const modelsList = document.getElementById("available-models-list");

    // Packets Page
    const packetTableBody = document.querySelector("#packet-table tbody");

    // Alerts Page
    const alertTableBody = document.querySelector("#alert-table tbody");

    // Statistics Page (Example elements)
    const statsPacketCount = document.getElementById("stat-packet-count");
    const statsAlertCount = document.getElementById("stat-alert-count");
    const statsSelectedInterface = document.getElementById("stat-selected-interface");
    const statsIsRunning = document.getElementById("stat-is-running");

    // --- Initial State --- 
    let isRunning = false;
    let selectedInterface = null;

    // --- SocketIO Event Handlers ---
    socket.on("connect", () => {
        console.log("Connected to server");
        showNotification("Connected to server", "success");
        // Request initial data when connected
        socket.emit("request_initial_data");
    });

    socket.on("disconnect", () => {
        console.log("Disconnected from server");
        updateStatusIndicator(false);
        showNotification("Disconnected from server", "error");
    });

    socket.on("status_update", (data) => {
        console.log("Status update received:", data);
        isRunning = data.is_running;
        selectedInterface = data.selected_interface;
        updateStatusIndicator(isRunning);
        updateButtonStates(isRunning);
        updateStatsPage(); // Update stats page if currently viewed
        if (data.message) {
            showNotification(data.message, "info");
        }
    });

    socket.on("interface_update", (data) => {
        console.log("Interface update received:", data);
        selectedInterface = data.interface;
        // Update selection in dropdown if on settings page
        if (interfaceSelect && interfaceSelect.value !== selectedInterface) {
            interfaceSelect.value = selectedInterface;
        }
        updateStatsPage(); // Update stats page if currently viewed
    });

    socket.on("stats_update", (data) => {
        console.log("Stats update received:", data);
        if (packetCountSpan) packetCountSpan.textContent = data.packet_count ?? "0";
        if (alertCountSpan) alertCountSpan.textContent = data.alert_count ?? "0";
        updateStatsPage(data); // Update stats page if currently viewed
    });

    socket.on("new_packet", (data) => {
        // console.log("New packet data:", data);
        if (packetCountSpan) packetCountSpan.textContent = data.packet_count ?? "N/A";
        if (packetTableBody) {
            addPacketToTable(data.info);
        }
        updateStatsPage({ packet_count: data.packet_count });
    });

    socket.on("new_alert", (data) => {
        console.log("New alert data:", data);
        if (alertCountSpan) alertCountSpan.textContent = data.alert_count ?? "N/A";
        if (alertTableBody) {
            addAlertToTable(data.alert);
        }
        showNotification(`Alert: ${data.alert?.prediction || "Unknown"} (Confidence: ${data.alert?.confidence?.toFixed(2) || "N/A"})`, "warning");
        updateStatsPage({ alert_count: data.alert_count });
    });

    socket.on("error_message", (data) => {
        console.error("Error from server:", data.message);
        showNotification(data.message, "error");
    });

    // --- UI Update Functions ---
    function updateStatusIndicator(running) {
        if (statusDot && statusText) {
            if (running) {
                statusDot.classList.remove("offline");
                statusDot.classList.add("online");
                statusText.textContent = "Online";
            } else {
                statusDot.classList.remove("online");
                statusDot.classList.add("offline");
                statusText.textContent = "Offline";
            }
        }
    }

    function updateButtonStates(running) {
        if (startBtn) startBtn.disabled = running;
        if (stopBtn) stopBtn.disabled = !running;
    }

    function updateStatsPage(latestStats = null) {
        if (statsPacketCount) {
            statsPacketCount.textContent = latestStats?.packet_count ?? packetCountSpan?.textContent ?? "0";
        }
        if (statsAlertCount) {
            statsAlertCount.textContent = latestStats?.alert_count ?? alertCountSpan?.textContent ?? "0";
        }
        if (statsSelectedInterface) {
            statsSelectedInterface.textContent = selectedInterface || "None";
        }
        if (statsIsRunning) {
            statsIsRunning.textContent = isRunning ? "Yes" : "No";
            statsIsRunning.style.color = isRunning ? "var(--success-color)" : "var(--accent-color)";
        }
    }

    function addPacketToTable(packetInfo) {
        if (!packetTableBody || !packetInfo) return;
        const row = packetTableBody.insertRow(0); // Add to top
        row.insertCell(0).textContent = formatDate(packetInfo.timestamp);
        row.insertCell(1).textContent = packetInfo.src_ip || "N/A";
        row.insertCell(2).textContent = packetInfo.dst_ip || "N/A";
        row.insertCell(3).textContent = packetInfo.protocol || "N/A";
        row.insertCell(4).textContent = packetInfo.length || "N/A"; // Assuming length is available
        // Limit table size
        if (packetTableBody.rows.length > 100) { // Keep last 100 packets
            packetTableBody.deleteRow(-1);
        }
    }

    function addAlertToTable(alertDetails) {
        if (!alertTableBody || !alertDetails) return;
        const row = alertTableBody.insertRow(0); // Add to top
        row.insertCell(0).textContent = formatDate(alertDetails.timestamp);
        row.insertCell(1).textContent = alertDetails.src_ip || "N/A";
        row.insertCell(2).textContent = alertDetails.dst_ip || "N/A";
        row.insertCell(3).textContent = alertDetails.protocol || "N/A";
        row.insertCell(4).textContent = alertDetails.prediction || "N/A";
        row.insertCell(5).textContent = alertDetails.confidence?.toFixed(4) || "N/A";
        row.classList.add(alertDetails.is_alert ? "alert-row" : "normal-row");
        // Limit table size
        if (alertTableBody.rows.length > 100) { // Keep last 100 alerts
            alertTableBody.deleteRow(-1);
        }
    }

    // --- Event Listeners Setup (Settings Page Specific) ---
    if (startBtn) {
        startBtn.addEventListener("click", () => {
            if (!selectedInterface) {
                showNotification("Please select a network interface first.", "error");
                return;
            }
            console.log("Requesting start capture...");
            socket.emit("start_capture");
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener("click", () => {
            console.log("Requesting stop capture...");
            socket.emit("stop_capture");
        });
    }

    if (interfaceSelect) {
        interfaceSelect.addEventListener("change", (event) => {
            const newInterface = event.target.value;
            if (newInterface) {
                console.log(`Interface selected by user: ${newInterface}`);
                selectedInterface = newInterface; // Update local state immediately
                // Send selection to backend via SocketIO
                socket.emit("interface_selected", { interface: newInterface });
                // Alternatively, use the POST API (less real-time)
                // fetch("/api/interfaces", { 
                //     method: "POST", 
                //     headers: { "Content-Type": "application/json" },
                //     body: JSON.stringify({ interface: newInterface })
                // }).then(res => res.json()).then(data => {
                //     if(data.success) showNotification("Interface selection saved", "success");
                //     else showNotification(`Error saving interface: ${data.message}`, "error");
                // });
            }
        });
    }

    // --- Initialization --- 
    function initializePage() {
        // Set active navigation link
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll(".nav-link");
        navLinks.forEach(link => {
            link.classList.remove("active"); // Remove active from all first
            const linkPath = link.getAttribute("href");
            // Handle root path mapping to dashboard
            const isActive = (currentPath === linkPath) || (currentPath === "/" && linkPath === "/dashboard");
            if (isActive) {
                link.classList.add("active");
            }
        });

        // Initial UI state based on defaults
        updateStatusIndicator(isRunning);
        updateButtonStates(isRunning);

        // Fetch initial data if needed (though socket connect handles most)
        // Example: Fetch interfaces and models specifically for settings page
        if (window.location.pathname.includes("settings")) {
            fetchInterfaces();
            fetchModels();
        }
        
        // Update stats page on load
        if (window.location.pathname.includes("statistics")) {
             updateStatsPage();
        }
    }

    // --- Fetch Functions (for Settings Page) ---
    function fetchInterfaces() {
        if (!interfaceSelect) return;
        fetch("/api/interfaces")
            .then(response => response.json())
            .then(data => {
                if (data.interfaces) {
                    interfaceSelect.innerHTML = 
`<option value="" disabled ${selectedInterface ? "" : "selected"}>Select Interface...</option>`
;
                    data.interfaces.forEach(iface => {
                        const option = document.createElement("option");
                        option.value = iface;
                        option.textContent = iface;
                        if (iface === selectedInterface) {
                            option.selected = true;
                        }
                        interfaceSelect.appendChild(option);
                    });
                } else {
                    console.error("Failed to fetch interfaces:", data.error);
                }
            })
            .catch(error => console.error("Error fetching interfaces:", error));
    }

    function fetchModels() {
        if (!modelsList) return;
        fetch("/api/models")
            .then(response => response.json())
            .then(data => {
                modelsList.innerHTML = ""; // Clear previous list
                if (data.models && data.models.length > 0) {
                    data.models.forEach(modelName => {
                        const li = document.createElement("li");
                        li.textContent = modelName;
                        modelsList.appendChild(li);
                    });
                } else if (data.models) {
                     modelsList.innerHTML = "<li>No models found or loaded.</li>";
                } else {
                    console.error("Failed to fetch models:", data.error);
                    modelsList.innerHTML = "<li>Error loading models.</li>";
                }
            })
            .catch(error => {
                console.error("Error fetching models:", error);
                modelsList.innerHTML = "<li>Error loading models.</li>";
            });
    }

    // --- Utility Functions ---
    function showNotification(message, type = "info") {
        const container = document.getElementById("notification-container") || document.body;
        const notification = document.createElement("div");
        notification.className = `notification ${type}`;
        notification.textContent = message;
        container.appendChild(notification);
        // Trigger reflow to enable animation
        void notification.offsetWidth;
        notification.classList.add("show");

        setTimeout(() => {
            notification.classList.remove("show");
            setTimeout(() => {
                if (container.contains(notification)) {
                     container.removeChild(notification);
                }
            }, 500); // Wait for fade out animation
        }, 3000); // Notification duration
    }

    function formatDate(timestamp) {
        if (!timestamp) return "N/A";
        try {
            const date = new Date(timestamp * 1000);
            return date.toLocaleString();
        } catch (e) {
            return "Invalid Date";
        }
    }

    // --- Run Initialization ---
    initializePage();
});

