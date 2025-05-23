document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    const interfaceSelect = document.getElementById('interface-select');
    const selectInterfaceBtn = document.getElementById('select-interface-btn');
    const refreshInterfacesBtn = document.getElementById('refresh-interfaces-btn');

    // Fetch and populate interfaces
    function updateInterfaces() {
        fetch('/api/interfaces', { method: 'GET' })
            .then(response => response.json())
            .then(data => {
                interfaceSelect.innerHTML = '<option value="">Select an interface...</option>';
                data.interfaces.forEach(interface => {
                    const option = document.createElement('option');
                    option.value = interface;
                    option.textContent = interface;
                    interfaceSelect.appendChild(option);
                });
            })
            .catch(error => console.error('Error fetching interfaces:', error));
    }

    // Handle interface selection
    selectInterfaceBtn.addEventListener('click', () => {
        const selectedInterface = interfaceSelect.value;
        if (selectedInterface) {
            fetch('/api/interfaces', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ interface: selectedInterface })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    socket.emit('interface_selected', { interface: selectedInterface });
                    console.log(`Interface selected: ${selectedInterface}`);
                } else {
                    console.error('Failed to select interface:', data.message);
                }
            })
            .catch(error => console.error('Error:', error));
        }
    });

    // Refresh interfaces
    refreshInterfacesBtn.addEventListener('click', updateInterfaces);

    // Initial load
    updateInterfaces();

    // Listen for interface updates
    socket.on('interface_update', (data) => {
        console.log(`Interface updated to: ${data.interface}`);
        // Update UI if needed (e.g., display selected interface)
    });
});