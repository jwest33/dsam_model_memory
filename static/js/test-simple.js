console.log('Test simple JS loaded');

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOMContentLoaded in test-simple.js');
    
    // Direct API call test
    fetch('/api/memories')
        .then(response => {
            console.log('API Response status:', response.status);
            return response.json();
        })
        .then(data => {
            console.log('API Data:', data);
            
            // Try to display data directly
            const tbody = document.getElementById('memoryTableBody');
            if (tbody) {
                console.log('Found tbody element');
                if (data.memories && data.memories.length > 0) {
                    tbody.innerHTML = '<tr><td colspan="8">Found ' + data.memories.length + ' memories</td></tr>';
                } else {
                    tbody.innerHTML = '<tr><td colspan="8">No memories found</td></tr>';
                }
            } else {
                console.log('Could not find memoryTableBody element');
            }
        })
        .catch(error => {
            console.error('API Error:', error);
        });
});
