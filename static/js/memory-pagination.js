// Memory pagination and sorting functionality

let currentPage = 1;
let itemsPerPage = 20;
let currentSort = { field: 'when', direction: 'desc' };
let filteredMemories = [];
let lastUserInteraction = 0; // Track last user interaction time

// Export lastUserInteraction globally
window.lastUserInteraction = 0;

// Make sortTable globally accessible
window.sortTable = function(field) {
    // Track user interaction
    lastUserInteraction = Date.now();
    window.lastUserInteraction = lastUserInteraction;
    
    // Toggle direction if same field
    if (currentSort.field === field) {
        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
    } else {
        currentSort.field = field;
        currentSort.direction = 'asc';
    }
    
    // Update sort icons
    document.querySelectorAll('.sortable .sort-icon').forEach(icon => {
        icon.className = 'bi bi-arrow-down-up sort-icon';
    });
    
    const currentHeader = document.querySelector(`[data-field="${field}"] .sort-icon`);
    if (currentHeader) {
        currentHeader.className = currentSort.direction === 'asc' 
            ? 'bi bi-arrow-up sort-icon text-cyan' 
            : 'bi bi-arrow-down sort-icon text-cyan';
    }
    
    // Sort memories
    sortMemories();
    
    // Reset to first page after sorting
    currentPage = 1;
    displayMemoriesWithPaging();
}

// Sort memories array
function sortMemories() {
    filteredMemories.sort((a, b) => {
        let aVal = a[currentSort.field] || '';
        let bVal = b[currentSort.field] || '';
        
        // Handle date sorting
        if (currentSort.field === 'when') {
            aVal = new Date(aVal).getTime() || 0;
            bVal = new Date(bVal).getTime() || 0;
        }
        
        // Handle string comparison
        if (typeof aVal === 'string') {
            aVal = aVal.toLowerCase();
            bVal = bVal.toLowerCase();
        }
        
        if (currentSort.direction === 'asc') {
            return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
        } else {
            return aVal < bVal ? 1 : aVal > bVal ? -1 : 0;
        }
    });
}

// Display memories with pagination
function displayMemoriesWithPaging() {
    const tableBody = document.getElementById('memoryTableBody');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    // Calculate pagination
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = Math.min(startIndex + itemsPerPage, filteredMemories.length);
    const pageMemories = filteredMemories.slice(startIndex, endIndex);
    
    // Display memories with all columns
    pageMemories.forEach(memory => {
        const row = createMemoryRow(memory, true); // true = includeAll columns
        tableBody.appendChild(row);
    });
    
    // Update pagination info
    document.getElementById('showingStart').textContent = filteredMemories.length > 0 ? startIndex + 1 : 0;
    document.getElementById('showingEnd').textContent = endIndex;
    document.getElementById('totalMemories').textContent = filteredMemories.length;
    
    // Restore sort icons based on current sort state
    document.querySelectorAll('.sortable .sort-icon').forEach(icon => {
        icon.className = 'bi bi-arrow-down-up sort-icon';
    });
    
    if (currentSort && currentSort.field) {
        const currentHeader = document.querySelector(`[data-field="${currentSort.field}"] .sort-icon`);
        if (currentHeader) {
            currentHeader.className = currentSort.direction === 'asc' 
                ? 'bi bi-arrow-up sort-icon text-cyan' 
                : 'bi bi-arrow-down sort-icon text-cyan';
        }
    }
    
    // Create pagination controls
    createPaginationControls();
}

// Create pagination controls
function createPaginationControls() {
    const paginationControls = document.getElementById('paginationControls');
    if (!paginationControls) return;
    
    paginationControls.innerHTML = '';
    
    const totalPages = Math.ceil(filteredMemories.length / itemsPerPage);
    
    // Previous button
    const prevLi = document.createElement('li');
    prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    prevLi.innerHTML = `<a class="page-link" href="#" onclick="changePage(${currentPage - 1}); return false;">Previous</a>`;
    paginationControls.appendChild(prevLi);
    
    // Page numbers
    const maxButtons = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxButtons / 2));
    let endPage = Math.min(totalPages, startPage + maxButtons - 1);
    
    if (endPage - startPage < maxButtons - 1) {
        startPage = Math.max(1, endPage - maxButtons + 1);
    }
    
    // First page if not shown
    if (startPage > 1) {
        const firstLi = document.createElement('li');
        firstLi.className = 'page-item';
        firstLi.innerHTML = `<a class="page-link" href="#" onclick="changePage(1); return false;">1</a>`;
        paginationControls.appendChild(firstLi);
        
        if (startPage > 2) {
            const ellipsisLi = document.createElement('li');
            ellipsisLi.className = 'page-item disabled';
            ellipsisLi.innerHTML = '<span class="page-link">...</span>';
            paginationControls.appendChild(ellipsisLi);
        }
    }
    
    // Page number buttons
    for (let i = startPage; i <= endPage; i++) {
        const li = document.createElement('li');
        li.className = `page-item ${i === currentPage ? 'active' : ''}`;
        li.innerHTML = `<a class="page-link" href="#" onclick="changePage(${i}); return false;">${i}</a>`;
        paginationControls.appendChild(li);
    }
    
    // Last page if not shown
    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            const ellipsisLi = document.createElement('li');
            ellipsisLi.className = 'page-item disabled';
            ellipsisLi.innerHTML = '<span class="page-link">...</span>';
            paginationControls.appendChild(ellipsisLi);
        }
        
        const lastLi = document.createElement('li');
        lastLi.className = 'page-item';
        lastLi.innerHTML = `<a class="page-link" href="#" onclick="changePage(${totalPages}); return false;">${totalPages}</a>`;
        paginationControls.appendChild(lastLi);
    }
    
    // Next button
    const nextLi = document.createElement('li');
    nextLi.className = `page-item ${currentPage === totalPages || totalPages === 0 ? 'disabled' : ''}`;
    nextLi.innerHTML = `<a class="page-link" href="#" onclick="changePage(${currentPage + 1}); return false;">Next</a>`;
    paginationControls.appendChild(nextLi);
}

// Change page - make globally accessible
window.changePage = function(page) {
    // Track user interaction
    lastUserInteraction = Date.now();
    window.lastUserInteraction = lastUserInteraction;
    
    const totalPages = Math.ceil(filteredMemories.length / itemsPerPage);
    
    if (page < 1 || page > totalPages) return;
    
    currentPage = page;
    displayMemoriesWithPaging();
}

// Filter memories based on search
function filterMemories(searchTerm = '') {
    const term = searchTerm.toLowerCase();
    
    if (!term) {
        filteredMemories = [...allMemories.raw];
    } else {
        filteredMemories = allMemories.raw.filter(memory => {
            return (
                (memory.who || '').toLowerCase().includes(term) ||
                (memory.what || '').toLowerCase().includes(term) ||
                (memory.where || '').toLowerCase().includes(term) ||
                (memory.why || '').toLowerCase().includes(term) ||
                (memory.how || '').toLowerCase().includes(term) ||
                (memory.type || '').toLowerCase().includes(term)
            );
        });
    }
    
    sortMemories();
    currentPage = 1;
    displayMemoriesWithPaging();
}

// Override the displayMemories function
window.displayMemories = function() {
    if (!allMemories.raw) {
        allMemories.raw = [];
    }
    
    // Initialize filtered memories
    filteredMemories = [...allMemories.raw];
    
    // Preserve current sort if it exists, otherwise default to when desc
    if (!currentSort || !currentSort.field) {
        currentSort = { field: 'when', direction: 'desc' };
    }
    
    // Apply current sort
    sortMemories();
    
    // Display with pagination (preserving current page if valid)
    const totalPages = Math.ceil(filteredMemories.length / itemsPerPage);
    if (currentPage > totalPages) {
        currentPage = 1;
    }
    displayMemoriesWithPaging();
};

// Add search functionality
document.addEventListener('DOMContentLoaded', function() {
    const searchBox = document.getElementById('memorySearch');
    if (searchBox) {
        searchBox.addEventListener('input', function(e) {
            // Track user interaction
            lastUserInteraction = Date.now();
            window.lastUserInteraction = lastUserInteraction;
            filterMemories(e.target.value);
        });
    }
});
