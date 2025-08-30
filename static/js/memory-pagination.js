// Memory pagination and sorting functionality
(function() {
    'use strict';
    
    // Local variables scoped to this module
    let paginationCurrentPage = 1;
    let paginationItemsPerPage = 20;
    let paginationCurrentSort = { field: 'when', direction: 'desc' };
    let paginationFilteredMemories = [];
    let paginationLastUserInteraction = 0;
    
    // Export lastUserInteraction globally
    window.lastUserInteraction = 0;
    
    // Function to create memory table row
    window.createMemoryRow = function(memory, includeAll = false) {
        
        const row = document.createElement('tr');
        row.style.cursor = 'pointer';
        row.classList.add('memory-row');
        row.setAttribute('data-memory-id', memory.id);
        
        // Add click handler for the entire row
        row.addEventListener('click', function(e) {
            // Don't trigger if clicking on a button or icon
            if (e.target.closest('button') || e.target.closest('i')) {
                return;
            }
            
            // Call viewMemoryDetails directly - simpler and more reliable
            if (window.viewMemoryDetails) {
                window.viewMemoryDetails(memory.id);
            }
        });
        
        // Get residual indicator  
        const residualIndicator = typeof getResidualIndicator === 'function' ? getResidualIndicator(memory) : '';
        
        // Handle potentially empty fields
        const who = memory.who || '';
        const what = memory.what || '';
        const when = memory.when || '';
        const where = memory.where || '';
        const why = memory.why || '';
        const how = memory.how || '';
        const whatDisplay = what.length > 40 ? what.substring(0, 40) + '...' : what;
        const whereDisplay = where.length > 20 ? where.substring(0, 20) + '...' : where;
        const whyDisplay = why.length > 30 ? why.substring(0, 30) + '...' : why;
        const howDisplay = how.length > 25 ? how.substring(0, 25) + '...' : how;
        
        // Calculate space weight for this memory
        const memorySpaceWeight = typeof calculateMemorySpaceWeight === 'function' ? calculateMemorySpaceWeight(memory) : '';
        
        // Helper to escape attributes (handles quotes)
        const escapeAttr = (text) => {
            if (!text) return '';
            return text
                .replace(/&/g, '&amp;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
        };
        
        row.innerHTML = `
            <td>${escapeHtml(who)}</td>
            <td title="${escapeAttr(what)}">${escapeHtml(whatDisplay)}</td>
            <td>${typeof formatDate === 'function' ? formatDate(when) : when}</td>
            <td>${escapeHtml(whereDisplay)}</td>
            <td title="${escapeAttr(why)}">${escapeHtml(whyDisplay)}</td>
            <td title="${escapeAttr(how)}">${escapeHtml(howDisplay)}</td>
            <td>${memorySpaceWeight}</td>
            <td>${residualIndicator}</td>
            <td>
                <button class="btn btn-sm btn-outline-info" onclick="window.viewMemoryDetails('${memory.id}')" title="View Details">
                    <i class="bi bi-eye"></i>
                </button>
                <button class="btn btn-sm btn-outline-primary" onclick="window.showMemoryInGraph('${memory.id}')" title="Show in Graph">
                    <i class="bi bi-diagram-3"></i>
                </button>
                <button class="btn btn-sm btn-outline-danger" onclick="window.deleteMemory('${memory.id}')" title="Delete">
                    <i class="bi bi-trash"></i>
                </button>
            </td>
        `;
        
        return row;
    }
    
    // Make sortTable globally accessible
    window.sortTable = function(field) {
        // Track user interaction
        paginationLastUserInteraction = Date.now();
        window.lastUserInteraction = paginationLastUserInteraction;
        
        // Toggle direction if same field
        if (paginationCurrentSort.field === field) {
            paginationCurrentSort.direction = paginationCurrentSort.direction === 'asc' ? 'desc' : 'asc';
        } else {
            paginationCurrentSort.field = field;
            paginationCurrentSort.direction = 'asc';
        }
        
        // Update sort icons
        document.querySelectorAll('.sortable .sort-icon').forEach(icon => {
            icon.className = 'bi bi-arrow-down-up sort-icon';
        });
        
        const currentHeader = document.querySelector(`[data-field="${field}"] .sort-icon`);
        if (currentHeader) {
            currentHeader.className = paginationCurrentSort.direction === 'asc' 
                ? 'bi bi-arrow-up sort-icon text-cyan' 
                : 'bi bi-arrow-down sort-icon text-cyan';
        }
        
        // Sort memories
        sortMemories();
        
        // Reset to first page after sorting
        paginationCurrentPage = 1;
        displayMemoriesWithPaging();
    }
    
    // Sort memories array
    function sortMemories() {
        paginationFilteredMemories.sort((a, b) => {
            let aVal = a[paginationCurrentSort.field] || '';
            let bVal = b[paginationCurrentSort.field] || '';
            
            // Handle date sorting
            if (paginationCurrentSort.field === 'when') {
                aVal = new Date(aVal).getTime() || 0;
                bVal = new Date(bVal).getTime() || 0;
            }
            
            // Handle string comparison
            if (typeof aVal === 'string') {
                aVal = aVal.toLowerCase();
                bVal = bVal.toLowerCase();
            }
            
            if (paginationCurrentSort.direction === 'asc') {
                return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
            } else {
                return aVal < bVal ? 1 : aVal > bVal ? -1 : 0;
            }
        });
    }
    
    // Display memories with pagination
    // Helper function to create a memory row
    function createMemoryRow(memory, includeAll) {
        const row = document.createElement('tr');
        row.style.cursor = 'pointer';
        row.classList.add('memory-row');
        row.setAttribute('data-memory-id', memory.id);
        
        // Add click handler for the entire row
        row.addEventListener('click', function(e) {
            // Don't trigger if clicking on a button or icon
            if (e.target.closest('button') || e.target.closest('i')) {
                return;
            }
            
            // Call viewMemoryDetails
            if (window.viewMemoryDetails) {
                window.viewMemoryDetails(memory.id);
            }
        });
        
        // Fields should already be flattened by loadMemories
        const who = memory.who || '—';
        const what = memory.what || '—';  // Default to em dash if missing
        const when = memory.when || memory.timestamp || '—';
        const where = memory.where || '—';
        const why = memory.why || '—';
        const how = memory.how || '—';
        
        // Truncate long fields for display (but preserve the dash for empty fields)
        const whatDisplay = what === '—' ? '—' : (what.length > 40 ? what.substring(0, 40) + '...' : what);
        const whereDisplay = where === '—' ? '—' : (where.length > 20 ? where.substring(0, 20) + '...' : where);
        const whyDisplay = why === '—' ? '—' : (why.length > 30 ? why.substring(0, 30) + '...' : why);
        const howDisplay = how === '—' ? '—' : (how.length > 25 ? how.substring(0, 25) + '...' : how);
        
        // Add merge indicator for raw view
        let mergeIndicator = '';
        if (memory.type === 'raw' && memory.merged_id && window.mergeGroups) {
            const mergeGroup = window.mergeGroups[memory.merged_id];
            const groupSize = mergeGroup ? mergeGroup.length : 1;
            mergeIndicator = `<span class="badge bg-info ms-1" title="Part of merged group ${memory.merged_id.substring(0, 8)}">
                ${groupSize}
            </span>`;
        }
        
        // Helper to escape HTML
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Helper to escape HTML attributes (handles quotes)
        function escapeAttr(text) {
            if (!text) return '';
            return text
                .replace(/&/g, '&amp;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
        }
        
        // Helper to format date
        function formatDate(dateStr) {
            if (!dateStr) return '';
            try {
                const date = new Date(dateStr);
                return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
            } catch {
                return dateStr;
            }
        }
        
        // Helper to calculate space weight
        function calculateSpaceWeight(mem) {
            // Use real space weights if available
            if (mem.euclidean_weight !== undefined && mem.hyperbolic_weight !== undefined) {
                const euclideanPct = Math.round(mem.euclidean_weight * 100);
                const hyperbolicPct = Math.round(mem.hyperbolic_weight * 100);
                return `
                    <div class="d-flex align-items-center">
                        <div class="progress flex-grow-1" style="height: 15px; min-width: 80px;">
                            <div class="progress-bar bg-info" style="width: ${euclideanPct}%" 
                                 title="Euclidean: ${euclideanPct}%"></div>
                            <div class="progress-bar bg-warning" style="width: ${hyperbolicPct}%" 
                                 title="Hyperbolic: ${hyperbolicPct}%"></div>
                        </div>
                        <small class="ms-2">${euclideanPct}/${hyperbolicPct}</small>
                    </div>
                `;
            }
            
            // Fallback to field-based calculation
            let concreteScore = 0;
            let abstractScore = 0;
            
            // Score concrete fields - check if they have actual content (not empty or dash)
            if (who && who !== '—' && who.trim()) concreteScore += 1.0;
            if (what && what !== '—' && what.trim()) concreteScore += 2.0;
            if (when && when !== '—' && when.trim()) concreteScore += 0.5;
            if (where && where !== '—' && where.trim()) concreteScore += 0.5;
            
            // Score abstract fields - check if they have actual content (not empty or dash)
            if (why && why !== '—' && why.trim()) abstractScore += 1.5;
            if (how && how !== '—' && how.trim()) abstractScore += 1.0;
            
            const total = concreteScore + abstractScore;
            if (total === 0) {
                return '<span class="badge bg-secondary">No data</span>';
            }
            
            const euclideanPct = Math.round((concreteScore / total) * 100);
            const hyperbolicPct = Math.round((abstractScore / total) * 100);
            
            return `
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar bg-info" style="width: ${euclideanPct}%">${euclideanPct}%</div>
                    <div class="progress-bar bg-warning" style="width: ${hyperbolicPct}%">${hyperbolicPct}%</div>
                </div>
            `;
        }
        
        // Helper to get residual badge
        function getResidualBadge(memory) {
            const norm = memory.residual_norm || 0;
            if (norm < 0.1) {
                return '<span class="badge bg-success">Low</span>';
            } else if (norm < 0.3) {
                return '<span class="badge bg-warning">Medium</span>';
            } else {
                return '<span class="badge bg-danger">High</span>';
            }
        }
        
        // Build row HTML
        row.innerHTML = `
            <td>${escapeHtml(who)}</td>
            <td title="${escapeAttr(what)}">${escapeHtml(whatDisplay)}${mergeIndicator}</td>
            <td>${formatDate(when)}</td>
            <td>${escapeHtml(whereDisplay)}</td>
            <td title="${escapeAttr(why)}">${escapeHtml(whyDisplay)}</td>
            <td title="${escapeAttr(how)}">${escapeHtml(howDisplay)}</td>
            <td>${calculateSpaceWeight(memory)}</td>
            <td>${getResidualBadge(memory)}</td>
            <td>
                <button class="btn btn-sm btn-outline-info" onclick="window.viewMemoryDetails('${memory.id}')" title="View Details">
                    <i class="bi bi-eye"></i>
                </button>
                ${memory.type !== 'raw' ? 
                    `<button class="btn btn-sm btn-outline-primary" onclick="window.showMemoryInGraph('${memory.id}')" title="Show in Graph">
                        <i class="bi bi-diagram-3"></i>
                    </button>` : ''
                }
                ${memory.type === 'raw' && memory.merged_id ? 
                    `<button class="btn btn-sm btn-outline-warning" onclick="window.showMergeGroup('${memory.merged_id}')" title="Show Merge Group">
                        <i class="bi bi-collection"></i>
                    </button>` : 
                    `<button class="btn btn-sm btn-outline-danger" onclick="window.deleteMemory('${memory.id}')" title="Delete">
                        <i class="bi bi-trash"></i>
                    </button>`
                }
            </td>
        `;
        
        return row;
    }
    
    function displayMemoriesWithPaging() {
        const tableBody = document.getElementById('memoryTableBody');
        if (!tableBody) return;
        
        tableBody.innerHTML = '';
        
        // Calculate pagination
        const startIndex = (paginationCurrentPage - 1) * paginationItemsPerPage;
        const endIndex = Math.min(startIndex + paginationItemsPerPage, paginationFilteredMemories.length);
        const pageMemories = paginationFilteredMemories.slice(startIndex, endIndex);
        
        
        // Display memories with all columns
        pageMemories.forEach(memory => {
            const row = createMemoryRow(memory, true); // true = includeAll columns
            tableBody.appendChild(row);
        });
        
        // Update pagination info if elements exist
        const showingStart = document.getElementById('showingStart');
        const showingEnd = document.getElementById('showingEnd');
        const totalMemories = document.getElementById('totalMemories');
        const totalMemoriesInfo = document.getElementById('totalMemoriesInfo');
        
        if (showingStart) showingStart.textContent = paginationFilteredMemories.length > 0 ? startIndex + 1 : 0;
        if (showingEnd) showingEnd.textContent = endIndex;
        if (totalMemories) totalMemories.textContent = paginationFilteredMemories.length;
        if (totalMemoriesInfo) totalMemoriesInfo.textContent = paginationFilteredMemories.length;
        
        // Restore sort icons based on current sort state
        document.querySelectorAll('.sortable .sort-icon').forEach(icon => {
            icon.className = 'bi bi-arrow-down-up sort-icon';
        });
        
        if (paginationCurrentSort && paginationCurrentSort.field) {
            const currentHeader = document.querySelector(`[data-field="${paginationCurrentSort.field}"] .sort-icon`);
            if (currentHeader) {
                currentHeader.className = paginationCurrentSort.direction === 'asc' 
                    ? 'bi bi-arrow-up sort-icon text-cyan' 
                    : 'bi bi-arrow-down sort-icon text-cyan';
            }
        }
        
        // Create pagination controls
        createPaginationControls();
    }
    
    // Create pagination controls
    function createPaginationControls() {
        // Try both possible IDs for pagination controls
        let paginationControls = document.getElementById('paginationControls');
        if (!paginationControls) {
            paginationControls = document.getElementById('memoryPagination');
        }
        if (!paginationControls) return;
        
        paginationControls.innerHTML = '';
        
        const totalPages = Math.ceil(paginationFilteredMemories.length / paginationItemsPerPage);
        
        // Previous button
        const prevLi = document.createElement('li');
        prevLi.className = `page-item ${paginationCurrentPage === 1 ? 'disabled' : ''}`;
        prevLi.innerHTML = `<a class="page-link" href="#" onclick="changePage(${paginationCurrentPage - 1}); return false;">Previous</a>`;
        paginationControls.appendChild(prevLi);
        
        // Page numbers
        const maxButtons = 5;
        let startPage = Math.max(1, paginationCurrentPage - Math.floor(maxButtons / 2));
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
            li.className = `page-item ${i === paginationCurrentPage ? 'active' : ''}`;
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
        nextLi.className = `page-item ${paginationCurrentPage === totalPages || totalPages === 0 ? 'disabled' : ''}`;
        nextLi.innerHTML = `<a class="page-link" href="#" onclick="changePage(${paginationCurrentPage + 1}); return false;">Next</a>`;
        paginationControls.appendChild(nextLi);
    }
    
    // Change page - make globally accessible
    window.changePage = function(page) {
        // Track user interaction
        paginationLastUserInteraction = Date.now();
        window.lastUserInteraction = paginationLastUserInteraction;
        
        const totalPages = Math.ceil(paginationFilteredMemories.length / paginationItemsPerPage);
        
        if (page < 1 || page > totalPages) return;
        
        paginationCurrentPage = page;
        displayMemoriesWithPaging();
    }
    
    // Filter memories based on search
    function filterMemories(searchTerm = '') {
        const term = searchTerm.toLowerCase();
        
        // Use allMemories directly (it's already an array)
        const memoriesArray = Array.isArray(allMemories) ? allMemories : [];
        
        if (!term) {
            paginationFilteredMemories = [...memoriesArray];
        } else {
            paginationFilteredMemories = memoriesArray.filter(memory => {
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
        paginationCurrentPage = 1;
        displayMemoriesWithPaging();
    }
    
    // Override the displayMemories function
    window.displayMemories = function() {
        // Check if we're in raw view or merged view
        const isRawView = document.getElementById('rawView') && document.getElementById('rawView').checked;
        
        // Use allMemories directly (it's already an array)
        const memoriesArray = Array.isArray(allMemories) ? allMemories : [];
        
        // Initialize filtered memories
        paginationFilteredMemories = [...memoriesArray];
        
        // Preserve current sort if it exists, otherwise default to when desc
        if (!paginationCurrentSort || !paginationCurrentSort.field) {
            paginationCurrentSort = { field: 'when', direction: 'desc' };
        }
        
        // Apply current sort
        sortMemories();
        
        // Display with pagination (preserving current page if valid)
        const totalPages = Math.ceil(paginationFilteredMemories.length / paginationItemsPerPage);
        if (paginationCurrentPage > totalPages) {
            paginationCurrentPage = 1;
        }
        displayMemoriesWithPaging();
    };
    
    // Add search functionality
    document.addEventListener('DOMContentLoaded', function() {
        const searchBox = document.getElementById('memorySearch');
        if (searchBox) {
            searchBox.addEventListener('input', function(e) {
                // Track user interaction
                paginationLastUserInteraction = Date.now();
                window.lastUserInteraction = paginationLastUserInteraction;
                filterMemories(e.target.value);
            });
        }
    });
    
})(); // End of IIFE