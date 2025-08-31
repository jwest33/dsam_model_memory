// Enhanced JavaScript for [DSAM] Dual-Space Agentic Memory

let allMemories = [];
let currentPage = 1;  // Keep for compatibility, but will use paginationCurrentPage
let itemsPerPage = 10;  // Keep for compatibility, but will use paginationItemsPerPage
let graphNetwork = null;

// Enhanced pagination variables from memory-pagination.js
let paginationCurrentPage = 1;
let paginationItemsPerPage = 20;
let paginationCurrentSort = { field: 'when', direction: 'desc' };
let paginationFilteredMemories = [];
let paginationLastUserInteraction = 0;
let lastUserInteraction = 0;
let residualChart = null;
let spaceUsageChart = null;
let clusteringEnabled = false;  // Disabled since we removed the UI controls
let currentCenterNode = null;  // Track if we're viewing an individual memory
let relationStrengthThreshold = 0.3;  // Default threshold for edge filtering
let currentGravity = -8000;  // Default gravity value (negative for repulsion)
let allGraphData = null;  // Store complete graph data for filtering
let sortField = null;  // Current sort field
let sortDirection = null;  // 'asc', 'desc', or null
let originalMemoryOrder = [];  // Store original order of memories
let currentMergeDimension = null;  // Current merge dimension being viewed (null = not initialized)
let mergeGroups = {};  // Cache of merge groups by dimension

// Pagination state for merge groups
let mergeGroupsCurrentPage = 1;
let mergeGroupsItemsPerPage = 20;
let mergeGroupsFilteredData = [];
let mergeGroupsSortField = 'when';
let mergeGroupsSortDirection = 'desc';

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOMContentLoaded fired');
    
    // Clean up any leftover modal backdrops from previous sessions
    document.querySelectorAll('.modal-backdrop').forEach(backdrop => backdrop.remove());
    
    try {
        console.log('Initializing app...');
        initializeApp();
    } catch (error) {
        console.error('Error in initializeApp:', error);
    }
    
    try {
        console.log('Setting up event listeners...');
        setupEventListeners();
    } catch (error) {
        console.error('Error in setupEventListeners:', error);
    }
    
    try {
        console.log('Setting up column resizing...');
        setupColumnResizing();
    } catch (error) {
        console.error('Error in setupColumnResizing:', error);
    }
    
    try {
        console.log('Loading memories...');
        loadMemories();
    } catch (error) {
        console.error('Error in loadMemories:', error);
    }
    
    try {
        console.log('Loading stats...');
        loadStats();
        loadMergeStats();
    } catch (error) {
        console.error('Error in loadStats:', error);
    }
    
    try {
        console.log('Initializing charts...');
        initializeCharts();
    } catch (error) {
        console.error('Error in initializeCharts:', error);
    }
    
    // Auto-refresh stats every 30 seconds
    setInterval(loadStats, 30000);
});

function initializeApp() {
    // Set up view mode toggle listeners
    const mergedView = document.getElementById('mergedView');
    const rawView = document.getElementById('rawView');
    const dimensionSelector = document.querySelector('.merge-dimension-selector');
    
    if (mergedView) {
        mergedView.addEventListener('change', () => {
            console.log('Switched to merged view');
            // Update nav-link active classes
            document.querySelector('label[for="mergedView"]').classList.add('active');
            document.querySelector('label[for="rawView"]').classList.remove('active');
            // Show dimension selector in merged view
            if (dimensionSelector) {
                dimensionSelector.style.display = 'block';
            }
            // If no dimension selected, default to temporal
            if (!currentMergeDimension) {
                currentMergeDimension = 'temporal';
            }
            loadMemories();
            // Reposition tabs after state change
            setTimeout(() => {
                if (typeof positionMemoryViewTabs === 'function') {
                    positionMemoryViewTabs();
                }
            }, 10);
        });
    }
    if (rawView) {
        rawView.addEventListener('change', () => {
            console.log('Switched to raw view');
            // Update nav-link active classes
            document.querySelector('label[for="rawView"]').classList.add('active');
            document.querySelector('label[for="mergedView"]').classList.remove('active');
            // Hide dimension selector in raw view
            if (dimensionSelector) {
                dimensionSelector.style.display = 'none';
            }
            // Reset pagination for raw view
            paginationCurrentPage = 1;
            loadMemories();
            // Reposition tabs after state change
            setTimeout(() => {
                if (typeof positionMemoryViewTabs === 'function') {
                    positionMemoryViewTabs();
                }
            }, 10);
        });
    }
    
    // Initialize with merged view showing temporal dimension
    const memoryList = document.getElementById('memoryList');
    const memoryTableContainer = document.getElementById('memoryTableContainer');
    
    if (mergedView && mergedView.checked) {
        if (dimensionSelector) {
            dimensionSelector.style.display = 'block';
        }
        if (memoryList) memoryList.style.display = 'block';
        if (memoryTableContainer) memoryTableContainer.style.display = 'none';
        currentMergeDimension = 'temporal';
    } else if (rawView && rawView.checked) {
        if (dimensionSelector) {
            dimensionSelector.style.display = 'none';
        }
        if (memoryList) memoryList.style.display = 'none';
        if (memoryTableContainer) memoryTableContainer.style.display = 'block';
    }
    
    // Initialize gravity slider - now inverted so higher values = more spread out
    const gravitySlider = document.getElementById('gravitySlider');
    if (gravitySlider) {
        gravitySlider.addEventListener('input', function(e) {
            const sliderValue = parseInt(e.target.value);
            // Invert the value: slider goes 1000-50000, gravity needs -50000 to -1000
            // When slider is at 1000 (left/compact), gravity should be -1000 (less repulsion)
            // When slider is at 50000 (right/spread), gravity should be -50000 (more repulsion)
            const gravityValue = -sliderValue;
            currentGravity = gravityValue;
            updateGraphPhysics(gravityValue);
        });
    }
    
    // Memory form field listeners for space prediction
    const memoryFields = ['memWho', 'memWhat', 'memWhen', 'memWhere', 'memWhy', 'memHow'];
    memoryFields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        if (field) {
            field.addEventListener('input', updateExpectedSpaceUsage);
        }
    });
}

function setupEventListeners() {
    // Chat form
    const chatForm = document.getElementById('chatForm');
    if (chatForm) {
        chatForm.addEventListener('submit', handleChatSubmit);
    }
    
    // Search functionality with enhanced filtering
    const searchInput = document.getElementById('memorySearch');
    if (searchInput) {
        // Real-time search on input
        searchInput.addEventListener('input', function(e) {
            // Track user interaction
            paginationLastUserInteraction = Date.now();
            window.lastUserInteraction = paginationLastUserInteraction;
            filterMemories(e.target.value);
        });
        
        // Keep Enter key functionality for server search if needed
        searchInput.addEventListener('keyup', (e) => {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    }
    
    // Tab change listeners
    const analyticsTab = document.getElementById('analytics-tab');
    if (analyticsTab) {
        analyticsTab.addEventListener('shown.bs.tab', () => {
            updateCharts();
        });
    }
    
    // Show/hide Merged/Raw tabs based on active main tab
    const memoryTab = document.getElementById('memory-tab');
    const chatTab = document.getElementById('chat-tab');
    const memoryViewTabs = document.getElementById('memoryViewTabs');
    
    // Function to position the sub-tabs correctly under Memory Store
    function positionMemoryViewTabs() {
        if (!memoryTab || !memoryViewTabs) return;
        
        const memoryTabRect = memoryTab.getBoundingClientRect();
        const parentRect = memoryTab.parentElement.parentElement.getBoundingClientRect();
        const tabsContainer = memoryViewTabs.querySelector('.memory-view-tabs-container');
        
        if (tabsContainer) {
            // Calculate the exact position without any offset
            const leftOffset = memoryTabRect.left - parentRect.left;
            const width = memoryTabRect.width;
            
            // Apply styles directly without additional margins
            tabsContainer.style.marginLeft = `${leftOffset}px`;
            tabsContainer.style.width = `${width}px`;
            tabsContainer.style.marginRight = '0';
            tabsContainer.style.paddingLeft = '0';
            tabsContainer.style.paddingRight = '0';
        }
    }
    
    if (memoryTab && memoryViewTabs) {
        memoryTab.addEventListener('shown.bs.tab', () => {
            memoryViewTabs.style.display = 'block';
            positionMemoryViewTabs();
        });
    }
    
    if (chatTab && memoryViewTabs) {
        chatTab.addEventListener('shown.bs.tab', () => {
            memoryViewTabs.style.display = 'none';
        });
    }
    
    if (analyticsTab && memoryViewTabs) {
        analyticsTab.addEventListener('shown.bs.tab', () => {
            memoryViewTabs.style.display = 'none';
        });
    }
    
    // Reposition on window resize
    window.addEventListener('resize', () => {
        if (memoryViewTabs && memoryViewTabs.style.display === 'block') {
            positionMemoryViewTabs();
        }
    });
}

// Chat functionality with query type detection
async function handleChatSubmit(e) {
    e.preventDefault();
    
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Detect query type
    const queryType = detectQueryType(message);
    updateQueryTypeIndicator(queryType);
    
    // Add user message to chat
    addChatMessage(message, 'user');
    
    // Clear input
    input.value = '';
    input.disabled = true;
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Add assistant response with space usage indicator and memory details
        addChatMessage(data.response, 'assistant', {
            memories_used: data.memories_used,
            memory_details: data.memory_details,
            space_weights: data.space_weights,
            dimension_weights: data.dimension_weights,
            dominant_dimension: data.dominant_dimension
        });
        
        // Update dimension weights display if present
        if (data.dimension_weights) {
            updateDimensionWeights(data.dimension_weights, data.dominant_dimension);
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator(typingId);
        addChatMessage('Sorry, I encountered an error processing your message.', 'assistant');
    } finally {
        input.disabled = false;
        input.focus();
    }
}

function detectQueryType(message) {
    const concreteWords = ['code', 'error', 'bug', 'file', 'line', 'syntax', 'specific', 'example'];
    const abstractWords = ['why', 'concept', 'theory', 'philosophy', 'principle', 'pattern', 'architecture'];
    
    const lowerMessage = message.toLowerCase();
    let concreteScore = 0;
    let abstractScore = 0;
    
    concreteWords.forEach(word => {
        if (lowerMessage.includes(word)) concreteScore++;
    });
    
    abstractWords.forEach(word => {
        if (lowerMessage.includes(word)) abstractScore++;
    });
    
    if (concreteScore > abstractScore) return 'concrete';
    if (abstractScore > concreteScore) return 'abstract';
    return 'balanced';
}

function updateQueryTypeIndicator(queryType) {
    const indicator = document.getElementById('queryTypeIndicator');
    if (indicator) {
        indicator.textContent = queryType.charAt(0).toUpperCase() + queryType.slice(1);
        indicator.className = 'badge ms-2 ';
        
        switch(queryType) {
            case 'concrete':
                indicator.className += 'bg-info';
                break;
            case 'abstract':
                indicator.className += 'bg-warning';
                break;
            default:
                indicator.className += 'bg-secondary';
        }
    }
}

function updateDimensionWeights(dimensionWeights, dominantDimension) {
    const container = document.getElementById('dimensionWeightsContainer');
    if (!container || !dimensionWeights) return;
    
    // Show the container
    container.style.display = 'block';
    
    // Update each dimension weight
    const dimensions = ['actor', 'temporal', 'conceptual', 'spatial'];
    dimensions.forEach(dim => {
        const weight = dimensionWeights[dim] || 0;
        const percentage = Math.round(weight * 100);
        
        const element = document.getElementById(`${dim}Weight`);
        if (element) {
            // Update the fill bar
            const fillBar = element.querySelector('.dimension-fill');
            if (fillBar) {
                fillBar.style.width = `${percentage}%`;
            }
            
            // Update the value text
            const valueText = element.querySelector('.dimension-value');
            if (valueText) {
                valueText.textContent = `${percentage}%`;
            }
            
            // Highlight dominant dimension
            if (dominantDimension === dim) {
                element.classList.add('dominant-dimension');
            } else {
                element.classList.remove('dominant-dimension');
            }
        }
    });
}

function addChatMessage(message, sender, metadata = {}) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message mb-3`;
    
    // Generate unique ID for this message
    const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    messageDiv.id = messageId;
    
    let spaceIndicator = '';
    if (metadata.space_weights) {
        const euclideanPct = Math.round(metadata.space_weights.euclidean * 100);
        const hyperbolicPct = Math.round(metadata.space_weights.hyperbolic * 100);
        spaceIndicator = `
            <div class="space-indicator mt-2">
                <div class="progress" style="height: 10px;">
                    <div class="progress-bar bg-info" style="width: ${euclideanPct}%"></div>
                    <div class="progress-bar bg-warning" style="width: ${hyperbolicPct}%"></div>
                </div>
                <small class="text-muted">E: ${euclideanPct}% | H: ${hyperbolicPct}%</small>
            </div>
        `;
    }
    
    // Add click indicator for assistant messages with memories
    const clickIndicator = (sender === 'assistant' && metadata.memories_used > 0) 
        ? `<i class="bi bi-info-circle ms-2" style="cursor: pointer;" title="Click message to see memories used"></i>` 
        : '';
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <strong>${sender === 'user' ? 'You' : 'Assistant'}</strong>
            <span class="text-muted ms-2">${new Date().toLocaleTimeString()}</span>
            ${metadata.memories_used ? `<span class="badge bg-dark ms-2">${metadata.memories_used} memories</span>` : ''}
            ${clickIndicator}
        </div>
        <div class="message-content" ${sender === 'assistant' && metadata.memories_used > 0 ? 'style="cursor: pointer;"' : ''}>${escapeHtml(message)}</div>
        ${spaceIndicator}
        <div class="memory-details-container" style="display: none;"></div>
    `;
    
    // Add click handler for assistant messages
    if (sender === 'assistant' && metadata.memory_details && metadata.memory_details.length > 0) {
        messageDiv.dataset.memoryDetails = JSON.stringify(metadata.memory_details);
        
        const contentDiv = messageDiv.querySelector('.message-content');
        contentDiv.addEventListener('click', function() {
            toggleMemoryDetails(messageId);
        });
        
        // Also make the info icon clickable
        const infoIcon = messageDiv.querySelector('.bi-info-circle');
        if (infoIcon) {
            infoIcon.addEventListener('click', function() {
                toggleMemoryDetails(messageId);
            });
        }
    }
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Function to toggle memory details display
function toggleMemoryDetails(messageId) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;
    
    const detailsContainer = messageDiv.querySelector('.memory-details-container');
    const memoryDetails = JSON.parse(messageDiv.dataset.memoryDetails || '[]');
    
    if (detailsContainer.style.display === 'none') {
        // Show memory details
        let detailsHtml = `
            <div class="mt-3 p-3 bg-dark rounded">
                <h6 class="text-cyan mb-3">
                    <i class="bi bi-database"></i> Retrieved Memories Used for This Response:
                </h6>
                <div class="memories-list">
        `;
        
        memoryDetails.forEach((mem, index) => {
            detailsHtml += `
                <div class="memory-item mb-3 p-2 border border-secondary rounded">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <div class="row small">
                                <div class="col-md-6">
                                    <strong class="text-info">Who:</strong> ${escapeHtml(mem.who || '—')}
                                </div>
                                <div class="col-md-6">
                                    <strong class="text-info">When:</strong> ${escapeHtml(mem.when || '—')}
                                </div>
                            </div>
                            <div class="mt-1">
                                <strong class="text-info">What:</strong> ${escapeHtml(mem.what || '—')}
                            </div>
                            <div class="row small mt-1">
                                <div class="col-md-4">
                                    <strong class="text-muted">Where:</strong> ${escapeHtml(mem.where || '—')}
                                </div>
                                <div class="col-md-4">
                                    <strong class="text-muted">Why:</strong> ${escapeHtml(mem.why || '—')}
                                </div>
                                <div class="col-md-4">
                                    <strong class="text-muted">How:</strong> ${escapeHtml(mem.how || '—')}
                                </div>
                            </div>
                        </div>
                        <div class="ms-3">
                            <span class="badge bg-success">Score: ${mem.score.toFixed(2)}</span>
                            <button class="btn btn-sm btn-outline-info mt-2" onclick="window.showMemoryInGraph('${mem.id}')">
                                <i class="bi bi-diagram-3"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
        });
        
        detailsHtml += `
                </div>
            </div>
        `;
        
        detailsContainer.innerHTML = detailsHtml;
        detailsContainer.style.display = 'block';
        
        // Update cursor style
        const contentDiv = messageDiv.querySelector('.message-content');
        if (contentDiv) contentDiv.style.backgroundColor = 'rgba(0, 188, 212, 0.1)';
    } else {
        // Hide memory details
        detailsContainer.style.display = 'none';
        
        // Reset cursor style
        const contentDiv = messageDiv.querySelector('.message-content');
        if (contentDiv) contentDiv.style.backgroundColor = '';
    }
}

// Memory loading and display
async function loadMemoriesOriginal() {
    console.log('Loading memories...');
    
    // Get current view mode
    const viewMode = document.getElementById('rawView').checked ? 'raw' : 'merged';
    console.log('View mode:', viewMode);
    
    // Show loading indicator in table
    const tbody = document.getElementById('memoryTableBody');
    if (tbody) {
        tbody.innerHTML = `
            <tr>
                <td colspan="9" class="text-center py-4">
                    <div class="spinner-border text-cyan" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <div class="mt-2 text-muted">Loading memories...</div>
                </td>
            </tr>
        `;
    }
    
    try {
        // Fetch memories with view mode
        const response = await fetch(`/api/memories?view=${viewMode}`);
        console.log('Response status:', response.status);
        const data = await response.json();
        console.log('Data received:', data);
        
        // Store merge groups if in raw view
        if (viewMode === 'raw' && data.merge_groups) {
            window.mergeGroups = data.merge_groups;
        }
        
        // For memory-pagination.js compatibility, store the correct format
        // The pagination expects a raw array with direct properties, not nested five_w1h
        if (viewMode === 'raw') {
            // For raw view, flatten the five_w1h properties for display
            allMemories = (data.memories || []).map(m => ({
                ...m,
                who: m.five_w1h?.who || m.who || '',
                what: m.five_w1h?.what || m.what || '',
                when: m.five_w1h?.when || m.when || m.timestamp || '',
                where: m.five_w1h?.where || m.where || '',
                why: m.five_w1h?.why || m.why || '',
                how: m.five_w1h?.how || m.how || '',
                // Preserve space weights
                euclidean_weight: m.euclidean_weight,
                hyperbolic_weight: m.hyperbolic_weight
            }));
        } else {
            // For merged view, memories already have direct properties
            allMemories = (data.memories || []).map(m => ({
                ...m,
                euclidean_weight: m.euclidean_weight,
                hyperbolic_weight: m.hyperbolic_weight
            }));
        }
        // Store original order for removing sort
        originalMemoryOrder = [...allMemories];
        console.log('Total memories loaded:', allMemories.length);
        
        // Update counts in UI
        if (data.total_raw !== undefined) {
            document.getElementById('rawCount').textContent = data.total_raw;
        }
        if (data.total_merged !== undefined) {
            const mergedCountEl = document.getElementById('mergedCount');
            if (mergedCountEl) {
                mergedCountEl.textContent = data.total_merged || data.total_events || allMemories.length;
            }
        }
        
        // Log first memory structure to debug
        if (allMemories.length > 0) {
            console.log('First memory structure:', allMemories[0]);
        }
        
        // Update counters
        const totalMemoriesEl = document.getElementById('totalMemories');
        if (totalMemoriesEl) {
            totalMemoriesEl.textContent = allMemories.length;
        }
        
        // Count adapted memories (those with residuals)
        const adaptedCount = allMemories.filter(m => m.has_residual).length;
        const adaptedMemoriesEl = document.getElementById('adaptedMemories');
        if (adaptedMemoriesEl) {
            adaptedMemoriesEl.textContent = adaptedCount;
        }
        
        // Display memories
        console.log('About to call displayMemories, allMemories:', allMemories.length);
        displayMemories();
        console.log('displayMemories completed');
        
    } catch (error) {
        console.error('Error loading memories:', error);
        
        // Show error in table
        const tbody = document.getElementById('memoryTableBody');
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="9" class="text-center py-4 text-danger">
                        <i class="bi bi-exclamation-triangle fs-3"></i>
                        <div class="mt-2">Failed to load memories</div>
                        <small class="text-muted">${error.message || 'Unknown error'}</small>
                    </td>
                </tr>
            `;
        }
    }
}

// Enhanced memory display with sorting and pagination from memory-pagination.js
function createMemoryRow(memory, includeAll = false) {
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
        
        // Call viewMemoryDetails directly
        if (window.viewMemoryDetails) {
            window.viewMemoryDetails(memory.id);
        }
    });
    
    // Get residual indicator  
    const residualIndicator = getResidualIndicator(memory);
    
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
    const memorySpaceWeight = calculateMemorySpaceWeight(memory);
    
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
    
    // Add merge indicator for raw view
    let mergeIndicator = '';
    if (memory.type === 'raw' && memory.merged_id && window.mergeGroups) {
        const mergeGroup = window.mergeGroups[memory.merged_id];
        const groupSize = mergeGroup ? mergeGroup.length : 1;
        mergeIndicator = `<span class="badge bg-info ms-1" title="Part of merged group ${memory.merged_id.substring(0, 8)}">
            ${groupSize}
        </span>`;
    }
    
    row.innerHTML = `
        <td>${escapeHtml(who)}</td>
        <td title="${escapeAttr(what)}">${escapeHtml(whatDisplay)}${mergeIndicator}</td>
        <td>${formatDate(when)}</td>
        <td>${escapeHtml(whereDisplay)}</td>
        <td title="${escapeAttr(why)}">${escapeHtml(whyDisplay)}</td>
        <td title="${escapeAttr(how)}">${escapeHtml(howDisplay)}</td>
        <td>${memorySpaceWeight}</td>
        <td>${residualIndicator}</td>
        <td>
            <button class="btn btn-sm btn-outline-info" onclick="window.viewMemoryDetails('${memory.id}')" title="View Details">
                <i class="bi bi-eye"></i>
            </button>
            ${memory.type !== 'raw' ? 
                `<button class="btn btn-sm btn-outline-primary" onclick="window.showMemoryInGraph('${memory.id}')" title="Show in Graph">
                    <i class="bi bi-diagram-3"></i>
                </button>` : ''
            }
            ${memory.type === 'raw' ? 
                `<button class="btn btn-sm btn-outline-warning" onclick="window.showMergeGroup('${memory.id}')" title="Show Multi-Dimensional Merge Groups">
                    <i class="bi bi-diagram-3-fill"></i>
                </button>` : 
                `<button class="btn btn-sm btn-outline-danger" onclick="window.deleteMemory('${memory.id}')" title="Delete">
                    <i class="bi bi-trash"></i>
                </button>`
            }
        </td>
    `;
    
    return row;
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
    
    // If only 1 page or no pages, hide pagination
    if (totalPages <= 1) {
        paginationControls.style.display = 'none';
        return;
    }
    
    // Show pagination controls
    paginationControls.style.display = 'flex';
    
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
function displayMemories() {
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
}

// Make createMemoryRow globally accessible
window.createMemoryRow = createMemoryRow;

function calculateMemorySpaceWeight(memory) {
    // Use real space weights from the backend if available
    if (memory.euclidean_weight !== undefined && memory.hyperbolic_weight !== undefined) {
        const euclideanPct = Math.round(memory.euclidean_weight * 100);
        const hyperbolicPct = Math.round(memory.hyperbolic_weight * 100);
        
        // Create a mini progress bar with real weights
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
    
    // Fallback to field-based calculation if weights not available
    let concreteScore = 0;
    let abstractScore = 0;
    
    // Check both direct properties and five_w1h
    const who = memory.five_w1h?.who || memory.who;
    const what = memory.five_w1h?.what || memory.what;
    const when = memory.five_w1h?.when || memory.when;
    const where = memory.five_w1h?.where || memory.where;
    const why = memory.five_w1h?.why || memory.why;
    const how = memory.five_w1h?.how || memory.how;
    
    // Score concrete fields
    if (who) concreteScore += 1.0;
    if (what) concreteScore += 2.0;
    if (when) concreteScore += 0.5;
    if (where) concreteScore += 0.5;
    
    // Score abstract fields
    if (why) abstractScore += 1.5;
    if (how) abstractScore += 1.0;
    
    const total = concreteScore + abstractScore;
    if (total === 0) {
        return `<span class="badge bg-secondary">No data</span>`;
    }
    
    const euclideanPct = Math.round((concreteScore / total) * 100);
    const hyperbolicPct = Math.round((abstractScore / total) * 100);
    
    // Create a mini progress bar
    return `
        <div class="d-flex align-items-center">
            <div class="progress flex-grow-1" style="height: 15px; min-width: 80px;">
                <div class="progress-bar bg-info" style="width: ${euclideanPct}%" 
                     title="Euclidean: ${euclideanPct}%"></div>
                <div class="progress-bar bg-warning" style="width: ${hyperbolicPct}%" 
                     title="Hyperbolic: ${hyperbolicPct}%"></div>
            </div>
            <small class="ms-2 text-muted" style="font-size: 0.7rem;">
                ${euclideanPct}/${hyperbolicPct}
            </small>
        </div>
    `;
}

function getSpaceIndicator(memory) {
    // Determine which space is dominant based on content
    const concreteFields = (memory.who ? 1 : 0) + (memory.what ? 1 : 0) + (memory.where ? 1 : 0);
    const abstractFields = (memory.why ? 1 : 0) + (memory.how ? 1 : 0);
    
    if (concreteFields > abstractFields) {
        return '<span class="badge bg-info">Euclidean</span>';
    } else if (abstractFields > concreteFields) {
        return '<span class="badge bg-warning">Hyperbolic</span>';
    } else {
        return '<span class="badge bg-secondary">Balanced</span>';
    }
}

function getResidualIndicator(memory) {
    if (!memory.residual_norm) {
        return '<span class="text-muted">—</span>';
    }
    
    const norm = parseFloat(memory.residual_norm);
    let color = 'success';
    if (norm > 0.2) color = 'warning';
    if (norm > 0.3) color = 'danger';
    
    return `
        <div class="progress" style="height: 10px; width: 100px;">
            <div class="progress-bar bg-${color}" style="width: ${Math.min(100, norm * 200)}%"></div>
        </div>
        <small class="text-muted">${norm.toFixed(3)}</small>
    `;
}

// Enhanced memory graph visualization
window.showMemoryGraph = function() {
    // Show loading indicator immediately
    const graphContainer = document.getElementById('memoryGraph');
    if (graphContainer) {
        graphContainer.innerHTML = `
            <div class="d-flex flex-column justify-content-center align-items-center h-100">
                <div class="spinner-grow text-purple" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="mt-3 text-muted">Preparing visualization...</div>
            </div>
        `;
    }
    
    const modalElement = document.getElementById('memoryGraphModal');
    const modal = new bootstrap.Modal(modalElement);
    
    // Remove any existing event listeners first
    const newModalElement = modalElement.cloneNode(true);
    modalElement.parentNode.replaceChild(newModalElement, modalElement);
    
    // Add event listener to resize graph when modal is fully shown (once)
    newModalElement.addEventListener('shown.bs.modal', function () {
        // Initialize graph after modal is fully shown
        initializeGraph().then(() => {
            // Ensure the network fits properly in the container
            if (graphNetwork) {
                graphNetwork.redraw();
                graphNetwork.fit();
            }
        }).catch(error => {
            console.error('Failed to initialize graph:', error);
            const graphContainer = document.getElementById('memoryGraph');
            if (graphContainer) {
                graphContainer.innerHTML = `
                    <div class="d-flex flex-column justify-content-center align-items-center h-100">
                        <i class="bi bi-exclamation-triangle text-danger" style="font-size: 3rem;"></i>
                        <div class="mt-3 text-danger">Failed to load graph</div>
                        <small class="text-muted mt-2">${error.message || 'Please check console for details'}</small>
                    </div>
                `;
            }
        });
    }, { once: true });  // Use once option to auto-remove after firing
    
    // Create modal with the new element
    const newModal = new bootstrap.Modal(newModalElement);
    newModal.show();
}

async function initializeGraph(centerNodeId = null) {
    // Show loading indicator in graph container
    const graphContainer = document.getElementById('memoryGraph');
    if (graphContainer) {
        graphContainer.innerHTML = `
            <div class="d-flex flex-column justify-content-center align-items-center h-100">
                <div class="spinner-border text-purple" style="width: 3rem; height: 3rem;" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="mt-3 text-cyan">Building memory graph...</div>
                <small class="text-muted mt-2">Analyzing ${allGraphData ? allGraphData.nodes.length : '...'} memories</small>
            </div>
        `;
    }
    
    try {
        // Update global center node tracking
        currentCenterNode = centerNodeId;
        
        // Get selected components
        const components = getSelectedComponents();
        
        const requestBody = {
            components: components,
            use_clustering: false,  // Clustering disabled since we removed the UI
            visualization_mode: 'dual',  // Default to dual-space
            min_cluster_size: 5,
            min_samples: 3
        };
        
        // Add center node if specified
        if (centerNodeId) {
            requestBody.center_node = centerNodeId;
            requestBody.similarity_threshold = 0.2;  // Lower threshold to show more related memories
        }
        
        console.log('Sending graph request:', requestBody);
        console.log('Request URL: /api/graph');
        console.log('Request method: POST');
        
        const response = await fetch('/api/graph', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
            },
            cache: 'no-store',
            body: JSON.stringify(requestBody)
        });
        
        console.log('Response status:', response.status);
        console.log('Response OK:', response.ok);
        console.log('Response URL:', response.url);
        
        if (!response.ok) {
            // Try to get error message from response
            let errorMessage = `HTTP error! status: ${response.status}`;
            try {
                const errorData = await response.json();
                if (errorData.error) {
                    errorMessage += ` - ${errorData.error}`;
                }
            } catch (e) {
                // Response might not be JSON
            }
            throw new Error(errorMessage);
        }
        
        const data = await response.json();
        
        // Log the data for debugging
        console.log('Graph API response:', data);
        
        // Check if data has the expected structure
        if (!data || typeof data !== 'object') {
            console.error('Invalid response from graph API:', data);
            throw new Error('Invalid response from graph API');
        }
        
        if (!data.nodes || !Array.isArray(data.nodes)) {
            console.error('Invalid graph data structure - nodes:', data.nodes);
            throw new Error('Invalid graph data: missing or invalid nodes array');
        }
        
        if (!data.edges || !Array.isArray(data.edges)) {
            console.error('Invalid graph data structure - edges:', data.edges);
            throw new Error('Invalid graph data: missing or invalid edges array');
        }
        
        // Create graph visualization and wait for stabilization
        await createGraphVisualization(data);
        
        // Update statistics
        updateGraphStats(data);
        
        // Update space weights with actual data from server
        console.log('Graph data received, space_weights:', data.space_weights);
        if (data.space_weights) {
            displaySpaceWeights(data.space_weights);
        } else {
            console.warn('No space_weights in response, calculating locally');
            // Fallback to calculating from data
            updateSpaceWeights(data);
        }
        
    } catch (error) {
        console.error('Error loading graph:', error);
        
        // Show error in graph container
        const graphContainer = document.getElementById('memoryGraph');
        if (graphContainer) {
            graphContainer.innerHTML = `
                <div class="d-flex flex-column justify-content-center align-items-center h-100">
                    <i class="bi bi-exclamation-triangle text-danger" style="font-size: 3rem;"></i>
                    <div class="mt-3 text-danger">Failed to load graph</div>
                    <small class="text-muted mt-2">${error.message || 'Unknown error'}</small>
                    <button class="btn btn-outline-cyan btn-sm mt-3" onclick="refreshGraph()">
                        <i class="bi bi-arrow-clockwise"></i> Try Again
                    </button>
                </div>
            `;
        }
    }
}

function getSelectedComponents() {
    // Always return all components since we removed the filter checkboxes
    return ['who', 'what', 'when', 'where', 'why', 'how'];
}

function createGraphVisualization(data) {
    const container = document.getElementById('memoryGraph');
    
    // Store complete graph data for filtering
    allGraphData = data;
    
    // Prepare nodes with color coding based on space/cluster
    const nodes = new vis.DataSet(data.nodes.map(node => {
        const nodeConfig = {
            id: node.id,
            label: node.label,
            color: getNodeColor(node),
            size: 10 + (node.centrality || 0) * 15,  // Reduced from 20 + 30
            title: createNodeTooltip(node),
            // Store node data for later access but don't spread all properties
            who: node.who,
            what: node.what,
            when: node.when,
            where: node.where,
            why: node.why,
            how: node.how,
            space: node.space,
            cluster_id: node.cluster_id,
            centrality: node.centrality,
            residual_norm: node.residual_norm,
            is_center: node.is_center
        };
        
        // Make center node larger and with a border
        if (node.is_center) {
            nodeConfig.size = 25;
            nodeConfig.borderWidth = 4;
            nodeConfig.borderWidthSelected = 6;
            nodeConfig.color = {
                background: getNodeColor(node),
                border: '#ff0000',
                highlight: {
                    background: getNodeColor(node),
                    border: '#ff0000'
                }
            };
            nodeConfig.shape = 'star';  // Different shape for center
        }
        
        return nodeConfig;
    }));
    
    // Check if we have a center node
    const centerNode = data.nodes.find(n => n.is_center);
    const centerNodeId = centerNode ? centerNode.id : null;
    
    // Prepare edges with gradient color based on similarity strength
    // Create all edges, but mark some as hidden based on threshold
    const edges = new vis.DataSet(data.edges.map((edge, index) => {
        let edgeColor;
        let edgeTitle;
        let edgeWidth = 1;
        let zIndex = 0;  // Default z-index
        
        // Check if edge is connected to center node
        const isConnectedToCenter = centerNodeId && (edge.from === centerNodeId || edge.to === centerNodeId);
        
        if (isConnectedToCenter) {
            // Cyan color for edges connected to center node
            edgeColor = {
                color: '#00ffff',  // Bright cyan
                highlight: '#00ffff',
                hover: '#00ffff',
                opacity: 0.9
            };
            edgeTitle = `Connected to center - Similarity: ${edge.weight.toFixed(3)}`;
            edgeWidth = 3;  // Thicker edges for center connections
            zIndex = 1000;  // Higher z-index to appear above other edges
        } else if (edge.type === 'conversation') {
            // Bright purple for User-Assistant conversation flow
            edgeColor = {
                color: '#ff00ff',  // Bright purple for conversation edges
                opacity: 0.6  // Reduced opacity so center edges stand out more
            };
            edgeTitle = 'Conversation Flow';
            edgeWidth = 2;
        } else {
            // Regular edges - make them more subtle
            const weight = edge.weight || 0.5;
            
            // More muted colors for non-center edges
            const r = Math.floor(100 + (100 * weight));  // Muted red
            const g = Math.floor(100 + (50 * (1 - weight)));  // Muted green
            const b = Math.floor(150 + (105 * weight));  // Muted blue
            
            edgeColor = {
                color: `rgba(${r}, ${g}, ${b}, 0.3)`,  // Lower opacity for regular edges
                opacity: 0.3  // Very subtle for non-center edges
            };
            edgeTitle = `Similarity: ${edge.weight.toFixed(3)}`;
        }
        
        // Determine if edge should be hidden based on threshold
        // For center view, only show edges connected to center and strong connections
        const hidden = centerNodeId ? 
            !(isConnectedToCenter || (edge.weight >= 0.7)) :  // In center view, hide most non-center edges
            !(edge.type === 'conversation' || edge.weight >= relationStrengthThreshold);
        
        return {
            id: `edge-${index}`,  // Add unique ID for each edge
            from: edge.from,
            to: edge.to,
            value: edge.weight || 0.8,
            color: edgeColor,
            title: edgeTitle,
            width: edgeWidth,
            hidden: hidden,
            weight: edge.weight,  // Store original weight
            type: edge.type,  // Store type for filtering
            chosen: isConnectedToCenter ? {
                edge: function(values, id, selected, hovering) {
                    values.width = 4;  // Even thicker when selected
                    values.color = '#00ffff';
                }
            } : false,
            smooth: {
                enabled: true,
                type: isConnectedToCenter ? 'straightCross' : 'dynamic'  // Different curve for center edges
            },
            zIndex: zIndex  // Higher z-index for center edges
        };
    }));
    
    // Sort edges so that center-connected edges are rendered last (on top)
    const edgesArray = edges.get();
    edgesArray.sort((a, b) => {
        // Check if edges are connected to center
        const aIsCenter = centerNodeId && (a.from === centerNodeId || a.to === centerNodeId);
        const bIsCenter = centerNodeId && (b.from === centerNodeId || b.to === centerNodeId);
        
        if (aIsCenter && !bIsCenter) return 1;  // a comes after b
        if (!aIsCenter && bIsCenter) return -1; // b comes after a
        return 0;  // Keep original order
    });
    
    // Clear and re-add edges in sorted order
    edges.clear();
    edges.add(edgesArray);
    
    const graphData = { nodes, edges };
    
    // Options for the network visualization
    const options = {
        nodes: {
            shape: 'dot',
            font: {
                size: 10,  // Reduced from 12
                color: '#ffffff'
            },
            borderWidth: 2,
            shadow: true
        },
        edges: {
            smooth: {
                type: 'continuous',
                roundness: 0.5
            },
            width: 1,  // Default width
            shadow: true,
            arrows: {
                to: {
                    enabled: false  // Disable arrows for cleaner look
                }
            },
            selectionWidth: function (width) { 
                return width * 2; 
            },
            hoverWidth: function (width) { 
                return width * 1.5; 
            }
        },
        physics: {
            enabled: true,
            stabilization: {
                enabled: true,
                iterations: 500,  // Increased for better initial positioning
                updateInterval: 50,
                fit: true
            },
            barnesHut: {
                gravitationalConstant: currentGravity,  // Use dynamic gravity
                centralGravity: 0.3,  // Increased for more center pull
                springConstant: 0.04,  // Increased for stronger springs
                springLength: 95,  // Slightly shorter for tighter layout
                damping: 0.5,  // Much higher damping to stop oscillation
                avoidOverlap: 0.1  // Reduced to allow tighter packing
            },
            timestep: 0.5,  // Smaller timestep for stability
            adaptiveTimestep: true
        },
        interaction: {
            hover: true,
            tooltipDelay: 200
        }
    };
    
    // If center node exists, use hierarchical layout
    if (centerNode) {
        options.layout = {
            improvedLayout: true,
            hierarchical: {
                enabled: false  // Disable hierarchical, we'll position manually
            }
        };
    }
    
    // Create network
    graphNetwork = new vis.Network(container, graphData, options);
    
    // Store the nodes dataset globally for access
    graphNetwork.nodesDataset = nodes;
    
    // Add click handlers
    graphNetwork.on('click', function(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            const node = nodes.get(nodeId);
            displayNodeDetails(node);
        }
    });
    
    // Enable drag to reposition - nodes will stay where placed
    graphNetwork.on('dragEnd', function(params) {
        if (params.nodes.length > 0) {
            // Node was dragged - it will stay in its new position since physics is off
            console.log('Node repositioned:', params.nodes[0]);
        }
    });
    
    // Return a promise that resolves when the network is stabilized
    return new Promise((resolve) => {
        graphNetwork.once('stabilizationIterationsDone', function() {
            console.log('Network stabilized');
            // Stop physics after stabilization to prevent continuous movement
            setTimeout(() => {
                graphNetwork.setOptions({
                    physics: {
                        enabled: false  // Stop physics to freeze positions
                    }
                });
                console.log('Physics stopped - nodes frozen');
            }, 500);  // Small delay to ensure final positions are set
            resolve();
        });
        
        // Fallback in case stabilization doesn't trigger
        setTimeout(() => {
            graphNetwork.setOptions({
                physics: {
                    enabled: false
                }
            });
            resolve();
        }, 3000);  // Give it 3 seconds max
    });
}

function getNodeColor(node) {
    const vizMode = 'dual';  // Default to dual-space since we removed the UI control
    
    // First, check if it's User or Assistant for special coloring
    if (node.who === 'User') {
        return '#00ffff';  // Cyan for User
    }
    if (node.who === 'Assistant') {
        return '#ff00ff';  // Magenta/Purple for Assistant
    }
    
    if (vizMode === 'residuals') {
        // Color based on residual magnitude
        const norm = node.residual_norm || 0;
        if (norm < 0.1) return '#00ff00';
        if (norm < 0.2) return '#ffff00';
        return '#ff0000';
    }
    
    if (node.cluster_id !== undefined && node.cluster_id >= 0) {
        // Color by cluster
        const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9'];
        return colors[node.cluster_id % colors.length];
    }
    
    // Color by dominant space for other nodes
    if (node.space === 'euclidean') return '#00bcd4';
    if (node.space === 'hyperbolic') return '#ffc107';
    return '#6c757d';
}

function createNodeTooltip(node) {
    // Return plain text for vis.js tooltip (no HTML)
    return `${node.label}
Space: ${node.space || 'balanced'}
Cluster: ${node.cluster_id >= 0 ? node.cluster_id : 'none'}
Centrality: ${(node.centrality || 0).toFixed(3)}
Residual: ${(node.residual_norm || 0).toFixed(3)}`;
}

function displayNodeDetails(node) {
    const detailsDiv = document.getElementById('nodeDetailsContent');
    
    detailsDiv.innerHTML = `
        <div class="mb-2">
            <strong>ID:</strong> ${node.id.substring(0, 8)}...
        </div>
        <div class="mb-2">
            <strong>Who:</strong> ${escapeHtml(node.who || '—')}
        </div>
        <div class="mb-2">
            <strong>What:</strong> ${escapeHtml(node.what || '—')}
        </div>
        <div class="mb-2">
            <strong>When:</strong> ${escapeHtml(node.when || '—')}
        </div>
        <div class="mb-2">
            <strong>Where:</strong> ${escapeHtml(node.where || '—')}
        </div>
        <div class="mb-2">
            <strong>Why:</strong> ${escapeHtml(node.why || '—')}
        </div>
        <div class="mb-2">
            <strong>How:</strong> ${escapeHtml(node.how || '—')}
        </div>
        <hr>
        <div class="mb-2">
            <strong>Dominant Space:</strong> 
            <span class="badge bg-${node.space === 'euclidean' ? 'info' : 'warning'}">
                ${node.space || 'balanced'}
            </span>
        </div>
        <div class="mb-2">
            <strong>Cluster:</strong> ${node.cluster_id >= 0 ? `Cluster ${node.cluster_id}` : 'No cluster'}
        </div>
        <div class="mb-2">
            <strong>Centrality:</strong> ${(node.centrality || 0).toFixed(3)}
        </div>
        <div class="mb-2">
            <strong>Residual Norm:</strong> ${(node.residual_norm || 0).toFixed(4)}
        </div>
    `;
}

function updateGraphStats(data) {
    document.getElementById('nodeCount').textContent = data.nodes.length;
    document.getElementById('edgeCount').textContent = data.edges.length;
    document.getElementById('clusterCount').textContent = data.cluster_count || 0;
    
    const avgDegree = data.edges.length * 2 / Math.max(1, data.nodes.length);
    document.getElementById('avgDegree').textContent = avgDegree.toFixed(1);
}

function displaySpaceWeights(spaceWeights) {
    // Display space weights from server
    console.log('displaySpaceWeights called with:', spaceWeights);
    const euclideanPct = Math.round(spaceWeights.euclidean * 100);
    const hyperbolicPct = Math.round(spaceWeights.hyperbolic * 100);
    console.log('Calculated percentages - Euclidean:', euclideanPct, 'Hyperbolic:', hyperbolicPct);
    
    const euclideanBar = document.getElementById('euclideanWeight');
    const hyperbolicBar = document.getElementById('hyperbolicWeight');
    
    if (euclideanBar && hyperbolicBar) {
        euclideanBar.style.width = `${euclideanPct}%`;
        euclideanBar.textContent = `Euclidean: ${euclideanPct}%`;
        hyperbolicBar.style.width = `${hyperbolicPct}%`;
        hyperbolicBar.textContent = `Hyperbolic: ${hyperbolicPct}%`;
    }
    
    // Update the space indicator badge
    const spaceIndicator = document.getElementById('graphSpaceIndicator');
    if (spaceIndicator) {
        if (euclideanPct > 60) {
            spaceIndicator.textContent = 'Euclidean-Heavy';
            spaceIndicator.className = 'badge bg-info ms-2';
        } else if (hyperbolicPct > 60) {
            spaceIndicator.textContent = 'Hyperbolic-Heavy';
            spaceIndicator.className = 'badge bg-warning ms-2';
        } else {
            spaceIndicator.textContent = 'Balanced';
            spaceIndicator.className = 'badge bg-success ms-2';
        }
    }
}

function updateSpaceWeights(graphData = null) {
    const components = getSelectedComponents();
    
    // If we have actual graph data, calculate weights based on the nodes
    if (graphData && graphData.nodes) {
        let concreteScore = 0;
        let abstractScore = 0;
        
        // Calculate based on actual node content
        graphData.nodes.forEach(node => {
            // Count filled fields weighted by their nature
            if (node.who && components.includes('who')) concreteScore += 1.0;
            if (node.what && components.includes('what')) concreteScore += 2.0;
            if (node.when && components.includes('when')) concreteScore += 0.5;
            if (node.where && components.includes('where')) concreteScore += 0.5;
            if (node.why && components.includes('why')) abstractScore += 1.5;
            if (node.how && components.includes('how')) abstractScore += 1.0;
        });
        
        // Average over number of nodes
        const nodeCount = Math.max(1, graphData.nodes.length);
        concreteScore = concreteScore / nodeCount;
        abstractScore = abstractScore / nodeCount;
    } else {
        // Fallback to component-based calculation
        var concreteScore = 0;
        var abstractScore = 0;
        
        if (components.includes('who')) concreteScore += 1.0;
        if (components.includes('what')) concreteScore += 2.0;
        if (components.includes('when')) concreteScore += 0.5;
        if (components.includes('where')) concreteScore += 0.5;
        if (components.includes('why')) abstractScore += 1.5;
        if (components.includes('how')) abstractScore += 1.0;
    }
    
    const total = concreteScore + abstractScore;
    const euclideanPct = total > 0 ? Math.round((concreteScore / total) * 100) : 50;
    const hyperbolicPct = 100 - euclideanPct;
    
    // Update display
    const euclideanBar = document.getElementById('euclideanWeight');
    const hyperbolicBar = document.getElementById('hyperbolicWeight');
    
    if (euclideanBar && hyperbolicBar) {
        euclideanBar.style.width = `${euclideanPct}%`;
        euclideanBar.textContent = `Euclidean: ${euclideanPct}%`;
        hyperbolicBar.style.width = `${hyperbolicPct}%`;
        hyperbolicBar.textContent = `Hyperbolic: ${hyperbolicPct}%`;
    }
}

function refreshGraph() {
    if (graphNetwork) {
        // Preserve center node if viewing individual memory
        initializeGraph(currentCenterNode);
    }
}

// Analytics and statistics
async function loadMergeStats() {
    try {
        const response = await fetch('/api/merge-stats');
        const data = await response.json();
        
        // Update UI with merge statistics
        if (data.total_raw !== undefined) {
            const rawCountEl = document.getElementById('rawCount');
            if (rawCountEl) rawCountEl.textContent = data.total_raw;
        }
        if (data.total_merged !== undefined) {
            const mergedCountEl = document.getElementById('mergedCount');
            if (mergedCountEl) mergedCountEl.textContent = data.total_merged;
        }
        
        // Update multi-dimensional merge counts in tabs
        if (data.multi_dimensional_stats) {
            const stats = data.multi_dimensional_stats;
            if (stats.temporal) {
                const temporalCount = document.getElementById('temporal-count');
                if (temporalCount) temporalCount.textContent = stats.temporal.group_count || 0;
            }
            if (stats.actor) {
                const actorCount = document.getElementById('actor-count');
                if (actorCount) actorCount.textContent = stats.actor.group_count || 0;
            }
            if (stats.conceptual) {
                const conceptualCount = document.getElementById('conceptual-count');
                if (conceptualCount) conceptualCount.textContent = stats.conceptual.group_count || 0;
            }
            if (stats.spatial) {
                const spatialCount = document.getElementById('spatial-count');
                if (spatialCount) spatialCount.textContent = stats.spatial.group_count || 0;
            }
        }
        
        console.log('Merge statistics loaded:', data);
    } catch (error) {
        console.error('Error loading merge stats:', error);
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        // Update stat cards
        document.getElementById('statTotalEvents').textContent = stats.total_events || 0;
        document.getElementById('statTotalQueries').textContent = stats.total_queries || 0;
        
        if (stats.average_residual_norm) {
            document.getElementById('statAvgEuclidean').textContent = 
                stats.average_residual_norm.euclidean.toFixed(3);
            document.getElementById('statAvgHyperbolic').textContent = 
                stats.average_residual_norm.hyperbolic.toFixed(3);
        }
        
        // Update charts if visible
        if (document.querySelector('#analytics.active')) {
            updateCharts();
        }
        
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

function initializeCharts() {
    // Residual evolution chart
    const residualCtx = document.getElementById('residualChart');
    if (residualCtx) {
        residualChart = new Chart(residualCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Euclidean',
                    data: [],
                    borderColor: '#00bcd4',
                    backgroundColor: 'rgba(0, 188, 212, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Hyperbolic',
                    data: [],
                    borderColor: '#ae00c5ff',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2,  // Match the pie chart aspect ratio
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff',
                            padding: 10,
                            font: {
                                size: 11
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        // Remove fixed max to allow auto-scaling
                        ticks: {
                            color: '#ffffff',
                            // Add padding to prevent clipping at top
                            padding: 5
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        // Add grace to automatically add some space above max value
                        grace: '10%'
                    },
                    x: {
                        ticks: {
                            color: '#ffffff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });
    }
    
    // Space usage distribution chart - now shows collective ratio as percentages
    const spaceCtx = document.getElementById('spaceUsageChart');
    if (spaceCtx) {
        spaceUsageChart = new Chart(spaceCtx, {
            type: 'doughnut',
            data: {
                labels: ['Euclidean Space', 'Hyperbolic Space'],
                datasets: [{
                    data: [50, 50],
                    backgroundColor: ['#00bcd4', '#ffc107']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2,  // Makes the chart wider than tall (reduces height)
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff',
                            padding: 10,
                            font: {
                                size: 11
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
    }
}

async function updateCharts() {
    try {
        const response = await fetch('/api/analytics');
        const data = await response.json();
        
        // Update residual evolution chart
        if (residualChart && data.residual_history) {
            residualChart.data.labels = data.residual_history.labels;
            residualChart.data.datasets[0].data = data.residual_history.euclidean;
            residualChart.data.datasets[1].data = data.residual_history.hyperbolic;
            residualChart.update();
        }
        
        // Update space usage chart - now shows percentages
        if (spaceUsageChart && data.space_distribution) {
            spaceUsageChart.data.datasets[0].data = [
                data.space_distribution.euclidean,
                data.space_distribution.hyperbolic
            ];
            spaceUsageChart.update();
            
            // Update average lambda values if available
            if (data.space_distribution.avg_lambda_e !== undefined) {
                const lambdaEElement = document.getElementById('avgLambdaE');
                if (lambdaEElement) {
                    lambdaEElement.textContent = data.space_distribution.avg_lambda_e.toFixed(2);
                }
            }
            if (data.space_distribution.avg_lambda_h !== undefined) {
                const lambdaHElement = document.getElementById('avgLambdaH');
                if (lambdaHElement) {
                    lambdaHElement.textContent = data.space_distribution.avg_lambda_h.toFixed(2);
                }
            }
        }
        
    } catch (error) {
        console.error('Error updating charts:', error);
    }
}

// Memory management functions
function showCreateMemoryModal() {
    const modal = new bootstrap.Modal(document.getElementById('createMemoryModal'));
    modal.show();
    updateExpectedSpaceUsage();
}

window.createMemory = async function() {
    const formData = {
        who: document.getElementById('memWho').value,
        what: document.getElementById('memWhat').value,
        when: document.getElementById('memWhen').value,
        where: document.getElementById('memWhere').value,
        why: document.getElementById('memWhy').value,
        how: document.getElementById('memHow').value,
        type: 'observation'
    };
    
    try {
        const response = await fetch('/api/memories', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (response.ok) {
            // Close modal
            bootstrap.Modal.getInstance(document.getElementById('createMemoryModal')).hide();
            
            // Clear form
            document.getElementById('createMemoryForm').reset();
            
            // Reload memories
            loadMemories();
            
            // Show success message
            showNotification('Memory created successfully', 'success');
        } else {
            const error = await response.json();
            showNotification(error.message || 'Failed to create memory', 'danger');
        }
    } catch (error) {
        console.error('Error creating memory:', error);
        showNotification('Failed to create memory', 'danger');
    }
}

function updateExpectedSpaceUsage() {
    const who = document.getElementById('memWho').value;
    const what = document.getElementById('memWhat').value;
    const when = document.getElementById('memWhen').value;
    const where = document.getElementById('memWhere').value;
    const why = document.getElementById('memWhy').value;
    const how = document.getElementById('memHow').value;
    
    // Calculate expected space usage
    let concreteScore = 0;
    let abstractScore = 0;
    
    if (who) concreteScore += who.split(' ').length;
    if (what) concreteScore += what.split(' ').length * 2;
    if (when) concreteScore += when.split(' ').length * 0.5;
    if (where) concreteScore += where.split(' ').length * 0.5;
    if (why) abstractScore += why.split(' ').length * 1.5;
    if (how) abstractScore += how.split(' ').length;
    
    const total = concreteScore + abstractScore;
    const euclideanPct = total > 0 ? Math.round((concreteScore / total) * 100) : 50;
    const hyperbolicPct = 100 - euclideanPct;
    
    // Update display
    const euclideanBar = document.getElementById('expectedEuclidean');
    const hyperbolicBar = document.getElementById('expectedHyperbolic');
    
    if (euclideanBar && hyperbolicBar) {
        euclideanBar.style.width = `${euclideanPct}%`;
        euclideanBar.textContent = `Euclidean ${euclideanPct}%`;
        hyperbolicBar.style.width = `${hyperbolicPct}%`;
        hyperbolicBar.textContent = `Hyperbolic ${hyperbolicPct}%`;
    }
}

// Function to show multi-dimensional merge groups for a raw event
async function showRawEventMergeGroups(eventId) {
    try {
        const response = await fetch(`/api/raw-event/${eventId}/merge-groups`);
        const data = await response.json();
        
        const mergeTypes = ['actor', 'temporal', 'conceptual', 'spatial'];
        const mergeTypeLabels = {
            'actor': { label: 'Actor (WHO)', icon: 'person-fill', color: 'primary' },
            'temporal': { label: 'Temporal (WHEN)', icon: 'clock-history', color: 'success' },
            'conceptual': { label: 'Conceptual (WHY/HOW)', icon: 'lightbulb-fill', color: 'warning' },
            'spatial': { label: 'Spatial (WHERE)', icon: 'geo-alt-fill', color: 'info' }
        };
        
        // Create tabs for each merge dimension - use flex-nowrap to keep them on one row
        let tabsHtml = '<ul class="nav nav-tabs flex-nowrap" id="mergeGroupTabs" role="tablist" style="overflow-x: auto; white-space: nowrap;">';
        let contentHtml = '<div class="tab-content mt-3" id="mergeGroupTabContent">';
        
        mergeTypes.forEach((type, index) => {
            const isActive = index === 0;
            const groupData = data.multi_dimensional_groups[type];
            const hasData = groupData && groupData.merge_id;
            const tabClass = hasData ? '' : 'disabled';
            const badge = hasData ? `<span class="badge bg-secondary ms-1">${groupData.merge_count}</span>` : '';
            
            tabsHtml += `
                <li class="nav-item flex-shrink-0" role="presentation">
                    <button class="nav-link ${isActive ? 'active' : ''} ${tabClass}" 
                            id="${type}-tab" 
                            data-bs-toggle="tab" 
                            data-bs-target="#${type}-panel" 
                            type="button" 
                            role="tab"
                            style="white-space: nowrap;"
                            ${!hasData ? 'disabled' : ''}>
                        <i class="bi bi-${mergeTypeLabels[type].icon}"></i> 
                        <span class="d-none d-md-inline">${mergeTypeLabels[type].label}</span>
                        <span class="d-inline d-md-none">${type.charAt(0).toUpperCase() + type.slice(1)}</span>
                        ${badge}
                    </button>
                </li>
            `;
            
            if (hasData) {
                contentHtml += `
                    <div class="tab-pane fade ${isActive ? 'show active' : ''}" 
                         id="${type}-panel" 
                         role="tabpanel" 
                         aria-labelledby="${type}-tab">
                        ${renderMergeGroupContent(groupData, type)}
                    </div>
                `;
            } else {
                contentHtml += `
                    <div class="tab-pane fade ${isActive ? 'show active' : ''}" 
                         id="${type}-panel" 
                         role="tabpanel" 
                         aria-labelledby="${type}-tab">
                        <div class="text-center text-muted p-4">
                            <i class="bi bi-${mergeTypeLabels[type].icon} fs-1"></i>
                            <p class="mt-2">This event is not part of any ${type} merge group</p>
                        </div>
                    </div>
                `;
            }
        });
        
        tabsHtml += '</ul>';
        contentHtml += '</div>';
        
        // Create modal
        const modalHtml = `
            <div class="modal fade" id="rawEventMergeGroupsModal" tabindex="-1">
                <div class="modal-dialog modal-xl">
                    <div class="modal-content bg-dark border-cyan">
                        <div class="modal-header border-cyan">
                            <h5 class="modal-title text-cyan">
                                <i class="bi bi-diagram-3-fill text-warning"></i> Multi-Dimensional Merge Groups
                                <span class="badge bg-purple ms-2">${eventId}</span>
                                <span class="badge bg-info ms-2">${data.total_groups} groups</span>
                            </h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="mb-3">
                                <small class="text-muted">
                                    This raw event belongs to ${data.total_groups} different merge groups across multiple dimensions.
                                    Each dimension groups events based on different aspects of the 5W1H structure.
                                </small>
                            </div>
                            ${tabsHtml}
                            ${contentHtml}
                        </div>
                        <div class="modal-footer border-cyan">
                            <button type="button" class="btn btn-outline-cyan" data-bs-dismiss="modal">
                                <i class="bi bi-x-circle"></i> Close
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Remove existing modal if any
        const existingModal = document.getElementById('rawEventMergeGroupsModal');
        if (existingModal) {
            existingModal.remove();
        }
        
        // Add modal to body
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('rawEventMergeGroupsModal'));
        modal.show();
        
    } catch (error) {
        console.error('Error loading raw event merge groups:', error);
        alert('Failed to load merge group details');
    }
}

// Helper function to render merge group content
function renderMergeGroupContent(groupData, type) {
    let html = `
        <div class="p-3">
            <div class="row mb-3">
                <div class="col-md-6">
                    <h6 class="text-cyan">Merge Information</h6>
                    <div class="small">
                        <div><span class="text-warning">Merge ID:</span> <code class="text-white">${groupData.merge_id}</code></div>
                        <div><span class="text-warning">Total Events:</span> <span class="text-white">${groupData.merge_count}</span></div>
                        <div><span class="text-warning">Created:</span> <span class="text-white">${formatDate(groupData.created_at)}</span></div>
                        <div><span class="text-warning">Last Updated:</span> <span class="text-white">${formatDate(groupData.last_updated)}</span></div>
                    </div>
                </div>
                <div class="col-md-6">
                    <h6 class="text-cyan">Latest State</h6>
                    <div class="small bg-darker p-2 rounded">
                        ${groupData.latest_state ? `
                            <div><span class="text-info">Who:</span> <span class="text-white">${escapeHtml(groupData.latest_state.who || '—')}</span></div>
                            <div><span class="text-info">What:</span> <span class="text-white">${escapeHtml((groupData.latest_state.what || '—').substring(0, 100))}...</span></div>
                            <div><span class="text-info">Why:</span> <span class="text-white">${escapeHtml(groupData.latest_state.why || '—')}</span></div>
                        ` : '<em>No state available</em>'}
                    </div>
                </div>
            </div>
    `;
    
    // Show variants based on merge type
    if (type === 'actor' && groupData.who_variants && Object.keys(groupData.who_variants).length > 0) {
        html += renderVariants('Who Variants', groupData.who_variants);
    }
    
    if (type === 'temporal' && groupData.when_timeline && groupData.when_timeline.length > 0) {
        html += `
            <div class="mb-3">
                <h6 class="text-cyan">Timeline</h6>
                <div class="timeline small bg-darker p-2 rounded" style="max-height: 200px; overflow-y: auto;">
        `;
        groupData.when_timeline.forEach(point => {
            html += `
                <div class="mb-2">
                    <span class="badge bg-secondary me-2">${formatDate(point.timestamp)}</span>
                    ${escapeHtml(point.description || '')}
                </div>
            `;
        });
        html += '</div></div>';
    }
    
    if (type === 'conceptual') {
        if (groupData.why_variants && Object.keys(groupData.why_variants).length > 0) {
            html += renderVariants('Why Variants', groupData.why_variants);
        }
        if (groupData.how_variants && Object.keys(groupData.how_variants).length > 0) {
            html += renderVariants('How Variants', groupData.how_variants);
        }
    }
    
    if (type === 'spatial' && groupData.where_locations && Object.keys(groupData.where_locations).length > 0) {
        html += renderVariants('Where Variants', groupData.where_locations);
    }
    
    // Removed 'Other Events in This Group' section as requested
    
    html += '</div>';
    return html;
}

// Helper function to render variants
function renderVariants(title, variants) {
    let html = `
        <div class="mb-3">
            <h6 class="text-cyan">${title}</h6>
            <div class="small bg-darker p-2 rounded" style="max-height: 150px; overflow-y: auto;">
    `;
    
    for (const [variant, count] of Object.entries(variants)) {
        html += `
            <div class="mb-1">
                <span class="badge bg-secondary me-2">${count}x</span>
                <span class="text-white">${escapeHtml(variant)}</span>
            </div>
        `;
    }
    
    html += '</div></div>';
    return html;
}

window.showMergeGroup = async function(mergedId) {
    try {
        // Check if this is a raw event ID - if so, show multi-dimensional groups
        if (mergedId.startsWith('raw_')) {
            return showRawEventMergeGroups(mergedId);
        }
        
        const response = await fetch(`/api/memory/${mergedId}/raw`);
        const data = await response.json();
        
        // Create modal content with synthwave styling
        const modalHtml = `
            <div class="modal fade" id="mergeGroupModal" tabindex="-1">
                <div class="modal-dialog modal-xl">
                    <div class="modal-content bg-dark border-cyan">
                        <div class="modal-header border-cyan">
                            <h5 class="modal-title text-cyan">
                                <i class="bi bi-layers text-warning"></i> Merge Group Details
                                <span class="badge bg-info ms-2">${data.total_raw} events</span>
                                <span class="badge bg-purple ms-2">ID: ${mergedId.substring(0, 8)}...</span>
                            </h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body" style="max-height: 70vh; overflow-y: auto;">
                            <div class="mb-3">
                                <small class="text-muted">
                                    This merge group contains ${data.total_raw} similar events that were automatically grouped based on semantic similarity.
                                </small>
                            </div>
                            <div class="accordion accordion-flush" id="mergeGroupAccordion">
                                ${data.raw_events.map((event, idx) => {
                                    const eventId = event.id || `event_${idx}`;
                                    const isFirst = idx === 0;
                                    return `
                                    <div class="accordion-item bg-dark border-secondary mb-2">
                                        <h2 class="accordion-header" id="heading${idx}">
                                            <button class="accordion-button ${!isFirst ? 'collapsed' : ''} bg-dark text-white" 
                                                    type="button" 
                                                    data-bs-toggle="collapse" 
                                                    data-bs-target="#collapse${idx}" 
                                                    aria-expanded="${isFirst}" 
                                                    aria-controls="collapse${idx}">
                                                <div class="d-flex justify-content-between align-items-center w-100 me-3">
                                                    <div>
                                                        <span class="badge bg-primary me-2">Event ${idx + 1}</span>
                                                        <span class="badge bg-${event.event_type === 'user_input' ? 'primary' : 
                                                                         event.event_type === 'assistant_response' ? 'success' : 
                                                                         'secondary'} me-2">
                                                            ${event.event_type}
                                                        </span>
                                                        <small class="text-info">${escapeHtml(event.five_w1h.who || 'Unknown')}</small>
                                                    </div>
                                                    <small class="text-muted">${formatDate(event.five_w1h.when || event.timestamp)}</small>
                                                </div>
                                            </button>
                                        </h2>
                                        <div id="collapse${idx}" 
                                             class="accordion-collapse collapse ${isFirst ? 'show' : ''}" 
                                             aria-labelledby="heading${idx}" 
                                             data-bs-parent="#mergeGroupAccordion">
                                            <div class="accordion-body bg-darker">
                                                <div class="row g-3">
                                                    <div class="col-12">
                                                        <div class="card bg-dark border-secondary">
                                                            <div class="card-body">
                                                                <h6 class="text-cyan mb-3">
                                                                    <i class="bi bi-info-circle"></i> Event Context (5W1H)
                                                                </h6>
                                                                <div class="row g-2">
                                                                    <div class="col-md-6">
                                                                        <small class="text-muted d-block">Who</small>
                                                                        <div class="text-white">${escapeHtml(event.five_w1h.who || '—')}</div>
                                                                    </div>
                                                                    <div class="col-md-6">
                                                                        <small class="text-muted d-block">When</small>
                                                                        <div class="text-white">${formatDate(event.five_w1h.when || event.timestamp)}</div>
                                                                    </div>
                                                                    <div class="col-md-6">
                                                                        <small class="text-muted d-block">Where</small>
                                                                        <div class="text-white">${escapeHtml(event.five_w1h.where || '—')}</div>
                                                                    </div>
                                                                    <div class="col-md-6">
                                                                        <small class="text-muted d-block">How</small>
                                                                        <div class="text-white">${escapeHtml(event.five_w1h.how || '—')}</div>
                                                                    </div>
                                                                    <div class="col-12">
                                                                        <small class="text-muted d-block">What</small>
                                                                        <div class="text-white bg-darker p-2 rounded mt-1" style="max-height: 150px; overflow-y: auto;">
                                                                            ${escapeHtml(event.five_w1h.what || '—')}
                                                                        </div>
                                                                    </div>
                                                                    <div class="col-12">
                                                                        <small class="text-muted d-block">Why</small>
                                                                        <div class="text-white">${escapeHtml(event.five_w1h.why || '—')}</div>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div class="col-12">
                                                        <div class="d-flex gap-2">
                                                            <small class="text-muted">Episode:</small>
                                                            <code class="text-cyan">${event.episode_id}</code>
                                                        </div>
                                                        <div class="d-flex gap-2">
                                                            <small class="text-muted">Raw Event ID:</small>
                                                            <code class="text-warning">${eventId}</code>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    `;
                                }).join('')}
                            </div>
                        </div>
                        <div class="modal-footer border-cyan">
                            <div class="me-auto">
                                <small class="text-muted">
                                    <i class="bi bi-info-circle"></i> 
                                    Events merged based on similarity threshold: 0.15
                                </small>
                            </div>
                            <button type="button" class="btn btn-outline-cyan" data-bs-dismiss="modal">
                                <i class="bi bi-x-circle"></i> Close
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Remove existing modal if any
        const existingModal = document.getElementById('mergeGroupModal');
        if (existingModal) {
            existingModal.remove();
        }
        
        // Add modal to body
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('mergeGroupModal'));
        modal.show();
        
    } catch (error) {
        console.error('Error loading merge group:', error);
        alert('Failed to load merge group details');
    }
};

window.deleteMemory = async function(memoryId) {
    if (!confirm('Are you sure you want to delete this memory?')) return;
    
    try {
        const response = await fetch(`/api/memories/${memoryId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            loadMemories();
            showNotification('Memory deleted successfully', 'success');
        } else {
            showNotification('Failed to delete memory', 'danger');
        }
    } catch (error) {
        console.error('Error deleting memory:', error);
        showNotification('Failed to delete memory', 'danger');
    }
}

window.viewMemoryDetailsOriginal = async function(memoryId) {
    // Find the memory in allMemories (it's already an array)
    let memory = Array.isArray(allMemories) ? allMemories.find(m => m.id === memoryId) : null;
    
    if (!memory) {
        console.error('Memory not found:', memoryId);
        console.log('Available memory IDs:', allMemories.map(m => m.id));
        return;
    }
    
    // Extract 5W1H fields - handle both nested (raw) and direct (merged) structures
    const who = memory.five_w1h?.who || memory.who || '—';
    const what = memory.five_w1h?.what || memory.what || '—';
    const when = memory.five_w1h?.when || memory.when || memory.timestamp || '—';
    const where = memory.five_w1h?.where || memory.where || '—';
    const why = memory.five_w1h?.why || memory.why || '—';
    const how = memory.five_w1h?.how || memory.how || '—';
    
    // Use real space weights if available, otherwise calculate from fields
    let euclideanPct, hyperbolicPct;
    
    if (memory.euclidean_weight !== undefined && memory.hyperbolic_weight !== undefined) {
        euclideanPct = Math.round(memory.euclidean_weight * 100);
        hyperbolicPct = Math.round(memory.hyperbolic_weight * 100);
    } else {
        // Fallback to field-based calculation
        let concreteScore = 0;
        let abstractScore = 0;
        
        if (who && who !== '—') concreteScore += 1.0;
        if (what && what !== '—') concreteScore += 2.0;
        if (when && when !== '—') concreteScore += 0.5;
        if (where && where !== '—') concreteScore += 0.5;
        if (why && why !== '—') abstractScore += 1.5;
        if (how && how !== '—') abstractScore += 1.0;
        
        const total = concreteScore + abstractScore;
        euclideanPct = total > 0 ? Math.round((concreteScore / total) * 100) : 50;
        hyperbolicPct = 100 - euclideanPct;
    }
    
    // Determine residual status color
    let residualColor = 'success';
    let residualStatus = 'Low';
    if (memory.residual_norm) {
        const norm = parseFloat(memory.residual_norm);
        if (norm > 0.3) {
            residualColor = 'danger';
            residualStatus = 'High';
        } else if (norm > 0.2) {
            residualColor = 'warning';
            residualStatus = 'Medium';
        }
    }
    
    // Create a modal to show details
    const modalHtml = `
        <div class="modal fade" id="memoryDetailModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content" style="background-color: #1a1a2e;">
                    <div class="modal-header">
                        <h5 class="modal-title text-cyan">Memory Details</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <!-- Basic Information -->
                        <h6 class="text-purple mb-3">Basic Information</h6>
                        <div class="mb-3">
                            <div class="text-info small">Memory ID</div>
                            <div class="text-white ms-3">${escapeHtml(memory.id || '—')}</div>
                        </div>
                        <div class="mb-3">
                            <div class="text-info small">Episode ID</div>
                            <div class="text-white ms-3">${escapeHtml(memory.episode_id || '—')}</div>
                        </div>
                        <div class="mb-4">
                            <div class="text-info small">Event Type</div>
                            <div class="ms-3"><span class="badge bg-secondary">${escapeHtml(memory.type || memory.event_type || 'observation')}</span></div>
                        </div>
                        
                        <!-- 5W1H Fields -->
                        <h6 class="text-purple mb-3">Context Information (5W1H)</h6>
                        
                        <div class="mb-3">
                            <div class="text-info small">Who</div>
                            <div class="text-white ms-3">${escapeHtml(who)}</div>
                        </div>
                        
                        <div class="mb-3">
                            <div class="text-info small">What</div>
                            <div class="ms-3">
                                <div class="p-2 bg-dark rounded text-white" style="background-color: #0f0f1a !important;">
                                    ${escapeHtml(what)}
                                </div>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <div class="text-info small">When</div>
                            <div class="text-white ms-3">${formatDate(when)}</div>
                        </div>
                        
                        <div class="mb-3">
                            <div class="text-info small">Where</div>
                            <div class="text-white ms-3">${escapeHtml(where)}</div>
                        </div>
                        
                        <div class="mb-3">
                            <div class="text-info small">Why</div>
                            <div class="ms-3">
                                <div class="p-2 bg-dark rounded text-white" style="background-color: #0f0f1a !important;">
                                    ${escapeHtml(why)}
                                </div>
                            </div>
                        </div>
                        
                        <div class="mb-4">
                            <div class="text-info small">How</div>
                            <div class="ms-3">
                                <div class="p-2 bg-dark rounded text-white" style="background-color: #0f0f1a !important;">
                                    ${escapeHtml(how)}
                                </div>
                            </div>
                        </div>
                        
                        <!-- Space Usage Information -->
                        <h6 class="text-purple mb-3">Space Encoding</h6>
                        <div class="row mb-3">
                            <div class="col-md-12">
                                <div class="d-flex align-items-center mb-2">
                                    <span class="text-muted me-2">Space Distribution:</span>
                                </div>
                                <div class="progress" style="height: 25px;">
                                    <div class="progress-bar bg-info" role="progressbar" style="width: ${euclideanPct}%">
                                        Euclidean ${euclideanPct}%
                                    </div>
                                    <div class="progress-bar bg-warning" role="progressbar" style="width: ${hyperbolicPct}%">
                                        Hyperbolic ${hyperbolicPct}%
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Residual Information -->
                        <h6 class="text-purple mb-3">Adaptation Metrics</h6>
                        
                        <div class="mb-3">
                            <div class="text-info small">Residual Norm</div>
                            <div class="text-white ms-3">
                                ${memory.residual_norm ? memory.residual_norm.toFixed(4) : '0.0000'}
                                <span class="badge bg-${residualColor} ms-2">${residualStatus}</span>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <div class="text-info small">Has Residual Adaptation</div>
                            <div class="ms-3">
                                <span class="badge ${memory.has_residual ? 'bg-success' : 'bg-secondary'}">
                                    ${memory.has_residual ? 'Yes' : 'No'}
                                </span>
                            </div>
                        </div>
                        
                        ${memory.residual_euclidean_norm ? `
                        <div class="mb-3">
                            <div class="text-info small">Euclidean Residual</div>
                            <div class="text-white ms-3">${memory.residual_euclidean_norm.toFixed(4)}</div>
                        </div>
                        ` : ''}
                        
                        ${memory.residual_hyperbolic_norm ? `
                        <div class="mb-3">
                            <div class="text-info small">Hyperbolic Residual</div>
                            <div class="text-white ms-3">${memory.residual_hyperbolic_norm.toFixed(4)}</div>
                        </div>
                        ` : ''}
                        
                        <!-- Additional Metadata -->
                        ${memory.cluster_id !== undefined ? `
                        <h6 class="text-purple mb-3">Clustering Information</h6>
                        <div class="mb-3">
                            <div class="text-info small">Cluster Assignment</div>
                            <div class="ms-3">
                                <span class="badge bg-primary">
                                    ${memory.cluster_id >= 0 ? 'Cluster ' + memory.cluster_id : 'Noise'}
                                </span>
                            </div>
                        </div>
                        ` : ''}
                    </div>
                    <div class="modal-footer">
                        ${memory.type !== 'raw' ? `
                        <button type="button" class="btn btn-info" onclick="window.showMemoryInGraph('${memoryId}')">
                            <i class="bi bi-diagram-3"></i> Show in Graph
                        </button>
                        ` : ''}
                        ${memory.type !== 'raw' ? `
                        <button type="button" class="btn btn-danger" onclick="window.deleteMemory('${memoryId}')">
                            <i class="bi bi-trash"></i> Delete
                        </button>
                        ` : ''}
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Remove existing modal if any
    const existingModal = document.getElementById('memoryDetailModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('memoryDetailModal'));
    modal.show();
    
    // Clean up after close
    document.getElementById('memoryDetailModal').addEventListener('hidden.bs.modal', function() {
        this.remove();
    });
}

// Function to view merged event details with all variations
async function viewMergedEventDetails(memoryId) {
    try {
        // Check if this is a multi-dimensional merge group (format: type_id)
        const mergeTypePattern = /^(temporal|actor|conceptual|spatial)_/;
        const match = memoryId.match(mergeTypePattern);
        
        let response;
        if (match) {
            // This is a multi-dimensional merge group
            const mergeType = match[1];
            response = await fetch(`/api/multi-merge/${mergeType}/${memoryId}/details`);
        } else {
            // Try standard merged event endpoint
            response = await fetch(`/api/memory/${memoryId}/merged-details`);
        }
        
        if (!response.ok) {
            // Not a merged event, fall back to regular details
            console.log(`Memory ${memoryId} is not a merged event, showing regular details`);
            return viewMemoryDetailsOriginal(memoryId);
        }
        
        const mergedEvent = await response.json();
        
        // Remove existing modal if any
        const existingModal = document.getElementById('mergedEventModal');
        if (existingModal) {
            const existingModalInstance = bootstrap.Modal.getInstance(existingModal);
            if (existingModalInstance) {
                existingModalInstance.hide();
                existingModalInstance.dispose();
            }
            existingModal.remove();
        }
        
        // Clean up any stray backdrops
        document.querySelectorAll('.modal-backdrop').forEach(backdrop => backdrop.remove());
        
        // Create enhanced modal for merged event
        const modalHtml = createMergedEventModal(mergedEvent);
        
        // Add modal to body using insertAdjacentHTML to preserve structure
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        
        // Get the newly added modal element
        const modalElement = document.getElementById('mergedEventModal');
        
        // Initialize tabs and other interactive elements
        initializeMergedEventModal(mergedEvent);
        
        // Create and show the modal using Bootstrap's standard approach
        const modal = new bootstrap.Modal(modalElement);
        modal.show();
        
        // Clean up after close
        modalElement.addEventListener('hidden.bs.modal', function() {
            const instance = bootstrap.Modal.getInstance(this);
            if (instance) {
                instance.dispose();
            }
            this.remove();
        }, { once: true });
    } catch (error) {
        console.error('Error fetching merged event details:', error);
        // Clean up any backdrops on error
        document.querySelectorAll('.modal-backdrop').forEach(backdrop => backdrop.remove());
        // Fall back to regular view
        return viewMemoryDetailsOriginal(memoryId);
    }
}

// Create the merged event modal HTML
function createMergedEventModal(mergedEvent) {
    const latest = mergedEvent.latest_state || {};
    
    // Determine if this is a multi-dimensional merge
    const isMultiDimensional = mergedEvent.merge_type && mergedEvent.merge_key;
    const title = isMultiDimensional 
        ? `${mergedEvent.merge_type.charAt(0).toUpperCase() + mergedEvent.merge_type.slice(1)} Merge Group`
        : 'Merged Event Details';
    
    return `
        <div class="modal fade" id="mergedEventModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content" style="background-color: #1a1a2e;">
                    <div class="modal-header">
                        <h5 class="modal-title text-cyan">
                            <i class="bi bi-layers"></i> ${title}
                            <span class="badge bg-info ms-2">${mergedEvent.merge_count || mergedEvent.event_count || 0} events</span>
                        </h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <!-- Tab Navigation -->
                        <ul class="nav nav-tabs mb-3" role="tablist">
                            <li class="nav-item">
                                <a class="nav-link active" data-bs-toggle="tab" href="#mergedOverview">Overview</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" data-bs-toggle="tab" href="#mergedTimeline">Timeline</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" data-bs-toggle="tab" href="#mergedContext">LLM Context</a>
                            </li>
                        </ul>
                        
                        <!-- Tab Content -->
                        <div class="tab-content">
                            <!-- Overview Tab -->
                            <div class="tab-pane fade show active" id="mergedOverview">
                                ${createMergedOverviewTab(mergedEvent, latest)}
                            </div>
                            
                            <!-- Timeline Tab -->
                            <div class="tab-pane fade" id="mergedTimeline">
                                ${createMergedVariationsTab(mergedEvent)}
                            </div>
                            
                            <!-- Context Tab -->
                            <div class="tab-pane fade" id="mergedContext">
                                ${createMergedContextTab(mergedEvent)}
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-info" onclick="window.showMemoryInGraph('${mergedEvent.id}')">
                            <i class="bi bi-diagram-3"></i> Show in Graph
                        </button>
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Create overview tab content
function createMergedOverviewTab(mergedEvent, latest) {
    // Helper function to create component table
    const createComponentTable = (title, variants, displayField = 'value') => {
        if (!variants || Object.keys(variants).length === 0) {
            return `<div class="mb-3">
                <div class="text-info small">${title}</div>
                <div class="text-white ms-3">—</div>
            </div>`;
        }
        
        let tableHtml = `
            <div class="mb-4">
                <h6 class="text-info">${title}</h6>
                <div class="table-responsive">
                    <table class="table table-sm table-dark table-striped">
                        <thead>
                            <tr>
                                <th>Value</th>
                                <th>Count</th>
                                <th>Last Updated</th>
                            </tr>
                        </thead>
                        <tbody>`;
        
        // Aggregate all variants
        const aggregated = {};
        for (const [key, variantList] of Object.entries(variants)) {
            for (const variant of variantList) {
                const value = variant[displayField] || variant.value || key;
                if (!aggregated[value]) {
                    aggregated[value] = {
                        count: 0,
                        lastUpdated: variant.timestamp
                    };
                }
                // Use the variant's count if available, otherwise increment by 1
                const variantCount = variant.count || 1;
                aggregated[value].count += variantCount;
                if (variant.timestamp > aggregated[value].lastUpdated) {
                    aggregated[value].lastUpdated = variant.timestamp;
                }
            }
        }
        
        // Sort by count descending
        const sorted = Object.entries(aggregated).sort((a, b) => b[1].count - a[1].count);
        
        for (const [value, data] of sorted.slice(0, 10)) { // Show top 10
            tableHtml += `
                <tr>
                    <td class="text-truncate" style="max-width: 300px;" title="${escapeHtml(value)}">
                        ${escapeHtml(value.substring(0, 100))}
                    </td>
                    <td><span class="badge bg-secondary">${data.count}</span></td>
                    <td class="text-muted small">${formatDate(data.lastUpdated)}</td>
                </tr>`;
        }
        
        tableHtml += `
                        </tbody>
                    </table>
                </div>
                ${sorted.length > 10 ? `<div class="text-muted small">Showing 10 of ${sorted.length} total values</div>` : ''}
            </div>`;
        
        return tableHtml;
    };
    
    // Helper function to create component rows for unified table
    const createComponentRows = (title, variants, displayField = 'value') => {
        if (!variants || Object.keys(variants).length === 0) {
            return `
                <tr class="table-secondary">
                    <td colspan="3"><strong>${title}</strong></td>
                </tr>
                <tr>
                    <td colspan="3" class="text-muted ps-4">No data available</td>
                </tr>`;
        }
        
        // Aggregate all variants
        const aggregated = {};
        for (const [key, variantList] of Object.entries(variants)) {
            for (const variant of variantList) {
                const value = variant[displayField] || variant.value || key;
                if (!aggregated[value]) {
                    aggregated[value] = {
                        count: 0,
                        lastUpdated: variant.timestamp
                    };
                }
                // Use the variant's count if available, otherwise increment by 1
                const variantCount = variant.count || 1;
                aggregated[value].count += variantCount;
                if (variant.timestamp > aggregated[value].lastUpdated) {
                    aggregated[value].lastUpdated = variant.timestamp;
                }
            }
        }
        
        // Sort by count descending
        const sorted = Object.entries(aggregated).sort((a, b) => b[1].count - a[1].count);
        
        let rows = `
            <tr class="table-secondary">
                <td colspan="3"><strong>${title}</strong></td>
            </tr>`;
        
        for (const [value, data] of sorted.slice(0, 5)) { // Show top 5 per section
            rows += `
                <tr>
                    <td class="text-truncate ps-4" style="max-width: 400px;" title="${escapeHtml(value)}">
                        ${escapeHtml(value.substring(0, 100))}
                    </td>
                    <td style="width: 100px;"><span class="badge bg-secondary">${data.count}</span></td>
                    <td style="width: 200px;" class="text-muted small">${formatDate(data.lastUpdated)}</td>
                </tr>`;
        }
        
        if (sorted.length > 5) {
            rows += `
                <tr>
                    <td colspan="3" class="text-muted small ps-4">...and ${sorted.length - 5} more</td>
                </tr>`;
        }
        
        return rows;
    };
    
    return `
        <div class="row">
            <div class="col-12">
                <h5 class="text-purple mb-3">Component Summary</h5>
                <div class="table-responsive">
                    <table class="table table-sm table-dark table-hover">
                        <tbody>
                            ${createComponentRows('WHO - Actors', mergedEvent.who_variants)}
                            ${createComponentRows('WHAT - Actions', mergedEvent.what_variants)}
                            ${createComponentRows('WHERE - Locations', mergedEvent.where_locations)}
                            ${createComponentRows('WHY - Reasons', mergedEvent.why_variants)}
                            ${createComponentRows('HOW - Methods', mergedEvent.how_variants || (mergedEvent.how_methods ? {'methods': Object.entries(mergedEvent.how_methods).map(([method, count]) => ({value: method, timestamp: new Date()}))} : {}))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col-md-12">
                <h5 class="text-purple mb-3">Merge Statistics</h5>
                <div class="mb-3">
                    <div class="text-info small">Total Events Merged</div>
                    <div class="text-white ms-3">${mergedEvent.merge_count}</div>
                </div>
                <div class="mb-3">
                    <div class="text-info small">Base Event ID</div>
                    <div class="text-white ms-3 font-monospace">${escapeHtml(mergedEvent.base_event_id)}</div>
                </div>
                <div class="mb-3">
                    <div class="text-info small">Created</div>
                    <div class="text-white ms-3">${formatDate(mergedEvent.created_at)}</div>
                </div>
                <div class="mb-3">
                    <div class="text-info small">Last Updated</div>
                    <div class="text-white ms-3">${formatDate(mergedEvent.last_updated)}</div>
                </div>
                ${mergedEvent.supersedes ? `
                <div class="mb-3">
                    <div class="text-info small">Supersedes</div>
                    <div class="text-white ms-3">${escapeHtml(mergedEvent.supersedes)}</div>
                </div>
                ` : ''}
                ${mergedEvent.superseded_by ? `
                <div class="mb-3">
                    <div class="text-info small">Superseded By</div>
                    <div class="text-white ms-3">${escapeHtml(mergedEvent.superseded_by)}</div>
                </div>
                ` : ''}
            </div>
        </div>
    `;
}

// Create enhanced LLM context display
function createEnhancedLLMContextDisplay(llmContext) {
    if (!llmContext || Object.keys(llmContext).length === 0) {
        return '<p class="text-muted text-center">No LLM context available</p>';
    }
    
    let html = '<div class="llm-context-display">';
    
    // Display narrative summary
    if (llmContext.narrative_summary) {
        html += `
            <div class="mb-3">
                <h6 class="text-cyan mb-2"><i class="bi bi-file-text"></i> Summary</h6>
                <div class="p-2 bg-dark bg-opacity-50 rounded">
                    <p class="mb-0">${escapeHtml(llmContext.narrative_summary)}</p>
                </div>
            </div>
        `;
    }
    
    // Display key information
    if (llmContext.key_information && Object.keys(llmContext.key_information).length > 0) {
        html += `
            <div class="mb-3">
                <h6 class="text-cyan mb-2"><i class="bi bi-info-square"></i> Key Information</h6>
                <div class="p-2 bg-dark bg-opacity-50 rounded">
                    <dl class="row mb-0">
        `;
        
        for (const [key, value] of Object.entries(llmContext.key_information)) {
            const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            let displayValue = value;
            
            if (Array.isArray(value)) {
                // Handle arrays of objects or strings
                if (value.length > 0) {
                    if (typeof value[0] === 'object') {
                        // Extract a meaningful property from objects
                        displayValue = value.map(v => v.value || v.name || JSON.stringify(v)).join(', ');
                    } else {
                        displayValue = value.join(', ');
                    }
                } else {
                    displayValue = 'None';
                }
            } else if (typeof value === 'object' && value !== null) {
                displayValue = JSON.stringify(value);
            }
            
            html += `
                <dt class="col-sm-4 text-muted">${escapeHtml(displayKey)}:</dt>
                <dd class="col-sm-8">${escapeHtml(String(displayValue))}</dd>
            `;
        }
        
        html += `
                    </dl>
                </div>
            </div>
        `;
    }
    
    // Display patterns
    if (llmContext.patterns && llmContext.patterns.length > 0) {
        html += `
            <div class="mb-3">
                <h6 class="text-cyan mb-2"><i class="bi bi-diagram-3"></i> Patterns Identified</h6>
                <div class="p-2 bg-dark bg-opacity-50 rounded">
                    <ul class="mb-0">
                        ${llmContext.patterns.map(pattern => `<li>${escapeHtml(pattern)}</li>`).join('')}
                    </ul>
                </div>
            </div>
        `;
    }
    
    // Display relationships
    if (llmContext.relationships && llmContext.relationships.length > 0) {
        html += `
            <div class="mb-3">
                <h6 class="text-cyan mb-2"><i class="bi bi-people"></i> Relationships</h6>
                <div class="p-2 bg-dark bg-opacity-50 rounded">
                    <ul class="mb-0">
                        ${llmContext.relationships.map(rel => `<li>${escapeHtml(rel)}</li>`).join('')}
                    </ul>
                </div>
            </div>
        `;
    }
    
    // Display timeline summary (not the full events, just a summary)
    if (llmContext.timeline && llmContext.timeline.length > 0) {
        const timelinePreview = llmContext.timeline.slice(0, 3);
        html += `
            <div class="mb-3">
                <h6 class="text-cyan mb-2"><i class="bi bi-clock-history"></i> Recent Events</h6>
                <div class="p-2 bg-dark bg-opacity-50 rounded">
                    <div class="small">
        `;
        
        timelinePreview.forEach((entry, idx) => {
            html += `
                <div class="mb-2 pb-2 ${idx < timelinePreview.length - 1 ? 'border-bottom border-secondary' : ''}">
                    <div class="text-muted">Event ${idx + 1}</div>
                    <div class="text-white small">${escapeHtml(entry.formatted)}</div>
                </div>
            `;
        });
        
        if (llmContext.timeline.length > 3) {
            html += `<div class="text-muted text-center">... and ${llmContext.timeline.length - 3} more events</div>`;
        }
        
        html += `
                    </div>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    return html;
}

// Create raw events timeline for multi-dimensional merge groups
function createRawEventsTimeline(rawEvents) {
    if (!rawEvents || rawEvents.length === 0) {
        return '<p class="text-muted text-center">No events in this merge group</p>';
    }
    
    return `
        <div class="raw-events-timeline">
            <div class="mb-3">
                <small class="text-muted">
                    This merge group contains ${rawEvents.length} events that form a coherent conversation or workflow.
                </small>
            </div>
            <div class="accordion accordion-flush" id="rawEventsAccordion">
                ${rawEvents.map((event, idx) => {
                    const eventId = event.id || `event_${idx}`;
                    const isFirst = idx === 0;
                    const five_w1h = event.five_w1h || {};
                    
                    return `
                    <div class="accordion-item bg-dark border-secondary mb-2">
                        <h2 class="accordion-header" id="heading${idx}">
                            <button class="accordion-button ${!isFirst ? 'collapsed' : ''} bg-dark text-white" 
                                    type="button" 
                                    data-bs-toggle="collapse" 
                                    data-bs-target="#collapse${idx}" 
                                    aria-expanded="${isFirst}" 
                                    aria-controls="collapse${idx}">
                                <div class="d-flex justify-content-between align-items-center w-100 me-3">
                                    <div>
                                        <span class="badge bg-primary me-2">Event ${idx + 1}</span>
                                        <span class="badge bg-${event.event_type === 'user_input' ? 'info' : 
                                                         event.event_type === 'assistant_response' ? 'success' : 
                                                         event.event_type === 'action' ? 'warning' :
                                                         'secondary'} me-2">
                                            ${event.event_type || 'observation'}
                                        </span>
                                        <small class="text-info">${escapeHtml(five_w1h.who || 'Unknown')}</small>
                                    </div>
                                    <small class="text-muted">${formatDate(five_w1h.when || event.timestamp)}</small>
                                </div>
                            </button>
                        </h2>
                        <div id="collapse${idx}" 
                             class="accordion-collapse collapse ${isFirst ? 'show' : ''}" 
                             aria-labelledby="heading${idx}" 
                             data-bs-parent="#rawEventsAccordion">
                            <div class="accordion-body bg-darker">
                                <div class="row g-3">
                                    <div class="col-12">
                                        <div class="card bg-dark border-secondary">
                                            <div class="card-body">
                                                <h6 class="text-cyan mb-3">
                                                    <i class="bi bi-info-circle"></i> Event Context (5W1H)
                                                </h6>
                                                <div class="row g-2">
                                                    <div class="col-md-6">
                                                        <small class="text-muted d-block">Who</small>
                                                        <div class="text-white">${escapeHtml(five_w1h.who || '—')}</div>
                                                    </div>
                                                    <div class="col-12">
                                                        <small class="text-muted d-block">What</small>
                                                        <div class="text-white bg-darker p-2 rounded mt-1" style="max-height: 200px; overflow-y: auto;">
                                                            ${escapeHtml(five_w1h.what || '—')}
                                                        </div>
                                                    </div>
                                                    <div class="col-md-6">
                                                        <small class="text-muted d-block">When</small>
                                                        <div class="text-white">${formatDate(five_w1h.when || event.timestamp)}</div>
                                                    </div>
                                                    <div class="col-md-6">
                                                        <small class="text-muted d-block">Where</small>
                                                        <div class="text-white">${escapeHtml(five_w1h.where || '—')}</div>
                                                    </div>
                                                    <div class="col-12">
                                                        <small class="text-muted d-block">Why</small>
                                                        <div class="text-white">${escapeHtml(five_w1h.why || '—')}</div>
                                                    </div>
                                                    <div class="col-12">
                                                        <small class="text-muted d-block">How</small>
                                                        <div class="text-white">${escapeHtml(five_w1h.how || '—')}</div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-12">
                                        <div class="d-flex gap-2">
                                            <small class="text-muted">Episode:</small>
                                            <code class="text-cyan">${event.episode_id || 'N/A'}</code>
                                        </div>
                                        <div class="d-flex gap-2">
                                            <small class="text-muted">Event ID:</small>
                                            <code class="text-warning">${eventId}</code>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    `;
                }).join('')}
            </div>
        </div>
    `;
}

// Create timeline tab content
function createMergedTimelineTab(mergedEvent) {
    if (!mergedEvent.when_timeline || mergedEvent.when_timeline.length === 0) {
        return '<p class="text-muted">No timeline data available</p>';
    }
    
    let html = '<div class="timeline">';
    mergedEvent.when_timeline.forEach((point, index) => {
        html += `
            <div class="timeline-item mb-3">
                <div class="d-flex align-items-start">
                    <div class="badge bg-primary me-3">${index + 1}</div>
                    <div class="flex-grow-1">
                        <div class="text-info small">${formatDate(point.timestamp)}</div>
                        <div class="text-white">${escapeHtml(point.description || 'Event occurred')}</div>
                        ${point.semantic_time ? `<div class="text-muted small">${escapeHtml(point.semantic_time)}</div>` : ''}
                        <div class="text-muted small font-monospace">Event: ${escapeHtml(point.event_id)}</div>
                    </div>
                </div>
            </div>
        `;
    });
    html += '</div>';
    return html;
}

// Create variations tab content (also handles raw events for multi-dimensional merges)
function createMergedVariationsTab(mergedEvent) {
    // Check if we have raw_events (for multi-dimensional merge groups)
    if (mergedEvent.raw_events && mergedEvent.raw_events.length > 0) {
        // Display raw events as timeline
        return createRawEventsTimeline(mergedEvent.raw_events);
    }
    
    // Check if there are any variations (for standard merges)
    const hasVariations = 
        Object.keys(mergedEvent.who_variants || {}).length > 0 ||
        Object.keys(mergedEvent.what_variants || {}).length > 0 ||
        Object.keys(mergedEvent.where_locations || {}).length > 0 ||
        Object.keys(mergedEvent.why_variants || {}).length > 0 ||
        Object.keys(mergedEvent.how_methods || {}).length > 0;
    
    if (!hasVariations) {
        return '<div class="p-3 text-center text-muted">No component variations available for this event.</div>';
    }
    
    let html = '<div class="accordion accordion-flush" id="variationsAccordion">';
    
    // WHO variations
    if (Object.keys(mergedEvent.who_variants || {}).length > 0) {
        html += createVariationAccordionItem('who', 'Who', mergedEvent.who_variants);
    }
    
    // WHAT variations
    if (Object.keys(mergedEvent.what_variants || {}).length > 0) {
        html += createVariationAccordionItem('what', 'What', mergedEvent.what_variants);
    }
    
    // WHERE locations
    if (mergedEvent.where_locations && mergedEvent.where_locations.locations && mergedEvent.where_locations.locations.length > 0) {
        const locations = mergedEvent.where_locations.locations;
        html += `
            <div class="accordion-item bg-dark border-secondary">
                <h2 class="accordion-header">
                    <button class="accordion-button collapsed bg-dark text-white" type="button" data-bs-toggle="collapse" data-bs-target="#whereVariations">
                        <span class="text-cyan">Where</span> <span class="badge bg-info ms-2">${locations.length}</span>
                    </button>
                </h2>
                <div id="whereVariations" class="accordion-collapse collapse" data-bs-parent="#variationsAccordion">
                    <div class="accordion-body bg-dark">
                        ${locations.map(loc => `
                            <div class="mb-2">
                                <span class="text-white">${escapeHtml(loc.value || 'Unknown')}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }
    
    // WHY variations
    if (Object.keys(mergedEvent.why_variants || {}).length > 0) {
        html += createVariationAccordionItem('why', 'Why', mergedEvent.why_variants);
    }
    
    // HOW methods - handle both how_variants.methods and how_methods formats
    const howMethods = mergedEvent.how_variants?.methods || 
                      (mergedEvent.how_methods ? Object.entries(mergedEvent.how_methods).map(([method, count]) => ({value: method, count})) : []);
    if (howMethods.length > 0) {
        html += `
            <div class="accordion-item bg-dark border-secondary">
                <h2 class="accordion-header">
                    <button class="accordion-button collapsed bg-dark text-white" type="button" data-bs-toggle="collapse" data-bs-target="#howVariations">
                        <span class="text-cyan">How</span> <span class="badge bg-info ms-2">${howMethods.length}</span>
                    </button>
                </h2>
                <div id="howVariations" class="accordion-collapse collapse" data-bs-parent="#variationsAccordion">
                    <div class="accordion-body bg-dark">
                        ${howMethods.map(method => `
                            <div class="mb-2">
                                <span class="text-white">${escapeHtml(method.value || 'Unknown')}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    return html;
}

// Helper function to create variation accordion items
function createVariationAccordionItem(id, label, variants) {
    const variantCount = Object.values(variants).reduce((sum, v) => sum + v.length, 0);
    
    return `
        <div class="accordion-item bg-dark border-secondary">
            <h2 class="accordion-header">
                <button class="accordion-button collapsed bg-dark text-white" type="button" data-bs-toggle="collapse" data-bs-target="#${id}Variations">
                    <span class="text-cyan">${label}</span> <span class="badge bg-info ms-2">${variantCount}</span>
                </button>
            </h2>
            <div id="${id}Variations" class="accordion-collapse collapse" data-bs-parent="#variationsAccordion">
                <div class="accordion-body bg-dark">
                    ${Object.entries(variants).map(([key, variantList]) => `
                        <div class="mb-3">
                            <h6 class="text-info">${escapeHtml(key.substring(0, 50))}</h6>
                            ${variantList.map(v => `
                                <div class="ms-3 mb-2 p-2 bg-dark rounded">
                                    <div class="d-flex justify-content-between align-items-start">
                                        <div class="flex-grow-1">
                                            <div class="text-white">${escapeHtml(v.value)}</div>
                                            <div class="text-muted small">
                                                ${formatDate(v.timestamp)} 
                                                <span class="badge bg-secondary ms-2">${v.relationship}</span>
                                                ${v.version > 1 ? `<span class="badge bg-primary ms-1">v${v.version}</span>` : ''}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
}

// Create context tab content
function createMergedContextTab(mergedEvent) {
    // Check if we have enhanced LLM context (for multi-dimensional merges)
    if (mergedEvent.llm_context && Object.keys(mergedEvent.llm_context).length > 0) {
        return createEnhancedLLMContextDisplay(mergedEvent.llm_context);
    }
    
    // Fall back to old context_preview format
    const context = mergedEvent.context_preview || {};
    
    return `
        <div>
            <h6 class="text-purple mb-3">Summary Context</h6>
            <div class="p-3 bg-dark rounded mb-4">
                <pre class="text-white mb-0">${escapeHtml(context.summary || 'No summary available')}</pre>
            </div>
            
            <h6 class="text-purple mb-3">Detailed Context (As Seen by LLM)</h6>
            <div class="p-3 bg-dark rounded">
                <pre class="text-white mb-0">${escapeHtml(context.detailed || 'No detailed context available')}</pre>
            </div>
        </div>
    `;
}

// Initialize merged event modal interactions
function initializeMergedEventModal(mergedEvent) {
    // Any additional initialization for interactive elements
    console.log('Merged event modal initialized for:', mergedEvent.id);
}

// Enhanced function to check if memory is merged and display accordingly
window.viewMemoryDetails = async function(memoryId) {
    // Check if this is a multi-dimensional merge ID (e.g., temporal_xxx, actor_xxx, etc.)
    const multiDimPattern = /^(temporal|actor|conceptual|spatial)_/;
    if (multiDimPattern.test(memoryId)) {
        // This is a multi-dimensional merge group, use the unified merged event viewer
        return viewMergedEventDetails(memoryId);
    }
    
    // Check if this is a raw event (raw events have IDs starting with 'raw_')
    if (memoryId.startsWith('raw_')) {
        // Raw events should use the original details viewer
        return window.viewMemoryDetailsOriginal(memoryId);
    }
    
    // For non-raw events, try to get it as a merged event
    try {
        await viewMergedEventDetails(memoryId);
    } catch (error) {
        console.error('Error in viewMergedEventDetails:', error);
        // Fall back to original view
        window.viewMemoryDetailsOriginal(memoryId);
    }
};

// Function to update edge filtering based on relation strength threshold
function updateEdgeFiltering(threshold) {
    if (!graphNetwork || !allGraphData) return;
    
    relationStrengthThreshold = threshold;
    
    // Get the edges dataset from the network
    const edgesDataset = graphNetwork.body.data.edges;
    const nodesDataset = graphNetwork.body.data.nodes;
    
    // Only update if we have valid datasets
    if (!edgesDataset || !nodesDataset) {
        console.warn('Datasets not ready for filtering');
        return;
    }
    
    const updates = [];
    
    // Go through all edges and update visibility
    edgesDataset.forEach(edge => {
        const shouldHide = !(edge.type === 'conversation' || edge.weight >= relationStrengthThreshold);
        updates.push({
            id: edge.id,
            hidden: shouldHide
        });
    });
    
    // Apply all updates at once
    edgesDataset.update(updates);
    
    // Update edge count in statistics
    const visibleCount = allGraphData.edges.filter(edge => 
        edge.type === 'conversation' || edge.weight >= relationStrengthThreshold
    ).length;
    document.getElementById('edgeCount').textContent = visibleCount;
    
    // Find connected components based on visible edges
    try {
        const connectedComponents = findConnectedComponents(nodesDataset, edgesDataset, relationStrengthThreshold);
        
        // Update node clustering to reflect actual connectivity
        const nodeUpdates = [];
        connectedComponents.forEach((component, componentId) => {
            component.forEach(nodeId => {
                nodeUpdates.push({
                    id: nodeId,
                    group: componentId  // Use group to visually distinguish components
                });
            });
        });
        
        if (nodeUpdates.length > 0) {
            nodesDataset.update(nodeUpdates);
        }
    } catch (error) {
        console.error('Error updating connected components:', error);
    }
    
    // Simply restart physics without temporary changes
    graphNetwork.setOptions({
        physics: {
            enabled: true,
            stabilization: {
                enabled: true,
                iterations: 100,
                updateInterval: 50
            }
        }
    });
    
    // Let physics run to separate disconnected components naturally
    graphNetwork.stabilize(100);
}

// Helper function to find connected components
function findConnectedComponents(nodesDataset, edgesDataset, threshold) {
    const adjacencyList = {};
    const visited = new Set();
    const components = [];
    const nodeIds = [];
    
    // Get all node IDs
    nodesDataset.forEach(node => {
        nodeIds.push(node.id);
        adjacencyList[node.id] = [];  // Initialize with empty array
    });
    
    // Build adjacency list from visible edges
    edgesDataset.forEach(edge => {
        if (!edge.hidden && (edge.type === 'conversation' || edge.weight >= threshold)) {
            // Only add if both nodes exist
            if (adjacencyList[edge.from] !== undefined && adjacencyList[edge.to] !== undefined) {
                adjacencyList[edge.from].push(edge.to);
                adjacencyList[edge.to].push(edge.from);
            }
        }
    });
    
    // DFS to find connected components
    function dfs(nodeId, component) {
        visited.add(nodeId);
        component.push(nodeId);
        
        if (adjacencyList[nodeId] && adjacencyList[nodeId].length > 0) {
            adjacencyList[nodeId].forEach(neighbor => {
                if (!visited.has(neighbor)) {
                    dfs(neighbor, component);
                }
            });
        }
    }
    
    // Find all components
    nodeIds.forEach(nodeId => {
        if (!visited.has(nodeId)) {
            const component = [];
            dfs(nodeId, component);
            components.push(component);
        }
    });
    
    return components;
}

// Function to update physics configuration for gravity
function updateGraphPhysics(gravity) {
    if (!graphNetwork) return;
    
    currentGravity = gravity;
    
    // Calculate proportional parameters based on gravity
    // More negative gravity = more spacing
    const centralGravity = 0.1 + (gravity + 50000) / 100000;  // Range: 0.1 to 0.6
    const springLength = 100 + (Math.abs(gravity) / 100);  // Range: 110 to 600
    const springConstant = 0.001 + (50000 + gravity) / 5000000;  // Range: 0.001 to 0.01
    
    // Update physics options with better stabilization
    graphNetwork.setOptions({
        physics: {
            enabled: true,
            stabilization: {
                enabled: true,
                iterations: 100,  // Quick re-stabilization
                updateInterval: 50,
                fit: false  // Don't refit view when adjusting
            },
            barnesHut: {
                gravitationalConstant: currentGravity,
                centralGravity: 0.3,  // Keep consistent center pull
                springConstant: 0.04,  // Consistent spring strength
                springLength: 95 + (Math.abs(gravity) / 500),  // Adjust spring length based on gravity
                damping: 0.5,  // High damping to prevent oscillation
                avoidOverlap: 0.1
            },
            timestep: 0.5,
            adaptiveTimestep: true
        }
    });
    
    // Start stabilization then stop physics
    graphNetwork.stabilize(100);
    
    // Stop physics after stabilization to prevent drift
    setTimeout(() => {
        graphNetwork.setOptions({
            physics: {
                enabled: false
            }
        });
        console.log('Physics stopped after adjustment');
    }, 2000);  // Give it 2 seconds to stabilize
}

window.showMemoryInGraph = function(memoryId) {
    // Close any open details modals
    const detailModal = bootstrap.Modal.getInstance(document.getElementById('memoryDetailModal'));
    if (detailModal) {
        detailModal.hide();
    }
    
    // Also close the merged event details modal if it's open
    const mergedModal = bootstrap.Modal.getInstance(document.getElementById('mergedEventModal'));
    if (mergedModal) {
        mergedModal.hide();
        // Give it a moment to close before removing
        setTimeout(() => {
            const mergedElement = document.getElementById('mergedEventModal');
            if (mergedElement) {
                mergedElement.remove();
            }
        }, 300);
    }
    
    // Show loading indicator immediately
    const graphContainer = document.getElementById('memoryGraph');
    if (graphContainer) {
        graphContainer.innerHTML = `
            <div class="d-flex flex-column justify-content-center align-items-center h-100">
                <div class="spinner-grow text-purple" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="mt-3 text-muted">Loading memory in graph...</div>
            </div>
        `;
    }
    
    // Open the graph modal
    const modalElement = document.getElementById('memoryGraphModal');
    const graphModal = new bootstrap.Modal(modalElement);
    
    // Use the modal shown event to initialize the graph
    modalElement.addEventListener('shown.bs.modal', function onModalShown() {
        // Remove this listener after it fires once
        modalElement.removeEventListener('shown.bs.modal', onModalShown);
        
        // Initialize graph with the selected memory as center
        initializeGraph(memoryId).then(() => {
            // Focus on the center node
            if (graphNetwork && memoryId) {
                // Check if the node exists in the dataset
                if (graphNetwork.nodesDataset && graphNetwork.nodesDataset.get(memoryId)) {
                    // The node should already be centered due to server-side filtering
                    // Additional focus for emphasis
                    graphNetwork.fit({
                        animation: {
                            duration: 500,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                    
                    // Select and highlight the center node
                    try {
                        graphNetwork.selectNodes([memoryId]);
                    } catch (e) {
                        console.warn('Could not select node:', e);
                    }
                    
                    // Get the node from the dataset and display details
                    const node = graphNetwork.nodesDataset.get(memoryId);
                    if (node) {
                        displayNodeDetails(node);
                    }
                } else {
                    console.warn('Center node not found in graph:', memoryId);
                    // Just fit the view to show all nodes
                    graphNetwork.fit({
                        animation: {
                            duration: 500,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                }
            }
        }).catch(error => {
            console.error('Error initializing graph:', error);
            // Show error in graph container
            const graphContainer = document.getElementById('memoryGraph');
            if (graphContainer) {
                graphContainer.innerHTML = `
                    <div class="d-flex flex-column justify-content-center align-items-center h-100">
                        <i class="bi bi-exclamation-triangle text-danger" style="font-size: 3rem;"></i>
                        <div class="mt-3 text-danger">Failed to load graph</div>
                        <small class="text-muted mt-2">${error.message || 'Please check console for details'}</small>
                    </div>
                `;
            }
        });
    });
    
    graphModal.show();
}

window.performSearch = async function() {
    const searchQuery = document.getElementById('memorySearch').value;
    
    if (!searchQuery) {
        loadMemories();
        return;
    }
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: searchQuery })
        });
        
        const data = await response.json();
        allMemories = data.results || [];
        paginationCurrentPage = 1;
        displayMemories();
        
    } catch (error) {
        console.error('Error searching:', error);
    }
}

// Utility functions
function updatePagination() {
    // This function is now replaced by createPaginationControls
    createPaginationControls();
}

// Change page - globally accessible
window.changePage = function(page) {
    // Track user interaction
    paginationLastUserInteraction = Date.now();
    window.lastUserInteraction = paginationLastUserInteraction;
    
    const totalPages = Math.ceil(paginationFilteredMemories.length / paginationItemsPerPage);
    
    if (page < 1 || page > totalPages) return;
    
    paginationCurrentPage = page;
    displayMemoriesWithPaging();
}

// Make sortTable globally accessible
window.sortTable = function(field) {
    // Check if we're in merged view
    const isMergedView = document.getElementById('mergedView') && document.getElementById('mergedView').checked;
    
    if (isMergedView) {
        // Handle sorting for merged view
        if (mergeGroupsSortField === field) {
            mergeGroupsSortDirection = mergeGroupsSortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            mergeGroupsSortField = field;
            mergeGroupsSortDirection = 'asc';
        }
        
        // Update sort icons
        document.querySelectorAll('.sortable .sort-icon').forEach(icon => {
            icon.className = 'bi bi-arrow-down-up sort-icon';
        });
        
        const currentHeader = document.querySelector(`[data-field="${field}"] .sort-icon`);
        if (currentHeader) {
            currentHeader.className = mergeGroupsSortDirection === 'asc' 
                ? 'bi bi-arrow-up sort-icon text-cyan' 
                : 'bi bi-arrow-down sort-icon text-cyan';
        }
        
        // Apply sorting and reset to first page
        mergeGroupsCurrentPage = 1;
        sortMergeGroups();
        displayMergeGroupsPage();
    } else {
        // Original sorting logic for raw view
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
}

// Legacy sortTable function for backwards compatibility
function sortTable(field) {
    window.sortTable(field);
}

// Keep the old sorting visualization code for compatibility
function updateSortVisualIndicators() {
    // Update visual indicators
    updateSortIndicators(sortField, sortDirection);
    
    // Reset to first page and display
    currentPage = 1;
    displayMemories();
}

function calculateEuclideanPercentage(memory) {
    // Calculate euclidean percentage for sorting
    let concreteScore = 0;
    let abstractScore = 0;
    
    if (memory.who) concreteScore += 1.0;
    if (memory.what) concreteScore += 2.0;
    if (memory.when) concreteScore += 0.5;
    if (memory.where) concreteScore += 0.5;
    if (memory.why) abstractScore += 1.5;
    if (memory.how) abstractScore += 1.0;
    
    const total = concreteScore + abstractScore;
    if (total === 0) return 0;
    
    return (concreteScore / total) * 100;
}

function setupColumnResizing() {
    const table = document.getElementById('memoryTable');
    if (!table) return;
    
    const resizeHandles = table.querySelectorAll('.resize-handle');
    
    resizeHandles.forEach(handle => {
        let startX = 0;
        let startWidth = 0;
        let column = null;
        
        handle.addEventListener('mousedown', (e) => {
            e.stopPropagation(); // Prevent sorting when resizing
            startX = e.pageX;
            column = handle.parentElement;
            startWidth = column.offsetWidth;
            
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
        });
        
        function handleMouseMove(e) {
            if (!column) return;
            const diff = e.pageX - startX;
            const newWidth = Math.max(50, startWidth + diff); // Minimum 50px width
            column.style.width = newWidth + 'px';
            column.style.minWidth = newWidth + 'px';
            column.style.maxWidth = newWidth + 'px';
        }
        
        function handleMouseUp() {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            column = null;
        }
    });
}

function updateSortIndicators(field, direction) {
    // Reset all sort indicators
    document.querySelectorAll('.sortable').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
        const icon = th.querySelector('.sort-icon');
        if (icon) {
            icon.className = 'bi bi-arrow-down-up sort-icon';
        }
    });
    
    // Update active column indicator
    if (field && direction) {
        const activeHeader = document.querySelector(`.sortable[data-field="${field}"]`);
        if (activeHeader) {
            activeHeader.classList.add(`sort-${direction}`);
            const icon = activeHeader.querySelector('.sort-icon');
            if (icon) {
                if (direction === 'asc') {
                    icon.className = 'bi bi-arrow-up sort-icon';
                } else {
                    icon.className = 'bi bi-arrow-down sort-icon';
                }
            }
        }
    }
}

function showTypingIndicator() {
    const id = 'typing-' + Date.now();
    const messagesDiv = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.id = id;
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = `
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    messagesDiv.appendChild(typingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3`;
    notification.style.zIndex = '9999';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

function formatDate(dateString) {
    if (!dateString) return '—';
    const date = new Date(dateString);
    if (isNaN(date)) return dateString;
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Multi-dimensional merge functions
async function loadMergeDimensions() {
    try {
        const response = await fetch('/api/merge-dimensions');
        const data = await response.json();
        
        // Update dimension counts in UI
        data.dimensions.forEach(dim => {
            const countElement = document.getElementById(`${dim.type}-count`);
            if (countElement) {
                countElement.textContent = dim.group_count;
            }
        });
        
        return data.dimensions;
    } catch (error) {
        console.error('Error loading merge dimensions:', error);
        return [];
    }
}

async function switchMergeDimension(dimension) {
    currentMergeDimension = dimension;
    
    // Reset pagination when switching dimensions
    mergeGroupsCurrentPage = 1;
    
    // Update nav-link active classes
    document.querySelectorAll('.merge-dimension-selector .nav-link').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.dimension === dimension) {
            btn.classList.add('active');
        }
    });
    
    // Update description
    const descriptions = {
        'temporal': 'Groups conversation threads and sequential actions',
        'actor': 'Groups all memories from the same person or actor',
        'conceptual': 'Groups memories by goals, purposes, and abstract concepts',
        'spatial': 'Groups memories by location or spatial context'
    };
    const descElement = document.getElementById('dimension-description');
    if (descElement) {
        descElement.textContent = descriptions[dimension] || '';
    }
    
    // Load merge groups for this dimension
    await loadMergeGroups(dimension);
}

async function loadMergeGroups(dimension) {
    try {
        const response = await fetch(`/api/merge-groups/${dimension}`);
        const data = await response.json();
        
        // Cache the groups
        mergeGroups[dimension] = data.groups;
        
        // Display the merge groups
        displayMergeGroups(data.groups);
        
        return data.groups;
    } catch (error) {
        console.error('Error loading merge groups:', error);
        return [];
    }
}

function displayMergeGroups(groups) {
    // Hide card view, show table view
    const memoryList = document.getElementById('memoryList');
    const memoryTableContainer = document.getElementById('memoryTableContainer');
    
    if (memoryList) memoryList.style.display = 'none';
    if (memoryTableContainer) memoryTableContainer.style.display = 'block';
    
    // Store groups for pagination
    mergeGroupsFilteredData = groups;
    
    // Apply sorting if needed
    sortMergeGroups();
    
    // Display page with pagination
    displayMergeGroupsPage();
}

function displayMergeGroupsPage() {
    const tbody = document.getElementById('memoryTableBody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (mergeGroupsFilteredData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="text-center text-muted py-4">No merge groups found for this dimension</td></tr>';
        updateMergeGroupsPaginationInfo();
        createMergeGroupsPaginationControls();
        return;
    }
    
    // Calculate pagination
    const startIndex = (mergeGroupsCurrentPage - 1) * mergeGroupsItemsPerPage;
    const endIndex = Math.min(startIndex + mergeGroupsItemsPerPage, mergeGroupsFilteredData.length);
    const pageGroups = mergeGroupsFilteredData.slice(startIndex, endIndex);
    
    // Display merge groups in table format
    pageGroups.forEach(group => {
        const row = document.createElement('tr');
        const latest = group.latest_state || {};
        
        // Get the most recent values from the merged event
        const who = latest.who || group.key || '—';
        const what = latest.what || '—';
        const when = latest.when || group.last_updated || '—';
        const where = latest.where || '—';
        const why = latest.why || '—';
        const how = latest.how || '—';
        
        // Create merge indicator
        const mergeIndicator = `<span class="badge bg-success ms-1" title="${group.merge_count} merged events">
            ${group.merge_count}
        </span>`;
        
        // Create space weight indicator (use dominant pattern)
        const spaceWeight = group.dominant_pattern === 'abstract' ? 
            '<span class="badge bg-warning">Hyperbolic</span>' : 
            '<span class="badge bg-info">Euclidean</span>';
        
        row.innerHTML = `
            <td>${escapeHtml(who)}</td>
            <td title="${escapeHtml(what)}">${escapeHtml(what.substring(0, 40))}${what.length > 40 ? '...' : ''}${mergeIndicator}</td>
            <td>${formatDate(when)}</td>
            <td>${escapeHtml(where.substring(0, 20))}${where.length > 20 ? '...' : ''}</td>
            <td title="${escapeHtml(why)}">${escapeHtml(why.substring(0, 30))}${why.length > 30 ? '...' : ''}</td>
            <td title="${escapeHtml(how)}">${escapeHtml(how.substring(0, 25))}${how.length > 25 ? '...' : ''}</td>
            <td>${spaceWeight}</td>
            <td><span class="badge bg-secondary">Merged</span></td>
            <td>
                <button class="btn btn-sm btn-outline-info" onclick="window.viewMemoryDetails('${group.id}')" title="View Details">
                    <i class="bi bi-eye"></i>
                </button>
                <button class="btn btn-sm btn-outline-primary" onclick="window.showMemoryInGraph('${group.id}')" title="Show in Graph">
                    <i class="bi bi-diagram-3"></i>
                </button>
            </td>
        `;
        
        // Make row clickable
        row.style.cursor = 'pointer';
        row.title = 'Click to view details';
        row.addEventListener('click', (e) => {
            // Don't trigger if clicking on a button or within the actions cell
            if (!e.target.closest('button') && !e.target.closest('td:last-child')) {
                window.viewMemoryDetails(group.id);
            }
        });
        
        tbody.appendChild(row);
    });
    
    // Update pagination info and controls
    updateMergeGroupsPaginationInfo();
    createMergeGroupsPaginationControls();
}

function sortMergeGroups() {
    if (!mergeGroupsFilteredData || mergeGroupsFilteredData.length === 0) return;
    
    mergeGroupsFilteredData.sort((a, b) => {
        let aValue, bValue;
        
        // Get values based on field
        switch(mergeGroupsSortField) {
            case 'who':
                aValue = (a.latest_state?.who || a.key || '').toLowerCase();
                bValue = (b.latest_state?.who || b.key || '').toLowerCase();
                break;
            case 'what':
                aValue = (a.latest_state?.what || '').toLowerCase();
                bValue = (b.latest_state?.what || '').toLowerCase();
                break;
            case 'when':
                aValue = a.last_updated || '';
                bValue = b.last_updated || '';
                break;
            case 'where':
                aValue = (a.latest_state?.where || '').toLowerCase();
                bValue = (b.latest_state?.where || '').toLowerCase();
                break;
            case 'why':
                aValue = (a.latest_state?.why || '').toLowerCase();
                bValue = (b.latest_state?.why || '').toLowerCase();
                break;
            case 'how':
                aValue = (a.latest_state?.how || '').toLowerCase();
                bValue = (b.latest_state?.how || '').toLowerCase();
                break;
            case 'spaceWeight':
                aValue = a.dominant_pattern === 'abstract' ? 1 : 0;
                bValue = b.dominant_pattern === 'abstract' ? 1 : 0;
                break;
            default:
                aValue = 0;
                bValue = 0;
        }
        
        // Compare values
        if (aValue < bValue) return mergeGroupsSortDirection === 'asc' ? -1 : 1;
        if (aValue > bValue) return mergeGroupsSortDirection === 'asc' ? 1 : -1;
        return 0;
    });
}

function updateMergeGroupsPaginationInfo() {
    const totalItems = mergeGroupsFilteredData.length;
    const startIndex = Math.min((mergeGroupsCurrentPage - 1) * mergeGroupsItemsPerPage + 1, totalItems);
    const endIndex = Math.min(mergeGroupsCurrentPage * mergeGroupsItemsPerPage, totalItems);
    
    // Update pagination info displays
    const showingStart = document.getElementById('showingStart');
    const showingEnd = document.getElementById('showingEnd');
    const totalMemoriesInfo = document.getElementById('totalMemoriesInfo');
    const totalMemoriesCount = document.getElementById('totalMemories');
    
    if (showingStart) showingStart.textContent = totalItems > 0 ? startIndex : 0;
    if (showingEnd) showingEnd.textContent = endIndex;
    if (totalMemoriesInfo) totalMemoriesInfo.textContent = totalItems;
    if (totalMemoriesCount) totalMemoriesCount.textContent = totalItems;
}

function createMergeGroupsPaginationControls() {
    const paginationControls = document.getElementById('memoryPagination');
    if (!paginationControls) return;
    
    paginationControls.innerHTML = '';
    
    const totalPages = Math.ceil(mergeGroupsFilteredData.length / mergeGroupsItemsPerPage);
    
    if (totalPages <= 1) {
        paginationControls.style.display = 'none';
        return;
    }
    
    paginationControls.style.display = 'flex';
    
    // Previous button
    const prevLi = document.createElement('li');
    prevLi.className = `page-item ${mergeGroupsCurrentPage === 1 ? 'disabled' : ''}`;
    prevLi.innerHTML = `
        <a class="page-link" href="#" onclick="changeMergeGroupsPage(${mergeGroupsCurrentPage - 1}); return false;">
            <i class="bi bi-chevron-left"></i>
        </a>
    `;
    paginationControls.appendChild(prevLi);
    
    // Page numbers with ellipsis
    const maxVisible = 5;
    let startPage = Math.max(1, mergeGroupsCurrentPage - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);
    
    if (endPage - startPage < maxVisible - 1) {
        startPage = Math.max(1, endPage - maxVisible + 1);
    }
    
    // First page + ellipsis
    if (startPage > 1) {
        const firstLi = document.createElement('li');
        firstLi.className = 'page-item';
        firstLi.innerHTML = `<a class="page-link" href="#" onclick="changeMergeGroupsPage(1); return false;">1</a>`;
        paginationControls.appendChild(firstLi);
        
        if (startPage > 2) {
            const ellipsisLi = document.createElement('li');
            ellipsisLi.className = 'page-item disabled';
            ellipsisLi.innerHTML = '<span class="page-link">...</span>';
            paginationControls.appendChild(ellipsisLi);
        }
    }
    
    // Visible page numbers
    for (let i = startPage; i <= endPage; i++) {
        const li = document.createElement('li');
        li.className = `page-item ${i === mergeGroupsCurrentPage ? 'active' : ''}`;
        li.innerHTML = `
            <a class="page-link" href="#" onclick="changeMergeGroupsPage(${i}); return false;">${i}</a>
        `;
        paginationControls.appendChild(li);
    }
    
    // Last page + ellipsis
    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            const ellipsisLi = document.createElement('li');
            ellipsisLi.className = 'page-item disabled';
            ellipsisLi.innerHTML = '<span class="page-link">...</span>';
            paginationControls.appendChild(ellipsisLi);
        }
        
        const lastLi = document.createElement('li');
        lastLi.className = 'page-item';
        lastLi.innerHTML = `<a class="page-link" href="#" onclick="changeMergeGroupsPage(${totalPages}); return false;">${totalPages}</a>`;
        paginationControls.appendChild(lastLi);
    }
    
    // Next button
    const nextLi = document.createElement('li');
    nextLi.className = `page-item ${mergeGroupsCurrentPage === totalPages ? 'disabled' : ''}`;
    nextLi.innerHTML = `
        <a class="page-link" href="#" onclick="changeMergeGroupsPage(${mergeGroupsCurrentPage + 1}); return false;">
            <i class="bi bi-chevron-right"></i>
        </a>
    `;
    paginationControls.appendChild(nextLi);
}

window.changeMergeGroupsPage = function(page) {
    const totalPages = Math.ceil(mergeGroupsFilteredData.length / mergeGroupsItemsPerPage);
    
    if (page < 1 || page > totalPages) return;
    
    mergeGroupsCurrentPage = page;
    displayMergeGroupsPage();
}

function createMergeGroupCard(group) {
    const card = document.createElement('div');
    card.className = 'memory-card mb-3';
    
    const primaryField = getPrimaryFieldForDimension(group.type);
    const primaryValue = group.latest_state[primaryField] || group.key || 'Unknown';
    
    card.innerHTML = `
        <div class="card">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <h5 class="card-title">${escapeHtml(primaryValue)}</h5>
                        <div class="text-muted small">
                            <span class="badge bg-info me-2">${group.merge_count} merged events</span>
                            <span>Last updated: ${formatDate(group.last_updated)}</span>
                        </div>
                    </div>
                    <div>
                        <button class="btn btn-sm btn-outline-primary" onclick="viewMergeGroupDetail('${group.type}', '${group.id}')">
                            <i class="bi bi-eye"></i> View Details
                        </button>
                    </div>
                </div>
                <div class="mt-2">
                    ${renderGroupSummary(group)}
                </div>
            </div>
        </div>
    `;
    
    return card;
}

function getPrimaryFieldForDimension(dimension) {
    const fields = {
        'actor': 'who',
        'temporal': 'what',
        'conceptual': 'why',
        'spatial': 'where'
    };
    return fields[dimension] || 'what';
}

function renderGroupSummary(group) {
    const state = group.latest_state;
    let summary = '<div class="small">';
    
    if (state.who) summary += `<div><strong>Who:</strong> ${escapeHtml(state.who)}</div>`;
    if (state.what) summary += `<div><strong>What:</strong> ${escapeHtml(state.what.substring(0, 100))}${state.what.length > 100 ? '...' : ''}</div>`;
    if (state.why) summary += `<div><strong>Why:</strong> ${escapeHtml(state.why)}</div>`;
    if (state.where) summary += `<div><strong>Where:</strong> ${escapeHtml(state.where)}</div>`;
    
    summary += '</div>';
    return summary;
}

async function viewMergeGroupDetail(mergeType, groupId) {
    try {
        const response = await fetch(`/api/merge-group/${mergeType}/${groupId}`);
        const data = await response.json();
        
        // Show detail modal or navigate to detail view
        showMergeGroupDetailModal(data);
    } catch (error) {
        console.error('Error loading merge group detail:', error);
    }
}

function showMergeGroupDetailModal(mergeGroup) {
    // Create or update modal for merge group details
    let modal = document.getElementById('mergeGroupDetailModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'mergeGroupDetailModal';
        modal.innerHTML = `
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Merge Group Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body" id="mergeGroupDetailBody">
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }
    
    const modalBody = document.getElementById('mergeGroupDetailBody');
    modalBody.innerHTML = renderMergeGroupDetail(mergeGroup);
    
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
}

function renderMergeGroupDetail(group) {
    let html = `
        <div class="merge-group-detail">
            <div class="row mb-3">
                <div class="col-md-6">
                    <h6>Basic Information</h6>
                    <div class="small">
                        <div><strong>Type:</strong> ${group.type}</div>
                        <div><strong>ID:</strong> ${group.id}</div>
                        <div><strong>Merge Count:</strong> ${group.merge_count}</div>
                        <div><strong>Created:</strong> ${formatDate(group.created_at)}</div>
                        <div><strong>Last Updated:</strong> ${formatDate(group.last_updated)}</div>
                    </div>
                </div>
                <div class="col-md-6">
                    <h6>Latest State</h6>
                    <div class="small">
                        ${renderGroupSummary({latest_state: group.latest_state})}
                    </div>
                </div>
            </div>
    `;
    
    // Component variations
    if (Object.keys(group.who_variants || {}).length > 0) {
        html += renderComponentVariations('Who', group.who_variants);
    }
    if (Object.keys(group.what_variants || {}).length > 0) {
        html += renderComponentVariations('What', group.what_variants);
    }
    if (Object.keys(group.why_variants || {}).length > 0) {
        html += renderComponentVariations('Why', group.why_variants);
    }
    
    // Timeline
    if (group.when_timeline && group.when_timeline.length > 0) {
        html += `
            <div class="mb-3">
                <h6>Timeline</h6>
                <div class="timeline small">
        `;
        group.when_timeline.forEach(point => {
            html += `
                <div class="timeline-item">
                    <span class="badge bg-secondary me-2">${formatDate(point.timestamp)}</span>
                    ${escapeHtml(point.description || point.semantic_time || '')}
                </div>
            `;
        });
        html += '</div></div>';
    }
    
    // Raw events
    if (group.raw_event_ids && group.raw_event_ids.length > 0) {
        html += `
            <div class="mb-3">
                <h6>Raw Events (${group.raw_event_ids.length})</h6>
                <div class="small text-muted">
                    ${group.raw_event_ids.slice(0, 5).join(', ')}
                    ${group.raw_event_ids.length > 5 ? '...' : ''}
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    return html;
}

function renderComponentVariations(componentName, variants) {
    let html = `
        <div class="mb-3">
            <h6>${componentName} Variations</h6>
            <div class="variations small">
    `;
    
    for (const [key, variantList] of Object.entries(variants)) {
        variantList.forEach(variant => {
            html += `
                <div class="variant-item mb-1">
                    <span class="badge bg-secondary me-1">${variant.relationship}</span>
                    <span class="badge bg-info me-1">v${variant.version}</span>
                    ${escapeHtml(variant.value)}
                    <span class="text-muted ms-2">(${formatDate(variant.timestamp)})</span>
                </div>
            `;
        });
    }
    
    html += '</div></div>';
    return html;
}

// Update the existing loadMemories function to check for merge dimension mode
window.loadMemories = async function() {
    // Check current view mode
    const isRawView = document.getElementById('rawView') && document.getElementById('rawView').checked;
    const isMergedView = document.getElementById('mergedView') && document.getElementById('mergedView').checked;
    
    // Get display elements
    const memoryList = document.getElementById('memoryList');
    const memoryTableContainer = document.getElementById('memoryTableContainer');
    
    // Load merge dimensions (for counts)
    await loadMergeDimensions();
    
    if (isRawView) {
        // Raw view - show table, hide cards
        if (memoryList) memoryList.style.display = 'none';
        if (memoryTableContainer) memoryTableContainer.style.display = 'block';
        
        // Ensure pagination element is visible (may have been hidden by merge view)
        const paginationElement = document.getElementById('memoryPagination');
        if (paginationElement) {
            paginationElement.style.display = '';
        }
        
        await loadMemoriesOriginal();
    } else if (isMergedView) {
        // Merged view - show table (will be handled by displayMergeGroups)
        // Default to temporal if not set
        if (!currentMergeDimension) {
            currentMergeDimension = 'temporal';
        }
        await loadMergeGroups(currentMergeDimension);
    }
}

// Export functions for global access
window.switchMergeDimension = switchMergeDimension;
window.viewMergeGroupDetail = viewMergeGroupDetail;
