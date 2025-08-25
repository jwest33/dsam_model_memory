// Main application JavaScript

let allMemories = { raw: [], processed: [], blocks: [] };
let currentMemoryId = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadMemories();
    loadStats();
    
    // Set up event listeners
    document.getElementById('chatForm').addEventListener('submit', handleChatSubmit);
    if (document.getElementById('createMemoryForm')) {
        document.getElementById('createMemoryForm').addEventListener('submit', handleMemoryCreate);
    }
    if (document.getElementById('searchForm')) {
        document.getElementById('searchForm').addEventListener('submit', handleSearch);
    }
    
    // Add tab change listener to refresh memories when memory tab is selected
    const memoryTab = document.getElementById('memory-tab');
    if (memoryTab) {
        memoryTab.addEventListener('click', function (e) {
            // Only refresh if user hasn't interacted recently (within 5 seconds)
            const timeSinceInteraction = Date.now() - (window.lastUserInteraction || 0);
            if (timeSinceInteraction > 5000) {
                console.log('Memory tab activated, refreshing memories...');
                setTimeout(() => {
                    loadMemories(true); // Show loading indicator
                    loadStats();
                }, 100); // Small delay to ensure tab is shown
            } else {
                console.log('Memory tab activated, skipping refresh due to recent interaction');
            }
        });
    }
    
    // Bootstrap 5 tab shown event
    const memoryTabEl = document.querySelector('button[data-bs-target="#memory"]');
    if (memoryTabEl) {
        memoryTabEl.addEventListener('shown.bs.tab', function (event) {
            // Only refresh if user hasn't interacted recently (within 5 seconds)
            const timeSinceInteraction = Date.now() - (window.lastUserInteraction || 0);
            if (timeSinceInteraction > 5000) {
                console.log('Memory tab shown, refreshing...');
                loadMemories();
                loadStats();
            }
        });
    }
    
    // Also listen for sub-tab changes within memory tab
    document.querySelectorAll('#memoryTypeTabs button').forEach(tab => {
        tab.addEventListener('click', function(e) {
            // Refresh when switching between raw/processed/blocks
            if (Date.now() - window.lastMemoryLoad > 2000) { // Only if more than 2 seconds passed
                setTimeout(() => {
                    loadMemories();
                }, 100);
            }
        });
    });
    
    // Auto-refresh stats every 30 seconds
    setInterval(loadStats, 30000);
});

// Track last memory load time
window.lastMemoryLoad = 0;

// Chat functionality
async function handleChatSubmit(e) {
    e.preventDefault();
    
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
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
        
        if (response.ok) {
            // Add AI response to chat
            addChatMessage(data.response, 'assistant');
            
            // No longer showing memories used indicator
            
            // Refresh memories in the background since new ones were created
            setTimeout(() => {
                loadMemories();
                loadStats();
            }, 500); // Small delay to ensure server has processed everything
        } else {
            addChatMessage('Error: ' + (data.error || 'Failed to get response'), 'error');
        }
    } catch (error) {
        removeTypingIndicator(typingId);
        addChatMessage('Error: ' + error.message, 'error');
    } finally {
        input.disabled = false;
        input.focus();
    }
}

function addChatMessage(message, type) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}-message mb-3`;
    
    const iconClass = type === 'user' ? 'bi-person-circle' : 
                     type === 'assistant' ? 'bi-robot' : 'bi-exclamation-circle';
    const bgClass = type === 'user' ? 'bg-primary' : 
                   type === 'assistant' ? 'bg-success' : 'bg-danger';
    
    messageDiv.innerHTML = `
        <div class="d-flex ${type === 'user' ? 'justify-content-end' : ''}">
            <div class="message-content ${bgClass} text-white rounded p-3" style="max-width: 70%;">
                <div class="d-flex align-items-center justify-content-between mb-2">
                    <div class="d-flex align-items-center">
                        <i class="bi ${iconClass} me-2"></i>
                        <small>${type === 'user' ? 'You' : type === 'assistant' ? 'AI Assistant' : 'System'}</small>
                    </div>
                    <small class="ms-3">${new Date().toLocaleTimeString()}</small>
                </div>
                <div>${escapeHtml(message)}</div>
            </div>
        </div>
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showTypingIndicator() {
    const id = 'typing-' + Date.now();
    const messagesDiv = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.id = id;
    typingDiv.className = 'typing-indicator mb-3';
    typingDiv.innerHTML = `
        <div class="d-flex">
            <div class="bg-light rounded p-3">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
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

// currentMemoryId already declared at the top of the file

// Function to show node details in the graph modal
function showNodeDetails(nodeId) {
    console.log('Showing details for node:', nodeId);
    
    // Find the node in the current graph data
    const node = window.currentGraphNodes ? window.currentGraphNodes.find(n => n.id === nodeId) : null;
    
    if (!node) {
        console.error('Node not found:', nodeId);
        return;
    }
    
    // Find the full memory data
    const memory = [...allMemories.raw, ...allMemories.processed].find(m => m.id === nodeId);
    
    if (!memory) {
        console.error('Memory data not found for node:', nodeId);
        return;
    }
    
    // Update the node details in the unified panel
    const detailsContent = document.getElementById('nodeDetailsContent');
    
    // Build compact 5W1H details
    detailsContent.innerHTML = `
        <div class="mb-2"><strong class="text-purple">Who:</strong> ${escapeHtml(memory.who || 'N/A')}</div>
        <div class="mb-2"><strong class="text-purple">What:</strong> ${escapeHtml(memory.what || 'N/A')}</div>
        <div class="mb-2"><strong class="text-purple">When:</strong> ${memory.when ? new Date(memory.when).toLocaleString() : 'N/A'}</div>
        <div class="mb-2"><strong class="text-purple">Where:</strong> ${escapeHtml(memory.where || 'N/A')}</div>
        <div class="mb-2"><strong class="text-purple">Why:</strong> ${escapeHtml(memory.why || 'N/A')}</div>
        <div class="mb-2"><strong class="text-purple">How:</strong> ${escapeHtml(memory.how || 'N/A')}</div>
        <div class="mb-2"><strong class="text-purple">Type:</strong> <span class="badge bg-purple">${memory.type || 'N/A'}</span></div>
        ${memory.episode_id ? `<div class="mb-2"><strong class="text-purple">Episode:</strong> <small>${memory.episode_id}</small></div>` : ''}
    `;
}

// Graph visualization
async function showMemoryGraph(memoryId) {
    console.log('showMemoryGraph called with memoryId:', memoryId);
    if (!memoryId) {
        console.error('No memory ID provided to showMemoryGraph');
        alert('Please select a memory first');
        return;
    }
    currentMemoryId = memoryId;
    
    // Show the graph modal
    const graphModal = new bootstrap.Modal(document.getElementById('memoryGraphModal'));
    graphModal.show();
    
    // Load the graph with default settings
    loadClusterGraph(memoryId);
}

async function loadClusterGraph(memoryId, mode = 'default', components = []) {
    // Show loading and reset panels
    document.getElementById('graphInfo').innerHTML = '<i class="spinner-border spinner-border-sm"></i> Loading...';
    document.getElementById('memoryGraph').innerHTML = '';
    document.getElementById('nodeDetailsContent').innerHTML = '<em class="text-muted">Click a node to view details</em>';
    document.getElementById('edgeDetails').style.display = 'none';
    document.getElementById('edgeDivider').style.display = 'none';
    
    try {
        // Build URL with parameters
        let url = `/api/memory/${memoryId}/cluster?mode=${mode}`;
        if (components.length > 0) {
            components.forEach(comp => {
                url += `&components=${comp}`;
            });
        }
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.success) {
            // Display info with mode details
            let modeInfo = '';
            if (data.mode === 'single') {
                modeInfo = `Single Component (${data.components ? data.components.join(', ') : 'default'})`;
            } else if (data.mode === 'combination') {
                modeInfo = `Combination (${data.components ? data.components.join(' + ') : 'default'})`;
            } else {
                modeInfo = 'Default Weighted';
            }
            
            // Format query for display
            let queryDisplay = [];
            for (const [key, value] of Object.entries(data.query)) {
                if (value) {
                    queryDisplay.push(`<span class="text-purple">${key}:</span> "${value.substring(0, 30)}${value.length > 30 ? '...' : ''}"`);
                }
            }
            
            document.getElementById('graphInfo').innerHTML = `
                <div><strong>Mode:</strong> ${modeInfo}</div>
                <div><strong>Query:</strong> ${queryDisplay.join(', ')}</div>
                <div><strong>Nodes:</strong> ${data.nodes.length} memories</div>
                <div><strong>Edges:</strong> ${data.edges.length} connections</div>
            `;
            
            // Clear previous details
            document.getElementById('nodeDetailsContent').innerHTML = '<em class="text-muted">Click a node to view details</em>';
            document.getElementById('edgeDetails').style.display = 'none';
            document.getElementById('edgeDivider').style.display = 'none';
            
            // Create the graph with enhanced node sizing
            const container = document.getElementById('memoryGraph');
            
            // Process nodes to adjust sizes - primary node should be smaller
            const processedNodes = data.nodes.map(node => {
                if (node.group === 'target') {
                    // Reduce primary node size significantly
                    node.size = 20; // Reduced from 30
                    node.font = { ...node.font, size: 14, bold: true };
                    node.borderWidth = 3;
                    node.shadow = {
                        enabled: true,
                        color: 'rgba(255, 16, 240, 0.8)',
                        size: 15
                    };
                } else {
                    // Scale other nodes based on relevance
                    node.size = Math.max(12, Math.min(25, node.size));
                    node.font = { ...node.font, size: 10 };
                }
                
                // Remove title/tooltip - we don't want hover tooltips
                node.title = undefined;
                
                return node;
            });
            
            const graphData = {
                nodes: new vis.DataSet(processedNodes),
                edges: new vis.DataSet(data.edges)
            };
            
            // Enhanced options with better interactivity and visual hierarchy
            const options = {
                nodes: {
                    shape: 'dot',
                    scaling: {
                        min: 10,
                        max: 30,
                        label: {
                            enabled: true,
                            min: 10,
                            max: 16
                        }
                    },
                    font: {
                        size: 12,
                        color: '#00ffff',
                        strokeWidth: 2,
                        strokeColor: 'rgba(20, 16, 31, 0.8)'
                    },
                    color: {
                        background: '#ff10f0',
                        border: '#00ffff',
                        highlight: {
                            background: '#00ffff',
                            border: '#ff10f0'
                        },
                        hover: {
                            background: '#ff40ff',
                            border: '#40ffff'
                        }
                    },
                    borderWidth: 2,
                    borderWidthSelected: 3,
                    shadow: {
                        enabled: true,
                        color: 'rgba(255, 16, 240, 0.5)',
                        size: 10,
                        x: 0,
                        y: 0
                    },
                    chosen: {
                        node: function(values, id, selected, hovering) {
                            // Only respond to selection, not hovering
                            if (selected) {
                                values.borderWidth = 4;
                                values.shadow = {
                                    enabled: true,
                                    color: 'rgba(0, 255, 255, 0.8)',
                                    size: 20
                                };
                            }
                        },
                        label: function(values, id, selected, hovering) {
                            if (selected) {
                                values.color = '#ffffff';
                            }
                        }
                    }
                },
                edges: {
                    font: {
                        size: 10,
                        align: 'horizontal',
                        color: 'rgba(0, 255, 255, 0.8)',
                        strokeWidth: 2,
                        strokeColor: 'rgba(20, 16, 31, 0.9)',
                        background: 'rgba(20, 16, 31, 0.9)'
                    },
                    arrows: {
                        to: {
                            enabled: true,
                            scaleFactor: 0.3
                        }
                    },
                    color: {
                        color: 'rgba(168, 85, 247, 0.5)',
                        highlight: '#00ffff',
                        opacity: 0.6
                    },
                    width: 0.5,
                    hoverWidth: 2,
                    selectionWidth: 2,
                    smooth: {
                        enabled: true,
                        type: 'curvedCW',
                        roundness: 0.2
                    },
                    scaling: {
                        min: 0.5,
                        max: 2
                    },
                    chosen: {
                        edge: function(values, id, selected, hovering) {
                            if (selected || hovering) {
                                values.width = 2;
                                values.color = '#00ffff';
                            }
                        }
                    },
                    title: undefined  // No tooltips on edges
                },
                physics: {
                    enabled: true,
                    stabilization: {
                        enabled: true,
                        iterations: 150,
                        updateInterval: 50,
                        fit: true
                    },
                    barnesHut: {
                        theta: 0.5,
                        gravitationalConstant: -12000,
                        centralGravity: 0.2,
                        springLength: 200,
                        springConstant: 0.01,
                        damping: 0.95,
                        avoidOverlap: 1
                    },
                    maxVelocity: 50,
                    minVelocity: 0.1,
                    solver: 'barnesHut'
                },
                layout: {
                    improvedLayout: true,
                    hierarchical: false
                },
                interaction: {
                    hover: false,  // Disable hover tooltips
                    tooltipDelay: 300000,  // Effectively disable tooltips
                    hideEdgesOnDrag: true,
                    navigationButtons: false,  // Disable navigation buttons
                    keyboard: {
                        enabled: true,
                        speed: { x: 10, y: 10, zoom: 0.02 },
                        bindToWindow: false
                    },
                    zoomView: true,
                    dragView: true,
                    multiselect: true,
                    selectable: true
                }
            };
            
            const network = new vis.Network(container, graphData, options);
            
            // Stop physics after stabilization to prevent continuous movement
            network.on("stabilizationIterationsDone", function () {
                network.setOptions({ physics: { enabled: false } });
                console.log("Graph stabilized, physics disabled");
                
                // Find the primary/target node (first node or node with group='target')
                const targetNode = nodes.find(n => n.group === 'target');
                if (targetNode) {
                    // Focus on the primary node
                    network.focus(targetNode.id, {
                        scale: 1.0,
                        animation: {
                            duration: 500,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                } else {
                    // Fallback to fit all nodes
                    network.fit({
                        animation: {
                            duration: 500,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                }
            });
            
            // Allow manual physics toggle
            window.toggleGraphPhysics = function() {
                const currentPhysics = network.physics.options.enabled;
                network.setOptions({ physics: { enabled: !currentPhysics } });
                console.log("Physics toggled to:", !currentPhysics);
            };
            
            // Store nodes data for details display
            window.currentGraphNodes = data.nodes;
            
            // Enhanced interaction handlers
            network.on("click", function(params) {
                // Handle node clicks
                if (params.nodes.length > 0) {
                    const nodeId = params.nodes[0];
                    showNodeDetails(nodeId);
                    
                    // Highlight connected nodes
                    const connectedNodes = network.getConnectedNodes(nodeId);
                    const allNodes = processedNodes.map(n => n.id);
                    const updateNodes = [];
                    
                    allNodes.forEach(id => {
                        if (id === nodeId) {
                            updateNodes.push({ id, color: { background: '#00ffff', border: '#ff10f0' } });
                        } else if (connectedNodes.includes(id)) {
                            updateNodes.push({ id, color: { background: '#ff40ff', border: '#00ffff' } });
                        } else {
                            updateNodes.push({ id, color: { background: '#ff10f0', border: '#00ffff' } });
                        }
                    });
                    
                    graphData.nodes.update(updateNodes);
                }
                
                // Handle edge clicks - show relation stats
                if (params.edges.length > 0 && params.nodes.length === 0) {
                    const edgeId = params.edges[0];
                    const edge = graphData.edges.get(edgeId);
                    
                    if (edge) {
                        // Find the nodes connected by this edge
                        const fromNode = processedNodes.find(n => n.id === edge.from);
                        const toNode = processedNodes.find(n => n.id === edge.to);
                        
                        if (fromNode && toNode) {
                            // Show edge details section
                            document.getElementById('edgeDetails').style.display = 'block';
                            document.getElementById('edgeDivider').style.display = 'block';
                            
                            // Display relation stats in the unified panel
                            document.getElementById('edgeDetailsContent').innerHTML = `
                                <div class="mb-2"><strong class="text-purple">From:</strong> ${fromNode.label}</div>
                                <div class="mb-2"><strong class="text-purple">To:</strong> ${toNode.label}</div>
                                <div class="mb-2"><strong class="text-purple">Score:</strong> ${edge.value ? edge.value.toFixed(3) : 'N/A'}</div>
                                <div class="mb-2"><strong class="text-purple">Type:</strong> <span class="badge ${edge.dashes ? 'bg-info' : 'bg-success'}">${edge.dashes ? 'Episode Link' : 'Similarity'}</span></div>
                            `;
                        }
                    }
                }
            });
            
            // Double-click to focus on node
            network.on("doubleClick", function(params) {
                if (params.nodes.length > 0) {
                    network.focus(params.nodes[0], {
                        scale: 1.5,
                        animation: {
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                }
            });
            
            // Right-click context menu
            network.on("oncontext", function(params) {
                params.event.preventDefault();
                if (params.nodes.length > 0) {
                    const nodeId = params.nodes[0];
                    const node = processedNodes.find(n => n.id === nodeId);
                    if (node) {
                        // Could add a context menu here
                        console.log('Right-clicked on node:', node.label);
                    }
                }
            });
        } else {
            document.getElementById('graphInfo').innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
        }
    } catch (error) {
        console.error('Error loading graph:', error);
        document.getElementById('graphInfo').innerHTML = `<div class="alert alert-danger">Failed to load cluster graph</div>`;
    }
}

// Refresh cluster graph with selected components
function refreshClusterGraph() {
    if (!currentMemoryId) {
        alert('No memory selected');
        return;
    }
    
    const mode = document.getElementById('clusteringMode').value;
    const components = [];
    
    // Get selected components
    ['who', 'what', 'when', 'where', 'why', 'how'].forEach(comp => {
        const checkbox = document.getElementById(`comp-${comp}`);
        if (checkbox && checkbox.checked) {
            components.push(comp);
        }
    });
    
    if (mode !== 'default' && components.length === 0) {
        alert('Please select at least one component for this mode');
        return;
    }
    
    // Reload graph with new settings
    loadClusterGraph(currentMemoryId, mode, components);
}

// Show Create Memory Modal
function showCreateMemoryModal() {
    const modal = new bootstrap.Modal(document.getElementById('createMemoryModal'));
    modal.show();
}

// Create memory from modal
async function createMemory() {
    const memory = {
        who: document.getElementById('modalWho').value,
        what: document.getElementById('modalWhat').value,
        where: document.getElementById('modalWhere').value || 'web_interface',
        why: document.getElementById('modalWhy').value || 'Manual entry',
        how: document.getElementById('modalHow').value || 'Direct input',
        type: document.getElementById('modalType').value
    };
    
    try {
        const response = await fetch('/api/memories', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(memory)
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Close modal
            bootstrap.Modal.getInstance(document.getElementById('createMemoryModal')).hide();
            // Clear form
            document.getElementById('createMemoryForm').reset();
            // Reload memories
            loadMemories();
            // Show success
            showNotification('Memory created successfully', 'success');
        } else {
            showNotification(data.message || 'Failed to create memory', 'error');
        }
    } catch (error) {
        console.error('Error creating memory:', error);
        showNotification('Error creating memory', 'error');
    }
}

// Show notification
function showNotification(message, type = 'info') {
    // Simple notification - could be enhanced with toast or other UI
    console.log(`[${type}] ${message}`);
}

// Memory management functionality
async function loadMemories(showLoading = false) {
    // Show loading indicator if requested
    if (showLoading) {
        const tables = ['processedMemoriesTable', 'rawMemoriesTable', 'memoryBlocksTable'];
        tables.forEach(tableId => {
            const table = document.getElementById(tableId);
            if (table) {
                table.innerHTML = '<tr><td colspan="7" class="text-center"><div class="spinner-border spinner-border-sm" role="status"></div> Loading...</td></tr>';
            }
        });
    }
    
    try {
        const response = await fetch('/api/memories');
        const data = await response.json();
        
        console.log('Loaded memory data:', data); // Debug log
        
        if (response.ok) {
            allMemories = data;
            displayMemories();
            updateCounts();
            displayMemoryBlocks();
            window.lastMemoryLoad = Date.now(); // Track when we last loaded
            
            // Show success indicator briefly
            if (showLoading) {
                const badge = document.createElement('span');
                badge.className = 'badge bg-success position-fixed top-0 end-0 m-3';
                badge.textContent = 'Memories refreshed';
                document.body.appendChild(badge);
                setTimeout(() => badge.remove(), 2000);
            }
        } else {
            console.error('API returned error:', data);
            showAlert('Failed to load memories: ' + (data.error || 'Unknown error'), 'danger');
        }
    } catch (error) {
        console.error('Failed to load memories:', error);
        if (showLoading) {
            showAlert('Failed to load memories', 'danger');
        }
    }
}

function displayMemories() {
    // Display processed memories with block info
    const processedTable = document.getElementById('processedMemoriesTable');
    processedTable.innerHTML = '';
    
    if (allMemories.processed && Array.isArray(allMemories.processed)) {
        allMemories.processed.forEach(memory => {
            processedTable.appendChild(createMemoryRow(memory, false));  // No block info in new system
        });
    } else {
        processedTable.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No processed memories</td></tr>';
    }
    
    // Display raw memories
    const rawTable = document.getElementById('rawMemoriesTable');
    rawTable.innerHTML = '';
    
    if (allMemories.raw && Array.isArray(allMemories.raw)) {
        allMemories.raw.forEach(memory => {
            rawTable.appendChild(createMemoryRow(memory, false));
        });
    } else {
        rawTable.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No raw memories</td></tr>';
    }
}

function displayMemoryBlocks() {
    const blocksTable = document.getElementById('memoryBlocksTable');
    if (!blocksTable) return;
    
    blocksTable.innerHTML = '';
    
    if (!allMemories.blocks) return;
    
    allMemories.blocks.forEach(block => {
        const tr = document.createElement('tr');
        
        const salience = block.salience || 0;
        const coherence = block.coherence || 0;
        
        tr.innerHTML = `
            <td><small>${block.id.substring(0, 12)}...</small></td>
            <td><span class="badge bg-primary">${block.type}</span></td>
            <td>${block.event_count}</td>
            <td>
                <span class="badge ${salience >= 0.7 ? 'bg-success' : salience >= 0.3 ? 'bg-warning' : 'bg-secondary'}">
                    ${salience.toFixed(2)}
                </span>
            </td>
            <td>
                <div class="progress" style="width: 100px;">
                    <div class="progress-bar ${coherence >= 0.7 ? 'bg-success' : coherence >= 0.4 ? 'bg-warning' : 'bg-danger'}" 
                         style="width: ${coherence * 100}%">
                        ${coherence.toFixed(2)}
                    </div>
                </div>
            </td>
            <td>${block.link_count}</td>
            <td>
                <button class="btn btn-sm btn-info" onclick="viewBlock('${block.id}')" title="View Details">
                    <i class="bi bi-eye"></i>
                </button>
                <button class="btn btn-sm btn-danger ms-1" onclick="deleteBlock('${block.id}')" title="Delete Block">
                    <i class="bi bi-trash"></i>
                </button>
            </td>
        `;
        
        blocksTable.appendChild(tr);
    });
}

async function viewBlock(blockId) {
    const block = allMemories.blocks.find(b => b.id === blockId);
    if (!block) return;
    
    // Get the events that belong to this block
    const blockEvents = [];
    if (block.event_ids && block.event_ids.length > 0) {
        // Find events from both processed and raw memories
        const allEvents = [...(allMemories.processed || []), ...(allMemories.raw || [])];
        block.event_ids.forEach(eventId => {
            const event = allEvents.find(e => e.id === eventId);
            if (event) {
                blockEvents.push(event);
            }
        });
    }
    
    // Build the modal content
    const modalContent = document.getElementById('memoryBlockContent');
    
    let html = `
        <div class="container-fluid">
            <!-- Block Overview -->
            <div class="row mb-4">
                <div class="col-md-12">
                    <div class="card border-primary">
                        <div class="card-header bg-primary text-white d-flex justify-content-between align-items-center">
                            <h6 class="mb-0">Block Overview</h6>
                            <button class="btn btn-sm btn-danger" onclick="deleteBlock('${block.id}')">
                                <i class="bi bi-trash"></i> Delete Block
                            </button>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-3">
                                    <strong>Block ID:</strong><br>
                                    <small class="text-muted">${block.id}</small>
                                </div>
                                <div class="col-md-2">
                                    <strong>Type:</strong><br>
                                    <span class="badge bg-info">${block.type}</span>
                                </div>
                                <div class="col-md-2">
                                    <strong>Events:</strong><br>
                                    <span class="badge bg-secondary">${block.event_count}</span>
                                </div>
                                <div class="col-md-2">
                                    <strong>Links:</strong><br>
                                    <span class="badge bg-secondary">${block.link_count}</span>
                                </div>
                                <div class="col-md-1">
                                    <strong>Salience:</strong><br>
                                    <span class="badge ${block.salience >= 0.7 ? 'bg-success' : block.salience >= 0.3 ? 'bg-warning' : 'bg-secondary'}">
                                        ${block.salience.toFixed(2)}
                                    </span>
                                </div>
                                <div class="col-md-2">
                                    <strong>Coherence:</strong><br>
                                    <div class="progress" style="height: 20px;">
                                        <div class="progress-bar ${block.coherence >= 0.7 ? 'bg-success' : block.coherence >= 0.4 ? 'bg-warning' : 'bg-danger'}" 
                                             style="width: ${block.coherence * 100}%">
                                            ${block.coherence.toFixed(2)}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Aggregate Signature -->
            ${block.aggregate_signature ? `
            <div class="row mb-4">
                <div class="col-md-12">
                    <div class="card border-info">
                        <div class="card-header bg-info text-white">
                            <h6 class="mb-0">Aggregate 5W1H Signature</h6>
                        </div>
                        <div class="card-body">
                            <dl class="row mb-0">
                                <dt class="col-sm-2">Who:</dt>
                                <dd class="col-sm-10">${escapeHtml(block.aggregate_signature.who || 'N/A')}</dd>
                                
                                <dt class="col-sm-2">What:</dt>
                                <dd class="col-sm-10">${escapeHtml(block.aggregate_signature.what || 'N/A')}</dd>
                                
                                <dt class="col-sm-2">When:</dt>
                                <dd class="col-sm-10">${escapeHtml(block.aggregate_signature.when || 'N/A')}</dd>
                                
                                <dt class="col-sm-2">Where:</dt>
                                <dd class="col-sm-10">${escapeHtml(block.aggregate_signature.where || 'N/A')}</dd>
                                
                                <dt class="col-sm-2">Why:</dt>
                                <dd class="col-sm-10">${escapeHtml(block.aggregate_signature.why || 'N/A')}</dd>
                                
                                <dt class="col-sm-2">How:</dt>
                                <dd class="col-sm-10">${escapeHtml(block.aggregate_signature.how || 'N/A')}</dd>
                            </dl>
                        </div>
                    </div>
                </div>
            </div>
            ` : ''}
            
            <!-- Memory Events -->
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header bg-dark text-white">
                            <h6 class="mb-0">Memory Events in Block (${blockEvents.length})</h6>
                        </div>
                        <div class="card-body" style="max-height: 400px; overflow-y: auto;">
                            ${blockEvents.length > 0 ? `
                                <div class="table-responsive">
                                    <table class="table table-sm table-hover">
                                        <thead class="table-light sticky-top">
                                            <tr>
                                                <th>#</th>
                                                <th>Who</th>
                                                <th>What</th>
                                                <th>When</th>
                                                <th>Salience</th>
                                                <th>Type</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            ${blockEvents.map((event, idx) => `
                                                <tr>
                                                    <td>${idx + 1}</td>
                                                    <td><small>${escapeHtml(event.who)}</small></td>
                                                    <td>
                                                        <small>${escapeHtml(event.what.length > 100 ? event.what.substring(0, 100) + '...' : event.what)}</small>
                                                    </td>
                                                    <td><small>${new Date(event.when).toLocaleString()}</small></td>
                                                    <td>
                                                        <span class="badge ${event.salience >= 0.7 ? 'bg-success' : event.salience >= 0.3 ? 'bg-warning' : 'bg-secondary'}">
                                                            ${event.salience.toFixed(2)}
                                                        </span>
                                                    </td>
                                                    <td><span class="badge bg-primary">${event.type}</span></td>
                                                </tr>
                                            `).join('')}
                                        </tbody>
                                    </table>
                                </div>
                            ` : '<p class="text-muted">No events found in this block. Event IDs may not be loaded.</p>'}
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Timestamps -->
            <div class="row mt-3">
                <div class="col-md-12">
                    <small class="text-muted">
                        Created: ${block.created_at ? new Date(block.created_at).toLocaleString() : 'Unknown'} | 
                        Updated: ${block.updated_at ? new Date(block.updated_at).toLocaleString() : 'Unknown'}
                    </small>
                </div>
            </div>
        </div>
    `;
    
    modalContent.innerHTML = html;
    
    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('memoryBlockModal'));
    modal.show();
}

function createMemoryRow(memory, includeAll = false) {
    const tr = document.createElement('tr');
    
    // Format previews
    const whatPreview = memory.what && memory.what.length > 50 ? 
                        memory.what.substring(0, 50) + '...' : 
                        (memory.what || 'N/A');
    
    const wherePreview = memory.where && memory.where.length > 30 ?
                         memory.where.substring(0, 30) + '...' :
                         (memory.where || 'N/A');
    
    const whyPreview = memory.why && memory.why.length > 30 ?
                       memory.why.substring(0, 30) + '...' :
                       (memory.why || 'N/A');
    
    const howPreview = memory.how && memory.how.length > 30 ?
                       memory.how.substring(0, 30) + '...' :
                       (memory.how || 'N/A');
    
    const whenDate = memory.when ? new Date(memory.when) : null;
    const whenFormatted = whenDate ? whenDate.toLocaleString() : 'N/A';
    
    // Determine memory type badge color
    const typeClass = {
        'observation': 'badge-observation',
        'action': 'badge-action',
        'user_input': 'badge-user-input',
        'system_event': 'badge-system-event'
    }[memory.type] || 'badge-secondary';
    
    // Build row HTML based on whether we're showing all columns
    if (includeAll) {
        tr.innerHTML = `
            <td>${escapeHtml(memory.who || 'N/A')}</td>
            <td title="${escapeHtml(memory.what || '')}">${escapeHtml(whatPreview)}</td>
            <td><small>${whenFormatted}</small></td>
            <td title="${escapeHtml(memory.where || '')}">${escapeHtml(wherePreview)}</td>
            <td title="${escapeHtml(memory.why || '')}">${escapeHtml(whyPreview)}</td>
            <td title="${escapeHtml(memory.how || '')}">${escapeHtml(howPreview)}</td>
            <td>
                <span class="badge ${typeClass}">
                    ${memory.type || 'unknown'}
                </span>
            </td>
            <td>
                <button class="btn btn-sm btn-accent" onclick="viewMemory('${memory.id}')">
                    <i class="bi bi-eye"></i>
                </button>
                <button class="btn btn-sm btn-danger" onclick="deleteMemory('${memory.id}')">
                    <i class="bi bi-trash"></i>
                </button>
            </td>
        `;
    } else {
        // Simplified row for other views
        tr.innerHTML = `
            <td>${escapeHtml(memory.who || 'N/A')}</td>
            <td title="${escapeHtml(memory.what || '')}">${escapeHtml(whatPreview)}</td>
            <td><small>${whenFormatted}</small></td>
            <td>
                <span class="badge ${typeClass}">
                    ${memory.type || 'unknown'}
                </span>
            </td>
            <td>
                <button class="btn btn-sm btn-accent" onclick="viewMemory('${memory.id}')">
                    <i class="bi bi-eye"></i>
                </button>
                <button class="btn btn-sm btn-danger" onclick="deleteMemory('${memory.id}')">
                    <i class="bi bi-trash"></i>
                </button>
            </td>
        `;
    }
    
    return tr;
}

function viewMemory(id) {
    console.log('Viewing memory with ID:', id);
    const memory = [...allMemories.raw, ...allMemories.processed].find(m => m.id === id);
    
    if (!memory) {
        console.error('Memory not found with ID:', id);
        return;
    }
    
    // Set the current memory ID for graph display
    currentMemoryId = id;
    console.log('Set currentMemoryId to:', currentMemoryId);
    
    const content = document.getElementById('memoryDetailContent');
    content.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h6>5W1H Structure</h6>
                <dl class="row">
                    <dt class="col-sm-3">Who:</dt>
                    <dd class="col-sm-9">${escapeHtml(memory.who)}</dd>
                    
                    <dt class="col-sm-3">What:</dt>
                    <dd class="col-sm-9">${escapeHtml(memory.what)}</dd>
                    
                    <dt class="col-sm-3">When:</dt>
                    <dd class="col-sm-9">${new Date(memory.when).toLocaleString()}</dd>
                    
                    <dt class="col-sm-3">Where:</dt>
                    <dd class="col-sm-9">${escapeHtml(memory.where || 'N/A')}</dd>
                    
                    <dt class="col-sm-3">Why:</dt>
                    <dd class="col-sm-9">${escapeHtml(memory.why || 'N/A')}</dd>
                    
                    <dt class="col-sm-3">How:</dt>
                    <dd class="col-sm-9">${escapeHtml(memory.how || 'N/A')}</dd>
                </dl>
            </div>
            <div class="col-md-6">
                <h6>Metadata</h6>
                <dl class="row">
                    <dt class="col-sm-4">ID:</dt>
                    <dd class="col-sm-8"><small>${memory.id}</small></dd>
                    
                    <dt class="col-sm-4">Type:</dt>
                    <dd class="col-sm-8"><span class="badge badge-${memory.type?.replace('_', '-')}">${memory.type || 'unknown'}</span></dd>
                    
                    <dt class="col-sm-4">Confidence:</dt>
                    <dd class="col-sm-8">
                        <div class="progress">
                            <div class="progress-bar bg-info" 
                                 style="width: ${(memory.confidence || 1) * 100}%">
                                ${((memory.confidence || 1) * 100).toFixed(0)}%
                            </div>
                        </div>
                    </dd>
                    
                    <dt class="col-sm-4">Episode:</dt>
                    <dd class="col-sm-8"><small>${memory.episode_id || 'N/A'}</small></dd>
                </dl>
            </div>
        </div>
    `;
    
    const modal = new bootstrap.Modal(document.getElementById('memoryDetailModal'));
    modal.show();
}

async function deleteBlock(blockId) {
    // Find the block to get more info for confirmation
    const block = allMemories.blocks ? allMemories.blocks.find(b => b.id === blockId) : null;
    const blockInfo = block ? `\n\nThis block contains ${block.event_count} events and has ${block.link_count} links.` : '';
    
    if (!confirm(`Are you sure you want to delete this memory block?${blockInfo}\n\nThis action cannot be undone.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/blocks/${blockId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            // Close modal if it's open
            const modal = bootstrap.Modal.getInstance(document.getElementById('memoryBlockModal'));
            if (modal) {
                modal.hide();
            }
            
            loadMemories();
            showAlert('Memory block deleted successfully', 'success');
        } else {
            const data = await response.json();
            showAlert('Failed to delete block: ' + data.error, 'danger');
        }
    } catch (error) {
        showAlert('Error deleting block: ' + error.message, 'danger');
    }
}

async function deleteMemory(id) {
    if (!confirm('Are you sure you want to delete this memory?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/memories/${id}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            loadMemories();
            showAlert('Memory deleted successfully', 'success');
        } else {
            const data = await response.json();
            showAlert('Failed to delete memory: ' + data.error, 'danger');
        }
    } catch (error) {
        showAlert('Error deleting memory: ' + error.message, 'danger');
    }
}

async function handleMemoryCreate(e) {
    e.preventDefault();
    
    const formData = {
        who: document.getElementById('memWho').value,
        what: document.getElementById('memWhat').value,
        where: document.getElementById('memWhere').value,
        why: document.getElementById('memWhy').value,
        how: document.getElementById('memHow').value,
        type: document.getElementById('memType').value
    };
    
    try {
        const response = await fetch('/api/memories', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            document.getElementById('memoryForm').reset();
            // Refresh memories and stats after creation
            loadMemories();
            loadStats();
            showAlert('Memory created successfully with salience: ' + data.memory.salience.toFixed(2), 'success');
        } else {
            showAlert('Failed to create memory: ' + data.message, 'danger');
        }
    } catch (error) {
        showAlert('Error creating memory: ' + error.message, 'danger');
    }
}

async function handleSearch(e) {
    e.preventDefault();
    
    const query = document.getElementById('searchQuery').value;
    
    if (!query) {
        displayMemories();
        return;
    }
    
    try {
        const response = await fetch('/api/recall', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                query: { what: query },
                k: 20
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Display search results
            const processedTable = document.getElementById('processedMemoriesTable');
            const rawTable = document.getElementById('rawMemoriesTable');
            
            processedTable.innerHTML = '';
            rawTable.innerHTML = '';
            
            data.memories.forEach(memory => {
                const row = createMemoryRow(memory);
                // Add score badge
                const scoreBadge = document.createElement('td');
                scoreBadge.innerHTML = `<span class="badge bg-info">Score: ${memory.score.toFixed(3)}</span>`;
                row.appendChild(scoreBadge);
                
                processedTable.appendChild(row);
            });
            
            showAlert(`Found ${data.memories.length} matching memories`, 'info');
        }
    } catch (error) {
        showAlert('Search error: ' + error.message, 'danger');
    }
}

async function loadStats() {
    // Stats display removed from navbar - function kept for compatibility
    // This function now only updates counts
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        if (response.ok) {
            updateCounts();
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

function updateCounts() {
    const processedCount = document.getElementById('processedCount');
    const rawCount = document.getElementById('rawCount');
    const blocksCount = document.getElementById('blocksCount');
    
    if (processedCount) {
        processedCount.textContent = allMemories.processed ? allMemories.processed.length : 0;
    }
    if (rawCount) {
        rawCount.textContent = allMemories.raw ? allMemories.raw.length : 0;
    }
    if (blocksCount) {
        blocksCount.textContent = allMemories.blocks ? allMemories.blocks.length : 0;
    }
}

function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3`;
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Make functions globally available for onclick handlers
window.loadMemories = loadMemories;
window.showCreateMemoryModal = showCreateMemoryModal;
window.createMemory = createMemory;
window.showMemoryGraph = showMemoryGraph;