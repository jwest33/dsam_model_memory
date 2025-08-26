// Enhanced JavaScript for [DSAM] Dual-Space Agentic Memory

let allMemories = [];
let currentPage = 1;
let itemsPerPage = 10;
let graphNetwork = null;
let residualChart = null;
let spaceUsageChart = null;
let clusteringEnabled = true;
let currentCenterNode = null;  // Track if we're viewing an individual memory

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOMContentLoaded fired');
    
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
        console.log('Loading memories...');
        loadMemories();
    } catch (error) {
        console.error('Error in loadMemories:', error);
    }
    
    try {
        console.log('Loading stats...');
        loadStats();
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
    // Set up component selector listeners for real-time graph updates
    const componentCheckboxes = document.querySelectorAll('#componentSelector input[type="checkbox"]');
    componentCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            // Don't update space weights here - will be done after graph refresh
            if (graphNetwork) {
                refreshGraph();
            }
        });
    });
    
    // Visualization mode change listener
    const vizMode = document.getElementById('visualizationMode');
    if (vizMode) {
        vizMode.addEventListener('change', refreshGraph);
    }
    
    // Toggle clustering button
    const toggleBtn = document.getElementById('toggleClustering');
    const clusteringParams = document.getElementById('clusteringParams');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            clusteringEnabled = !clusteringEnabled;
            toggleBtn.classList.toggle('btn-outline-warning');
            toggleBtn.classList.toggle('btn-warning');
            
            // Show/hide clustering parameters
            if (clusteringParams) {
                clusteringParams.style.display = clusteringEnabled ? 'block' : 'none';
            }
            
            refreshGraph();
        });
    }
    
    // Update graph when clustering parameters change
    const minClusterSizeInput = document.getElementById('minClusterSize');
    const minSamplesInput = document.getElementById('minSamples');
    
    if (minClusterSizeInput) {
        minClusterSizeInput.addEventListener('change', refreshGraph);
    }
    
    if (minSamplesInput) {
        minSamplesInput.addEventListener('change', refreshGraph);
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
    
    // Search functionality
    const searchInput = document.getElementById('memorySearch');
    if (searchInput) {
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
        
        // Add assistant response with space usage indicator
        addChatMessage(data.response, 'assistant', {
            memories_used: data.memories_used,
            space_weights: data.space_weights
        });
        
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

function addChatMessage(message, sender, metadata = {}) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message mb-3`;
    
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
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <strong>${sender === 'user' ? 'You' : 'Assistant'}</strong>
            <span class="text-muted ms-2">${new Date().toLocaleTimeString()}</span>
            ${metadata.memories_used ? `<span class="badge bg-dark ms-2">${metadata.memories_used} memories</span>` : ''}
        </div>
        <div class="message-content">${escapeHtml(message)}</div>
        ${spaceIndicator}
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Memory loading and display
async function loadMemories() {
    console.log('Loading memories...');
    try {
        const response = await fetch('/api/memories');
        console.log('Response status:', response.status);
        const data = await response.json();
        console.log('Data received:', data);
        
        allMemories = data.memories || [];
        console.log('Total memories loaded:', allMemories.length);
        
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
        displayMemories();
        
    } catch (error) {
        console.error('Error loading memories:', error);
    }
}

function displayMemories() {
    const tbody = document.getElementById('memoryTableBody');
    if (!tbody) return;
    
    const startIdx = (currentPage - 1) * itemsPerPage;
    const endIdx = startIdx + itemsPerPage;
    const pageMemories = allMemories.slice(startIdx, endIdx);
    
    tbody.innerHTML = '';
    
    pageMemories.forEach(memory => {
        const row = document.createElement('tr');
        
        // Determine dominant space
        const spaceIndicator = getSpaceIndicator(memory);
        const residualIndicator = getResidualIndicator(memory);
        
        // Handle potentially empty fields
        const who = memory.who || '';
        const what = memory.what || '';
        const when = memory.when || '';
        const whatDisplay = what.length > 50 ? what.substring(0, 50) + '...' : what;
        
        row.innerHTML = `
            <td>${escapeHtml(who)}</td>
            <td>${escapeHtml(whatDisplay)}</td>
            <td>${formatDate(when)}</td>
            <td>${spaceIndicator}</td>
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
        
        tbody.appendChild(row);
    });
    
    // Update pagination
    updatePagination();
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
    const modal = new bootstrap.Modal(document.getElementById('memoryGraphModal'));
    modal.show();
    
    // Initialize graph after modal is shown
    setTimeout(() => {
        initializeGraph();
    }, 300);
}

async function initializeGraph(centerNodeId = null) {
    try {
        // Update global center node tracking
        currentCenterNode = centerNodeId;
        
        // Get selected components
        const components = getSelectedComponents();
        
        const requestBody = {
            components: components,
            use_clustering: clusteringEnabled,
            visualization_mode: document.getElementById('visualizationMode').value,
            // Include HDBSCAN parameters if clustering is enabled
            min_cluster_size: clusteringEnabled ? parseInt(document.getElementById('minClusterSize').value) : 5,
            min_samples: clusteringEnabled ? parseInt(document.getElementById('minSamples').value) : 3
        };
        
        // Add center node if specified
        if (centerNodeId) {
            requestBody.center_node = centerNodeId;
            requestBody.similarity_threshold = 0.4;  // Only show closely related memories
        }
        
        const response = await fetch('/api/graph', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        // Create graph visualization
        createGraphVisualization(data);
        
        // Update statistics
        updateGraphStats(data);
        
        // Update space weights with actual data
        updateSpaceWeights(data);
        
    } catch (error) {
        console.error('Error loading graph:', error);
    }
}

function getSelectedComponents() {
    const components = [];
    document.querySelectorAll('#componentSelector input[type="checkbox"]:checked').forEach(checkbox => {
        components.push(checkbox.value);
    });
    return components;
}

function createGraphVisualization(data) {
    const container = document.getElementById('memoryGraph');
    
    // Prepare nodes with color coding based on space/cluster
    const nodes = new vis.DataSet(data.nodes.map(node => {
        const nodeConfig = {
            id: node.id,
            label: node.label,
            color: getNodeColor(node),
            size: 10 + (node.centrality || 0) * 15,  // Reduced from 20 + 30
            title: createNodeTooltip(node),
            ...node
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
    
    // Prepare edges with gradient color based on similarity strength
    const edges = new vis.DataSet(data.edges.map(edge => {
        let edgeColor;
        let edgeTitle;
        
        // Special coloring for conversation edges
        if (edge.type === 'conversation') {
            // Bright cyan-purple for User-Assistant conversation flow
            edgeColor = {
                color: '#ff00ff',  // Bright purple for conversation edges
                opacity: 0.8
            };
            edgeTitle = 'Conversation Flow';
        } else {
            // Gradient from blue to purple based on weight for similarity edges
            const weight = edge.weight || 0.5;
            const r = Math.floor(0 + (255 * weight));  // Red increases with weight
            const g = Math.floor(100 * (1 - weight));  // Green decreases with weight
            const b = Math.floor(255);  // Blue stays high
            
            edgeColor = {
                color: `rgb(${r}, ${g}, ${b})`,
                opacity: Math.min(0.9, 0.4 + edge.weight * 0.5)
            };
            edgeTitle = `Similarity: ${edge.weight.toFixed(3)}`;
        }
        
        return {
            from: edge.from,
            to: edge.to,
            value: edge.weight || 0.8,
            color: edgeColor,
            title: edgeTitle,
            width: edge.type === 'conversation' ? 2 : 1  // Make conversation edges thicker
        };
    }));
    
    const graphData = { nodes, edges };
    
    // Check if we have a center node for special layout
    const centerNode = data.nodes.find(n => n.is_center);
    
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
                type: 'continuous'
            },
            width: 1,  // Reduced from 2
            shadow: true
        },
        physics: {
            stabilization: {
                iterations: 200
            },
            barnesHut: {
                gravitationalConstant: -10000,  // Increased for more spacing
                centralGravity: 0.2,  // Reduced to allow more spread
                springConstant: 0.015,  // Further reduced for looser connections
                springLength: 200,  // Increased spring length for more spacing
                damping: 0.09,  // Added damping for stability
                avoidOverlap: 0.7  // Increased to prevent overlap with more spacing
            }
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
    
    // Add click handlers
    graphNetwork.on('click', function(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            const node = nodes.get(nodeId);
            displayNodeDetails(node);
        }
    });
}

function getNodeColor(node) {
    const vizMode = document.getElementById('visualizationMode').value;
    
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
                    borderColor: '#ffc107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 0.4,
                        ticks: {
                            color: '#ffffff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
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
    
    // Space usage distribution chart
    const spaceCtx = document.getElementById('spaceUsageChart');
    if (spaceCtx) {
        spaceUsageChart = new Chart(spaceCtx, {
            type: 'doughnut',
            data: {
                labels: ['Euclidean Dominant', 'Hyperbolic Dominant', 'Balanced'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#00bcd4', '#ffc107', '#6c757d']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
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
        
        // Update space usage chart
        if (spaceUsageChart && data.space_distribution) {
            spaceUsageChart.data.datasets[0].data = [
                data.space_distribution.euclidean,
                data.space_distribution.hyperbolic,
                data.space_distribution.balanced
            ];
            spaceUsageChart.update();
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

window.viewMemoryDetails = async function(memoryId) {
    // Find the memory
    const memory = allMemories.find(m => m.id === memoryId);
    if (!memory) {
        console.error('Memory not found:', memoryId);
        return;
    }
    
    // Create a modal to show details
    const modalHtml = `
        <div class="modal fade" id="memoryDetailModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Memory Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <strong>Who:</strong> ${escapeHtml(memory.who || '—')}
                            </div>
                            <div class="col-md-6">
                                <strong>When:</strong> ${formatDate(memory.when)}
                            </div>
                        </div>
                        <div class="row mb-3">
                            <div class="col-md-12">
                                <strong>What:</strong><br>
                                <div class="p-2 bg-dark rounded">${escapeHtml(memory.what || '—')}</div>
                            </div>
                        </div>
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <strong>Where:</strong> ${escapeHtml(memory.where || '—')}
                            </div>
                            <div class="col-md-6">
                                <strong>Type:</strong> ${escapeHtml(memory.type || '—')}
                            </div>
                        </div>
                        <div class="row mb-3">
                            <div class="col-md-12">
                                <strong>Why:</strong><br>
                                <div class="p-2 bg-dark rounded">${escapeHtml(memory.why || '—')}</div>
                            </div>
                        </div>
                        <div class="row mb-3">
                            <div class="col-md-12">
                                <strong>How:</strong><br>
                                <div class="p-2 bg-dark rounded">${escapeHtml(memory.how || '—')}</div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-6">
                                <strong>Residual Norm:</strong> ${memory.residual_norm ? memory.residual_norm.toFixed(4) : '0.0000'}
                            </div>
                            <div class="col-md-6">
                                <strong>Has Residual:</strong> ${memory.has_residual ? 'Yes' : 'No'}
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-info" onclick="window.showMemoryInGraph('${memoryId}')">
                            <i class="bi bi-diagram-3"></i> Show in Graph
                        </button>
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

window.showMemoryInGraph = function(memoryId) {
    // Close the details modal
    const detailModal = bootstrap.Modal.getInstance(document.getElementById('memoryDetailModal'));
    if (detailModal) {
        detailModal.hide();
    }
    
    // Open the graph modal
    const graphModal = new bootstrap.Modal(document.getElementById('memoryGraphModal'));
    graphModal.show();
    
    // Initialize graph after modal is shown, with the selected memory as center
    setTimeout(() => {
        initializeGraph(memoryId).then(() => {
            // Focus on the center node
            if (graphNetwork && memoryId) {
                // The node should already be centered due to server-side filtering
                // Additional focus for emphasis
                graphNetwork.fit({
                    animation: {
                        duration: 500,
                        easingFunction: 'easeInOutQuad'
                    }
                });
                
                // Select and highlight the center node
                graphNetwork.selectNodes([memoryId]);
                
                // Find and display the node details
                const node = allMemories.find(m => m.id === memoryId);
                if (node) {
                    displayNodeDetails(node);
                }
            }
        });
    }, 300);
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
        currentPage = 1;
        displayMemories();
        
    } catch (error) {
        console.error('Error searching:', error);
    }
}

// Utility functions
function updatePagination() {
    const totalPages = Math.ceil(allMemories.length / itemsPerPage);
    const pagination = document.getElementById('memoryPagination');
    
    if (!pagination) return;
    
    pagination.innerHTML = '';
    
    // Previous button
    const prevLi = document.createElement('li');
    prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    prevLi.innerHTML = `<a class="page-link" href="#" onclick="changePage(${currentPage - 1})">Previous</a>`;
    pagination.appendChild(prevLi);
    
    // Page numbers
    for (let i = 1; i <= totalPages; i++) {
        const li = document.createElement('li');
        li.className = `page-item ${i === currentPage ? 'active' : ''}`;
        li.innerHTML = `<a class="page-link" href="#" onclick="changePage(${i})">${i}</a>`;
        pagination.appendChild(li);
    }
    
    // Next button
    const nextLi = document.createElement('li');
    nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
    nextLi.innerHTML = `<a class="page-link" href="#" onclick="changePage(${currentPage + 1})">Next</a>`;
    pagination.appendChild(nextLi);
}

function changePage(page) {
    const totalPages = Math.ceil(allMemories.length / itemsPerPage);
    if (page < 1 || page > totalPages) return;
    
    currentPage = page;
    displayMemories();
}

function sortTable(field) {
    allMemories.sort((a, b) => {
        if (a[field] < b[field]) return -1;
        if (a[field] > b[field]) return 1;
        return 0;
    });
    
    currentPage = 1;
    displayMemories();
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
