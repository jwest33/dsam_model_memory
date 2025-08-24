// Main application JavaScript

let allMemories = { raw: [], processed: [], blocks: [] };

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadMemories();
    loadStats();
    
    // Set up event listeners
    document.getElementById('chatForm').addEventListener('submit', handleChatSubmit);
    document.getElementById('memoryForm').addEventListener('submit', handleMemoryCreate);
    document.getElementById('searchForm').addEventListener('submit', handleSearch);
    
    // Add tab change listener to refresh memories when memory tab is selected
    const memoryTab = document.getElementById('memory-tab');
    if (memoryTab) {
        memoryTab.addEventListener('click', function (e) {
            // Refresh when clicking on memory tab
            console.log('Memory tab activated, refreshing memories...');
            setTimeout(() => {
                loadMemories(true); // Show loading indicator
                loadStats();
            }, 100); // Small delay to ensure tab is shown
        });
    }
    
    // Bootstrap 5 tab shown event
    const memoryTabEl = document.querySelector('button[data-bs-target="#memory"]');
    if (memoryTabEl) {
        memoryTabEl.addEventListener('shown.bs.tab', function (event) {
            console.log('Memory tab shown, refreshing...');
            loadMemories();
            loadStats();
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
            
            // Update memories used indicator
            if (data.memories_used > 0) {
                updateRecentMemories(data.memories_used);
            }
            
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

function updateRecentMemories(count) {
    const recentDiv = document.getElementById('recentMemories');
    recentDiv.innerHTML = `
        <div class="alert alert-info">
            <i class="bi bi-info-circle"></i> Used ${count} memories in this response
        </div>
    `;
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
            processedTable.appendChild(createMemoryRow(memory, true));  // Include block info
        });
    } else {
        processedTable.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No processed memories</td></tr>';
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
                <button class="btn btn-sm btn-info" onclick="viewBlock('${block.id}')">
                    <i class="bi bi-eye"></i>
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
                        <div class="card-header bg-primary text-white">
                            <h6 class="mb-0">Block Overview</h6>
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

function createMemoryRow(memory, includeBlock = false) {
    const tr = document.createElement('tr');
    const whatPreview = memory.what.length > 50 ? 
                        memory.what.substring(0, 50) + '...' : 
                        memory.what;
    
    const whenDate = new Date(memory.when);
    const whenFormatted = whenDate.toLocaleString();
    
    let blockInfo = '';
    if (includeBlock && memory.block_ids && memory.block_ids.length > 0) {
        const blockSalience = memory.block_saliences && memory.block_saliences[0] ? 
                             memory.block_saliences[0].toFixed(2) : 'N/A';
        blockInfo = `<td>
            <small>${memory.block_ids[0].substring(0, 8)}...</small>
            <span class="badge bg-info ms-1">${blockSalience}</span>
        </td>`;
    } else if (includeBlock) {
        blockInfo = '<td>-</td>';
    }
    
    tr.innerHTML = `
        <td>${escapeHtml(memory.who)}</td>
        <td>${escapeHtml(whatPreview)}</td>
        <td><small>${whenFormatted}</small></td>
        <td>
            <span class="badge ${memory.salience >= 0.7 ? 'bg-success' : memory.salience >= 0.3 ? 'bg-warning' : 'bg-secondary'}">
                ${memory.salience.toFixed(2)}
            </span>
        </td>
        ${blockInfo}
        <td>
            <button class="btn btn-sm btn-info" onclick="viewMemory('${memory.id}')">
                <i class="bi bi-eye"></i>
            </button>
            <button class="btn btn-sm btn-danger" onclick="deleteMemory('${memory.id}')">
                <i class="bi bi-trash"></i>
            </button>
        </td>
    `;
    
    return tr;
}

function viewMemory(id) {
    const memory = [...allMemories.raw, ...allMemories.processed].find(m => m.id === id);
    
    if (!memory) return;
    
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
                    <dd class="col-sm-8"><span class="badge bg-primary">${memory.type}</span></dd>
                    
                    <dt class="col-sm-4">Salience:</dt>
                    <dd class="col-sm-8">
                        <div class="progress">
                            <div class="progress-bar ${memory.salience >= 0.7 ? 'bg-success' : memory.salience >= 0.3 ? 'bg-warning' : 'bg-danger'}" 
                                 style="width: ${memory.salience * 100}%">
                                ${memory.salience.toFixed(2)}
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
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        if (response.ok) {
            const summary = document.getElementById('stats-summary');
            summary.innerHTML = `
                <i class="bi bi-database"></i> Raw: ${stats.raw_count} | 
                Processed: ${stats.processed_count} | 
                Episodes: ${stats.episode_count}
            `;
            
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