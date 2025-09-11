// Global state
let currentWeights = {
    semantic: 0.45,
    lexical: 0.25,
    recency: 0.02,
    actor: 0.1,
    temporal: 0.1,
    spatial: 0.04,
    usage: 0.04
};

let currentResults = [];
let currentQuery = '';
let appSettings = {};
let saveTimeout = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Load initial settings and weights
    loadSettings();
    
    // Set up enter key handler for query input
    const queryInput = document.getElementById('query');
    if (queryInput) {
        queryInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    }
});

// Load all settings from server
async function loadSettings() {
    try {
        const response = await fetch('/api/settings');
        if (response.ok) {
            appSettings = await response.json();
            
            // Update weights from settings
            if (appSettings.weights) {
                currentWeights = appSettings.weights;
                updateWeightDisplay();
            }
            
            // Apply other settings to UI
            applySettingsToUI();
        }
    } catch (error) {
        console.error('Failed to load settings:', error);
        // Fall back to just loading weights
        loadWeights();
    }
}

// Apply settings to UI elements
function applySettingsToUI() {
    // Apply analyzer settings
    if (appSettings.analyzer) {
        const topK = document.getElementById('top-k');
        if (topK && appSettings.analyzer.top_k) {
            topK.value = appSettings.analyzer.top_k;
        }
    }
    
    // Apply browser settings (if on browser page)
    if (appSettings.browser && document.getElementById('per-page')) {
        document.getElementById('per-page').value = appSettings.browser.per_page || 50;
        document.getElementById('sort-by').value = appSettings.browser.sort_by || 'when_ts';
        document.getElementById('sort-order').value = appSettings.browser.sort_order || 'desc';
    }
}

// Save settings with debouncing
function saveSettings(immediate = false) {
    if (saveTimeout) {
        clearTimeout(saveTimeout);
    }
    
    if (immediate) {
        performSaveSettings();
    } else {
        // Debounce saves to avoid too many requests
        saveTimeout = setTimeout(performSaveSettings, 1000);
    }
}

// Actually perform the settings save
async function performSaveSettings() {
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(appSettings)
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('Settings saved successfully');
        }
    } catch (error) {
        console.error('Failed to save settings:', error);
    }
}

// Load current weights from server
async function loadWeights() {
    try {
        const response = await fetch('/api/weights');
        if (response.ok) {
            currentWeights = await response.json();
            updateWeightDisplay();
        }
    } catch (error) {
        console.error('Failed to load weights:', error);
    }
}

// Update weight value and display
function updateWeight(dimension, value) {
    const floatValue = value / 100;
    currentWeights[dimension] = floatValue;
    
    // Update display
    const valSpan = document.getElementById(`weight-${dimension}-val`);
    if (valSpan) {
        valSpan.textContent = floatValue.toFixed(2);
    }
    
    // Update sum
    updateWeightSum();
    
    // Update settings and save
    if (!appSettings.weights) {
        appSettings.weights = {};
    }
    appSettings.weights[dimension] = floatValue;
    saveSettings(); // Debounced save
    
    // If we have results, re-score them
    if (currentResults.length > 0) {
        reScoreResults();
    }
}

// Update weight sum display
function updateWeightSum() {
    const sum = Object.values(currentWeights).reduce((a, b) => a + b, 0);
    const sumSpan = document.getElementById('weight-sum');
    if (sumSpan) {
        sumSpan.textContent = `Sum: ${sum.toFixed(2)}`;
        sumSpan.style.color = Math.abs(sum - 1.0) < 0.001 ? 'var(--synth-green)' : 'var(--synth-pink)';
    }
}

// Update all weight displays
function updateWeightDisplay() {
    for (const [dimension, value] of Object.entries(currentWeights)) {
        const slider = document.getElementById(`weight-${dimension}`);
        const valSpan = document.getElementById(`weight-${dimension}-val`);
        
        if (slider) {
            slider.value = value * 100;
        }
        if (valSpan) {
            valSpan.textContent = value.toFixed(2);
        }
    }
    updateWeightSum();
}

// Normalize weights to sum to 1.0
async function normalizeWeights() {
    try {
        const response = await fetch('/api/weights', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(currentWeights)
        });
        
        if (response.ok) {
            const data = await response.json();
            currentWeights = data.weights;
            appSettings.weights = data.weights;
            updateWeightDisplay();
            saveSettings(true); // Immediate save
            
            // Re-search if we have a query
            if (currentQuery) {
                performSearch();
            }
        }
    } catch (error) {
        console.error('Failed to normalize weights:', error);
    }
}

// Reset weights to defaults
async function resetWeights() {
    try {
        const response = await fetch('/api/weights/reset', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        });
        
        if (response.ok) {
            const data = await response.json();
            currentWeights = data.weights;
            appSettings.weights = data.weights;
            updateWeightDisplay();
            
            // Re-search if we have a query
            if (currentQuery) {
                performSearch();
            }
        }
    } catch (error) {
        console.error('Failed to reset weights:', error);
        // Fallback to local defaults
        currentWeights = {
            semantic: 0.45,
            lexical: 0.25,
            recency: 0.02,
            actor: 0.1,
            temporal: 0.1,
            spatial: 0.04,
            usage: 0.04
        };
        updateWeightDisplay();
    }
}

// Perform search
async function performSearch() {
    const query = document.getElementById('query').value.trim();
    const topKElement = document.getElementById('top-k');
    const topK = parseInt(topKElement.value) || 100;
    
    // Save top_k preference
    if (appSettings.analyzer) {
        appSettings.analyzer.top_k = topK;
        saveSettings();
    }
    
    if (!query) {
        alert('Please enter a search query');
        return;
    }
    
    currentQuery = query;
    
    // Show loading state
    const resultsBody = document.getElementById('results-tbody');
    resultsBody.innerHTML = '<tr><td colspan="14" style="text-align: center;">Searching...</td></tr>';
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: query,
                weights: currentWeights,
                top_k: topK
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            currentResults = data.results;
            
            // Display decomposition
            displayDecomposition(data.decomposition);
            
            // Display results
            displayResults(data.results);
            
            // Update visualizations
            updateVisualizations(data.results);
            
            // Update result count
            document.getElementById('result-count').textContent = `(${data.results.length} found)`;
        } else {
            const error = await response.json();
            alert('Search failed: ' + error.error);
        }
    } catch (error) {
        console.error('Search error:', error);
        alert('Search failed: ' + error.message);
    }
}

// Display query decomposition
function displayDecomposition(decomposition) {
    const decomp = document.getElementById('decomposition');
    if (!decomp) return;
    
    decomp.style.display = 'block';
    
    // WHO
    const who = decomposition.who;
    document.getElementById('decomp-who').textContent = 
        who && who.id ? `${who.type || ''}:${who.id}` : 'N/A';
    
    // WHAT
    document.getElementById('decomp-what').textContent = 
        decomposition.what || 'N/A';
    
    // WHEN
    document.getElementById('decomp-when').textContent = 
        decomposition.when || 'N/A';
    
    // WHERE
    const where = decomposition.where;
    document.getElementById('decomp-where').textContent = 
        where && where.value ? `${where.type || ''}:${where.value}` : 'N/A';
    
    // WHY
    document.getElementById('decomp-why').textContent = 
        decomposition.why || 'N/A';
    
    // HOW
    document.getElementById('decomp-how').textContent = 
        decomposition.how || 'N/A';
    
    // ENTITIES
    const entities = decomposition.entities || [];
    const entitiesSpan = document.getElementById('decomp-entities');
    if (entities.length > 0) {
        entitiesSpan.innerHTML = entities.map(e => 
            `<span class="entity-tag">${e}</span>`
        ).join('');
    } else {
        entitiesSpan.textContent = 'None extracted';
    }
}

// Display search results
function displayResults(results) {
    const tbody = document.getElementById('results-tbody');
    
    if (!results || results.length === 0) {
        tbody.innerHTML = '<tr><td colspan="14" class="table-loading">No results found</td></tr>';
        return;
    }
    
    tbody.innerHTML = '';
    
    results.forEach((result, index) => {
        const row = document.createElement('tr');
        
        // Add score-based class
        if (result.scores.total >= 0.8) {
            row.classList.add('high-score');
        } else if (result.scores.total >= 0.5) {
            row.classList.add('medium-score');
        } else {
            row.classList.add('low-score');
        }
        
        // Build row HTML
        row.innerHTML = `
            <td class="col-rank">${index + 1}</td>
            <td class="col-memory-id">
                <a class="memory-id-link" onclick="showMemoryDetails('${result.memory_id}')">${result.memory_id.substring(0, 12)}...</a>
            </td>
            <td class="col-score">
                <span class="score-value ${getScoreClass(result.scores.total)}">${result.scores.total.toFixed(3)}</span>
            </td>
            <td class="col-score">${result.scores.semantic.toFixed(2)}</td>
            <td class="col-score">${result.scores.lexical.toFixed(2)}</td>
            <td class="col-score">${result.scores.recency.toFixed(2)}</td>
            <td class="col-score">${result.scores.actor.toFixed(2)}</td>
            <td class="col-score">${result.scores.temporal.toFixed(2)}</td>
            <td class="col-score">${result.scores.spatial.toFixed(2)}</td>
            <td class="col-score">${result.scores.usage.toFixed(2)}</td>
            <td class="col-entities">${formatEntities(result.entities)}</td>
            <td class="col-when">${formatDate(result.when)}</td>
            <td class="col-text" title="${escapeHtml(result.raw_text)}">${truncateText(result.raw_text, 100)}</td>
            <td class="col-actions">
                <button onclick="showMemoryDetails('${result.memory_id}')" class="btn-table">View</button>
            </td>
        `;
        
        // Add click handler for entire row
        row.addEventListener('click', function(e) {
            // Don't trigger if clicking a button or link
            if (!e.target.matches('button') && !e.target.matches('a')) {
                showMemoryDetails(result.memory_id);
            }
        });
        
        tbody.appendChild(row);
    });
}

// Get score class for styling
function getScoreClass(score) {
    if (score >= 0.8) return 'score-high';
    if (score >= 0.5) return 'score-medium';
    return 'score-low';
}

// Format entities for display
function formatEntities(entities) {
    if (!entities || entities.length === 0) return '';
    
    const maxShow = 3;
    const shown = entities.slice(0, maxShow);
    const html = shown.map(e => `<span class="entity-tag">${e}</span>`).join('');
    
    if (entities.length > maxShow) {
        return html + ` <small>+${entities.length - maxShow} more</small>`;
    }
    return html;
}

// Format date for display
function formatDate(dateStr) {
    if (!dateStr) return 'N/A';
    try {
        const date = new Date(dateStr);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    } catch {
        return dateStr;
    }
}

// Truncate text
function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Show memory details modal
async function showMemoryDetails(memoryId) {
    try {
        const response = await fetch(`/api/memories/${memoryId}`);
        if (!response.ok) {
            const error = await response.json();
            console.error('Failed to fetch memory:', error);
            alert(`Failed to load memory details: ${error.error || 'Unknown error'}`);
            return;
        }
        
        const memory = await response.json();
            
            const modal = document.getElementById('memory-modal');
            const details = document.getElementById('memory-details');
            
            // Format entities
            const entitiesHtml = memory.entities && memory.entities.length > 0
                ? `<div class="entity-list-modal">${memory.entities.map(e => 
                    `<span class="entity-tag" onclick="searchForEntity('${e}')">${e}</span>`
                  ).join('')}</div>`
                : '<span style="color: var(--synth-text-muted);">No entities extracted</span>';
            
            details.innerHTML = `
                <div class="memory-detail">
                    <div class="detail-section">
                        <h5>Memory Identifier</h5>
                        <div style="font-family: monospace; color: var(--synth-cyan); font-size: 1.1rem;">
                            ${memory.memory_id}
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h5>5W1H Components</h5>
                        <div class="detail-grid">
                            <div>
                                <strong>WHO:</strong> 
                                <span style="color: var(--synth-text);">
                                    ${memory.who_type}:${memory.who_id}
                                    ${memory.who_label ? `<em>(${memory.who_label})</em>` : ''}
                                </span>
                            </div>
                            <div>
                                <strong>WHAT:</strong> 
                                <span style="color: var(--synth-text);" title="${escapeHtml(memory.what || '')}">
                                    ${truncateText(memory.what || 'N/A', 100)}
                                </span>
                            </div>
                            <div>
                                <strong>WHEN:</strong> 
                                <span style="color: var(--synth-text);">
                                    ${formatDate(memory.when_ts)}
                                </span>
                            </div>
                            <div>
                                <strong>WHERE:</strong> 
                                <span style="color: var(--synth-text);">
                                    ${memory.where_type}:${memory.where_value}
                                </span>
                            </div>
                            <div>
                                <strong>WHY:</strong> 
                                <span style="color: var(--synth-text);">
                                    ${memory.why || 'N/A'}
                                </span>
                            </div>
                            <div>
                                <strong>HOW:</strong> 
                                <span style="color: var(--synth-text);">
                                    ${memory.how || 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h5>Extracted Entities (${memory.entities ? memory.entities.length : 0})</h5>
                        ${entitiesHtml}
                    </div>
                    
                    <div class="detail-section">
                        <h5>Metadata</h5>
                        <div class="detail-grid">
                            <div><strong>Session ID:</strong> ${memory.session_id || 'N/A'}</div>
                            <div><strong>Source Event:</strong> ${memory.source_event_id || 'N/A'}</div>
                            <div><strong>Token Count:</strong> ${memory.token_count || 0}</div>
                            <div><strong>Created At:</strong> ${formatDate(memory.created_at)}</div>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h5>Raw Text Content</h5>
                        <div class="raw-text-display">
                            ${escapeHtml(memory.raw_text || 'No raw text available')}
                        </div>
                    </div>
                </div>
            `;
            
            modal.style.display = 'block';
        }
    } catch (error) {
        console.error('Failed to load memory details:', error);
        alert('Failed to load memory details');
    }
}

// Search for specific entity (navigate to analyzer)
function searchForEntity(entity) {
    // Close modal first
    closeModal();
    
    // If we're on the analyzer page, just populate the search
    const queryInput = document.getElementById('query');
    if (queryInput) {
        queryInput.value = entity;
        performSearch();
    } else {
        // Navigate to analyzer with entity as query
        window.location.href = `/?query=${encodeURIComponent(entity)}`;
    }
}

// Close modal
function closeModal() {
    document.getElementById('memory-modal').style.display = 'none';
}

// Re-score results with new weights
async function reScoreResults() {
    if (!currentQuery || currentResults.length === 0) return;
    
    // Perform new search with updated weights
    await performSearch();
}

// Update visualizations
function updateVisualizations(results) {
    // Update score distribution chart
    updateScoreChart(results);
    
    // Update entity cloud
    updateEntityCloud(results);
}

// Update score distribution chart
function updateScoreChart(results) {
    const canvas = document.getElementById('score-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Calculate score distribution
    const buckets = new Array(10).fill(0);
    results.forEach(r => {
        const bucket = Math.min(9, Math.floor(r.scores.total * 10));
        buckets[bucket]++;
    });
    
    // Draw bars
    const barWidth = width / 10;
    const maxCount = Math.max(...buckets, 1);
    
    ctx.fillStyle = 'var(--synth-cyan)';
    ctx.strokeStyle = 'var(--synth-border)';
    
    buckets.forEach((count, i) => {
        const barHeight = (count / maxCount) * (height - 40);
        const x = i * barWidth;
        const y = height - barHeight - 20;
        
        // Draw bar
        ctx.fillRect(x + 5, y, barWidth - 10, barHeight);
        ctx.strokeRect(x + 5, y, barWidth - 10, barHeight);
        
        // Draw label
        ctx.fillStyle = 'var(--synth-text)';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${i/10}-${(i+1)/10}`, x + barWidth/2, height - 5);
        
        // Draw count
        if (count > 0) {
            ctx.fillText(count.toString(), x + barWidth/2, y - 5);
        }
        
        ctx.fillStyle = 'var(--synth-cyan)';
    });
    
    // Draw title
    ctx.fillStyle = 'var(--synth-text)';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Score Distribution', width/2, 15);
}

// Update entity cloud
function updateEntityCloud(results) {
    const container = document.getElementById('entity-cloud');
    if (!container) return;
    
    // Count entity frequencies
    const entityCounts = {};
    results.forEach(r => {
        if (r.entities) {
            r.entities.forEach(entity => {
                entityCounts[entity] = (entityCounts[entity] || 0) + 1;
            });
        }
    });
    
    // Sort by frequency
    const sortedEntities = Object.entries(entityCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20);
    
    // Generate cloud
    container.innerHTML = '';
    sortedEntities.forEach(([entity, count]) => {
        const size = Math.min(2, 0.8 + (count / sortedEntities[0][1]) * 1.2);
        const item = document.createElement('div');
        item.className = 'entity-item';
        item.style.fontSize = `${size}rem`;
        item.innerHTML = `${entity}<span class="entity-count">${count}</span>`;
        item.onclick = () => searchForEntity(entity);
        container.appendChild(item);
    });
    
    if (sortedEntities.length === 0) {
        container.innerHTML = '<p style="color: var(--synth-text-muted);">No entities found</p>';
    }
}

// Search for specific entity
function searchForEntity(entity) {
    document.getElementById('query').value = entity;
    performSearch();
}

// Window click handler for modal
window.onclick = function(event) {
    const modal = document.getElementById('memory-modal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
}