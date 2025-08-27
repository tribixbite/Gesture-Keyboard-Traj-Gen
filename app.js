
// App state
let currentDataset = null;
let generatedDataset = null;
let selectedTrace = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupGenerator();
  setupViewer();
  
  // Auto-load default wordlist
  const defaultUrl = document.getElementById('wordlist-url').value;
  if (defaultUrl) {
    setTimeout(async () => {
      try {
        await loadWordlistFromUrl(defaultUrl);
      } catch (error) {
        console.error('Failed to auto-load wordlist:', error);
        // Fallback to a basic wordlist
        const wordlistFile = document.getElementById('wordlist-file');
        const fallbackWords = ['hello', 'world', 'test', 'example', 'demo', 'sample', 'keyboard', 'swipe', 'gesture', 'trace'];
        wordlistFile.dataset.words = JSON.stringify(fallbackWords);
        showStatus('Using fallback wordlist (' + fallbackWords.length + ' words)', 'info');
      }
    }, 500);
  }
});

// Tab switching
function setupTabs() {
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      const tabName = tab.dataset.tab;
      
      // Update tab buttons
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      
      // Update tab content
      document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
      });
      document.getElementById(tabName + '-tab').classList.add('active');
    });
  });
}

// Generator setup
function setupGenerator() {
  const generatorType = document.getElementById('generator-type');
  const generateBtn = document.getElementById('generate-btn');
  const exportBtn = document.getElementById('export-btn');
  const viewBtn = document.getElementById('view-generated-btn');
  const wordlistFile = document.getElementById('wordlist-file');
  const wordlistUrl = document.getElementById('wordlist-url');
  
  // Update settings when generator changes
  generatorType.addEventListener('change', updateGeneratorSettings);
  updateGeneratorSettings();
  
  // Auto-load default wordlist
  if (wordlistUrl.value) {
    loadWordlistFromUrl(wordlistUrl.value);
  }
  
  // Handle wordlist file upload
  wordlistFile.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const content = await file.text();
    const format = document.getElementById('wordlist-format').value;
    
    try {
      const response = await fetch('/api/parse-wordlist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, format })
      });
      
      const data = await response.json();
      wordlistFile.dataset.words = JSON.stringify(data.words);
      showStatus('Loaded ' + data.words.length + ' words', 'success');
    } catch (error) {
      showStatus('Error parsing wordlist: ' + error.message, 'error');
    }
  });
  
  // Handle URL input
  wordlistUrl.addEventListener('blur', async () => {
    const url = wordlistUrl.value.trim();
    if (url) {
      loadWordlistFromUrl(url);
    }
  });
  
  // Generate dataset
  generateBtn.addEventListener('click', async () => {
    const words = JSON.parse(wordlistFile.dataset.words || '[]');
    
    if (words.length === 0) {
      showStatus('Please load a wordlist first', 'error');
      return;
    }
    
    const config = {
      generator: {
        type: generatorType.value,
        settings: getGeneratorSettings()
      },
      words: words,
      options: {
        noDuplicates: document.getElementById('no-duplicates').checked,
        random: document.getElementById('random-order').checked,
        minLength: parseInt(document.getElementById('min-length').value),
        maxLength: parseInt(document.getElementById('max-length').value),
        maxQuantity: parseInt(document.getElementById('max-quantity').value)
      }
    };
    
    generateBtn.disabled = true;
    showStatus('Generating dataset...', 'info');
    
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      
      generatedDataset = await response.json();
      
      showStatus('Generated ' + generatedDataset.traces.length + ' traces', 'success');
      showPreview(generatedDataset);
      
      exportBtn.disabled = false;
      viewBtn.disabled = false;
    } catch (error) {
      showStatus('Error generating dataset: ' + error.message, 'error');
    } finally {
      generateBtn.disabled = false;
    }
  });
  
  // Export dataset
  exportBtn.addEventListener('click', () => {
    if (!generatedDataset) return;
    
    const blob = new Blob([JSON.stringify(generatedDataset, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'swipe_dataset_' + Date.now() + '.json';
    a.click();
    URL.revokeObjectURL(url);
  });
  
  // View generated dataset
  viewBtn.addEventListener('click', () => {
    if (!generatedDataset) return;
    
    currentDataset = generatedDataset;
    document.querySelector('[data-tab="viewer"]').click();
    displayDataset(currentDataset);
  });
}

// Update generator-specific settings
function updateGeneratorSettings() {
  const type = document.getElementById('generator-type').value;
  const settingsDiv = document.getElementById('generator-settings');
  
  let html = '<label>Generator Settings</label>';
  
  switch (type) {
    case 'rnn':
      html += `
        <div class="options">
          <div class="input-group">
            <label>Temperature</label>
            <input type="range" id="rnn-temperature" min="0.1" max="2" step="0.1" value="1">
          </div>
          <div class="input-group">
            <label>Hidden size</label>
            <select id="rnn-hidden-size">
              <option value="128">128</option>
              <option value="256" selected>256</option>
              <option value="512">512</option>
            </select>
          </div>
        </div>
      `;
      break;
      
    case 'gan':
      html += `
        <div class="options">
          <div class="input-group">
            <label>Noise dim</label>
            <input type="number" id="gan-noise-dim" value="100" min="50" max="200">
          </div>
          <div class="input-group">
            <label>Style</label>
            <select id="gan-style">
              <option value="natural">Natural</option>
              <option value="fast">Fast</option>
              <option value="precise">Precise</option>
            </select>
          </div>
        </div>
      `;
      break;
      
    case 'transformer':
      html += `
        <div class="options">
          <div class="input-group">
            <label>Model</label>
            <select id="transformer-model">
              <option value="small">Small (Fast)</option>
              <option value="base">Base</option>
              <option value="large">Large (Accurate)</option>
            </select>
          </div>
          <div class="input-group">
            <label>Beam width</label>
            <input type="number" id="transformer-beam" value="5" min="1" max="10">
          </div>
        </div>
      `;
      break;
      
    case 'rule-based':
      html += `
        <div class="options">
          <div class="input-group">
            <label>Smoothness</label>
            <input type="range" id="rule-smoothness" min="0" max="1" step="0.1" value="0.7">
          </div>
          <div class="input-group">
            <label>Noise level</label>
            <input type="range" id="rule-noise" min="0" max="1" step="0.1" value="0.2">
          </div>
        </div>
      `;
      break;
  }
  
  settingsDiv.innerHTML = html;
}

// Load wordlist from URL
async function loadWordlistFromUrl(url) {
  if (!url) return;
  
  try {
    showStatus('Loading wordlist from URL...', 'info');
    
    // Use server-side proxy to avoid CORS issues
    const fetchResponse = await fetch('/api/fetch-wordlist', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    
    const fetchData = await fetchResponse.json();
    
    if (fetchData.error) {
      throw new Error(fetchData.error);
    }
    
    const content = fetchData.content;
    const format = document.getElementById('wordlist-format').value;
    
    const parseResponse = await fetch('/api/parse-wordlist', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, format })
    });
    
    const data = await parseResponse.json();
    
    if (data.error) {
      throw new Error(data.error);
    }
    
    const wordlistFile = document.getElementById('wordlist-file');
    wordlistFile.dataset.words = JSON.stringify(data.words);
    showStatus('Loaded ' + data.words.length + ' words from URL', 'success');
    
  } catch (error) {
    console.error('URL loading error:', error);
    showStatus('Error loading wordlist from URL: ' + error.message, 'error');
    throw error; // Re-throw so the fallback can handle it
  }
}

// Get current generator settings
function getGeneratorSettings() {
  const type = document.getElementById('generator-type').value;
  const settings = {};
  
  switch (type) {
    case 'rnn':
      settings.temperature = parseFloat(document.getElementById('rnn-temperature')?.value || 1);
      settings.hiddenSize = parseInt(document.getElementById('rnn-hidden-size')?.value || 256);
      break;
    case 'gan':
      settings.noiseDim = parseInt(document.getElementById('gan-noise-dim')?.value || 100);
      settings.style = document.getElementById('gan-style')?.value || 'natural';
      break;
    case 'transformer':
      settings.model = document.getElementById('transformer-model')?.value || 'base';
      settings.beamWidth = parseInt(document.getElementById('transformer-beam')?.value || 5);
      break;
    case 'rule-based':
      settings.smoothness = parseFloat(document.getElementById('rule-smoothness')?.value || 0.7);
      settings.noise = parseFloat(document.getElementById('rule-noise')?.value || 0.2);
      break;
  }
  
  return settings;
}

// Viewer setup
function setupViewer() {
  const datasetFile = document.getElementById('dataset-file');
  const filterInput = document.getElementById('filter-input');
  const sortBy = document.getElementById('sort-by');
  
  // Handle dataset upload
  datasetFile.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Show loading status
    showStatus('Loading dataset...', 'info');
    
    try {
      // Check file type
      if (!file.name.toLowerCase().endsWith('.json')) {
        throw new Error('Please select a JSON file (.json)');
      }
      
      // Check file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        throw new Error('File is too large. Maximum size is 10MB.');
      }
      
      const content = await file.text();
      
      // Validate JSON format
      let dataset;
      try {
        dataset = JSON.parse(content);
      } catch (parseError) {
        throw new Error('Invalid JSON format. Please check your file.');
      }
      
      // Validate dataset structure
      if (!validateDatasetStructure(dataset)) {
        throw new Error('Invalid dataset format. Expected structure: {traces: [{word, trace: [{x, y, t, p}]}]}');
      }
      
      currentDataset = dataset;
      displayDataset(currentDataset);
      showStatus(`Successfully loaded ${dataset.traces?.length || 0} traces`, 'success');
      
    } catch (error) {
      console.error('Dataset loading error:', error);
      showStatus('Error loading dataset: ' + error.message, 'error');
      
      // Clear the file input
      datasetFile.value = '';
    }
  });
  
  // Filter and sort
  filterInput.addEventListener('input', () => displayDataset(currentDataset));
  sortBy.addEventListener('change', () => displayDataset(currentDataset));
}

// Validate dataset structure
function validateDatasetStructure(dataset) {
  try {
    // Check if it's an object
    if (!dataset || typeof dataset !== 'object') {
      return false;
    }
    
    // Check if it has traces array
    if (!Array.isArray(dataset.traces)) {
      return false;
    }
    
    // Check if traces are not empty
    if (dataset.traces.length === 0) {
      return false;
    }
    
    // Validate first few traces structure
    const samplesToCheck = Math.min(5, dataset.traces.length);
    
    for (let i = 0; i < samplesToCheck; i++) {
      const trace = dataset.traces[i];
      
      // Check trace structure
      if (!trace || typeof trace !== 'object') {
        return false;
      }
      
      // Check if it has word and trace
      if (typeof trace.word !== 'string' || !Array.isArray(trace.trace)) {
        return false;
      }
      
      // Check if trace has points
      if (trace.trace.length === 0) {
        continue; // Allow empty traces
      }
      
      // Check first point structure
      const point = trace.trace[0];
      if (!point || typeof point !== 'object') {
        return false;
      }
      
      // Check point has required properties
      if (typeof point.x !== 'number' || typeof point.y !== 'number') {
        return false;
      }
      
      // t and p are optional but should be numbers if present
      if (point.t !== undefined && typeof point.t !== 'number') {
        return false;
      }
      
      if (point.p !== undefined && typeof point.p !== 'number') {
        return false;
      }
    }
    
    return true;
    
  } catch (error) {
    console.error('Validation error:', error);
    return false;
  }
}

// Display dataset in viewer
function displayDataset(dataset) {
  if (!dataset) return;
  
  // Show info
  const infoDiv = document.getElementById('dataset-info');
  infoDiv.innerHTML = `
    <div class="info-item">
      <div class="value">${dataset.traces.length}</div>
      <div class="label">Traces</div>
    </div>
    <div class="info-item">
      <div class="value">${dataset.words || dataset.traces.length}</div>
      <div class="label">Words</div>
    </div>
    <div class="info-item">
      <div class="value">${dataset.generator || 'Unknown'}</div>
      <div class="label">Generator</div>
    </div>
    <div class="info-item">
      <div class="value">${new Date(dataset.timestamp).toLocaleDateString()}</div>
      <div class="label">Generated</div>
    </div>
  `;
  
  // Filter and sort traces
  const filterText = document.getElementById('filter-input').value.toLowerCase();
  const sortKey = document.getElementById('sort-by').value;
  
  let traces = dataset.traces.filter(t => 
    !filterText || t.word.toLowerCase().includes(filterText)
  );
  
  traces.sort((a, b) => {
    switch (sortKey) {
      case 'word':
        return a.word.localeCompare(b.word);
      case 'length':
        return a.word.length - b.word.length;
      case 'points':
        return a.trace.length - b.trace.length;
      case 'duration':
        return (a.metadata?.duration || 0) - (b.metadata?.duration || 0);
      default:
        return 0;
    }
  });
  
  // Display trace list
  const listDiv = document.getElementById('trace-list');
  listDiv.innerHTML = traces.slice(0, 50).map((trace, i) => `
    <div class="trace-item" data-index="${i}">
      <div class="word">${trace.word}</div>
      <div class="meta">
        ${trace.trace.length} points · 
        ${((trace.metadata?.duration || 0) / 1000).toFixed(1)}s
      </div>
    </div>
  `).join('');
  
  // Handle trace selection
  document.querySelectorAll('.trace-item').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('.trace-item').forEach(i => i.classList.remove('selected'));
      item.classList.add('selected');
      
      const index = parseInt(item.dataset.index);
      selectedTrace = traces[index];
      drawTrace(selectedTrace);
      showTraceDetails(selectedTrace);
    });
  });
  
  // Select first trace
  if (traces.length > 0) {
    document.querySelector('.trace-item').click();
  }
}

// QWERTY keyboard layout definition
const QWERTY_LAYOUT = [
  ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
  ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
  ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
];

// Get keyboard coordinates for QWERTY layout
function getKeyboardCoordinates(width, height) {
  const keys = {};
  const keyWidth = width / 10; // Base width on 10 keys in top row
  const keyHeight = height / 4; // 4 rows including space for spacing
  const startY = keyHeight * 0.2; // Small top margin
  
  // Top row (QWERTYUIOP)
  QWERTY_LAYOUT[0].forEach((key, i) => {
    keys[key] = {
      x: i * keyWidth + keyWidth / 2,
      y: startY + keyHeight / 2,
      width: keyWidth * 0.9,
      height: keyHeight * 0.8
    };
  });
  
  // Middle row (ASDFGHJKL) - slightly offset
  const middleOffset = keyWidth * 0.25;
  QWERTY_LAYOUT[1].forEach((key, i) => {
    keys[key] = {
      x: middleOffset + i * keyWidth + keyWidth / 2,
      y: startY + keyHeight * 1.2 + keyHeight / 2,
      width: keyWidth * 0.9,
      height: keyHeight * 0.8
    };
  });
  
  // Bottom row (ZXCVBNM) - more offset
  const bottomOffset = keyWidth * 0.75;
  QWERTY_LAYOUT[2].forEach((key, i) => {
    keys[key] = {
      x: bottomOffset + i * keyWidth + keyWidth / 2,
      y: startY + keyHeight * 2.4 + keyHeight / 2,
      width: keyWidth * 0.9,
      height: keyHeight * 0.8
    };
  });
  
  // Space bar
  keys[' '] = {
    x: width / 2,
    y: startY + keyHeight * 3.6 + keyHeight / 2,
    width: keyWidth * 4,
    height: keyHeight * 0.6
  };
  
  return keys;
}

// Draw QWERTY keyboard
function drawKeyboard(canvas, ctx) {
  const width = canvas.width;
  const height = canvas.height;
  const keys = getKeyboardCoordinates(width, height);
  
  // Draw keyboard background
  ctx.fillStyle = '#1a1a1a';
  ctx.fillRect(0, 0, width, height);
  
  // Draw keys
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#2a2a2a';
  ctx.font = '12px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  
  Object.entries(keys).forEach(([key, pos]) => {
    // Draw key background
    const x = pos.x - pos.width / 2;
    const y = pos.y - pos.height / 2;
    
    ctx.fillStyle = '#2a2a2a';
    ctx.fillRect(x, y, pos.width, pos.height);
    ctx.strokeRect(x, y, pos.width, pos.height);
    
    // Draw key label
    ctx.fillStyle = '#888';
    ctx.fillText(key === ' ' ? 'SPACE' : key, pos.x, pos.y);
  });
  
  return keys;
}

// Draw trace on keyboard
function drawTrace(traceData) {
  const canvas = document.getElementById('trace-canvas');
  if (!canvas) {
    console.error('Canvas not found!');
    return;
  }
  const ctx = canvas.getContext('2d');
  const trace = traceData.trace;
  
  // Ensure canvas maintains 1.4:1 aspect ratio
  const displayWidth = canvas.clientWidth;
  const displayHeight = displayWidth / 1.4;
  
  if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
    canvas.width = displayWidth;
    canvas.height = displayHeight;
  }
  
  // Clear canvas and draw keyboard
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const keys = drawKeyboard(canvas, ctx);
  
  if (trace.length < 2) return;
  
  // Analyze trace coordinate range to determine scaling
  const minX = Math.min(...trace.map(p => p.x));
  const maxX = Math.max(...trace.map(p => p.x));
  const minY = Math.min(...trace.map(p => p.y));
  const maxY = Math.max(...trace.map(p => p.y));
  
  const rangeX = maxX - minX;
  const rangeY = maxY - minY;
  
  // Determine if coordinates need scaling
  let normalizeX, normalizeY;
  
  if (maxX <= canvas.width && maxY <= canvas.height && minX >= 0 && minY >= 0) {
    // Coordinates seem to be in canvas pixel range
    normalizeX = (x) => x;
    normalizeY = (y) => y;
  } else if (maxX <= 1 && maxY <= 1 && minX >= 0 && minY >= 0) {
    // Coordinates are normalized 0-1
    normalizeX = (x) => x * canvas.width;
    normalizeY = (y) => y * canvas.height;
  } else {
    // Scale to fit canvas with padding
    const padding = 40;
    normalizeX = (x) => padding + ((x - minX) / rangeX) * (canvas.width - 2 * padding);
    normalizeY = (y) => padding + ((y - minY) / rangeY) * (canvas.height - 2 * padding);
  }
  
  // Draw trace path
  ctx.strokeStyle = '#667eea';
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = '#667eea';
  ctx.shadowBlur = 5;
  
  ctx.beginPath();
  let pathStarted = false;
  let wasUp = false;
  
  trace.forEach((point, i) => {
    const x = normalizeX(point.x);
    const y = normalizeY(point.y);
    const isPenDown = point.p === undefined || point.p === 0; // Default to pen down if no p value
    
    
    if (i === 0 || (wasUp && isPenDown)) {
      // Start new path segment
      if (pathStarted) {
        ctx.stroke();
        ctx.beginPath();
      }
      ctx.moveTo(x, y);
      pathStarted = true;
    } else if (isPenDown) {
      // Continue current path
      ctx.lineTo(x, y);
    } else {
      // Pen up - don't draw but keep track
      ctx.moveTo(x, y);
    }
    
    wasUp = !isPenDown;
  });
  
  if (pathStarted) {
    ctx.stroke();
  }
  
  // Reset shadow
  ctx.shadowBlur = 0;
  
  // Draw key highlights for word letters
  if (traceData.word) {
    ctx.fillStyle = 'rgba(102, 126, 234, 0.2)';
    ctx.strokeStyle = '#667eea';
    ctx.lineWidth = 2;
    
    for (const char of traceData.word.toUpperCase()) {
      if (keys[char]) {
        const key = keys[char];
        const x = key.x - key.width / 2;
        const y = key.y - key.height / 2;
        
        ctx.fillRect(x, y, key.width, key.height);
        ctx.strokeRect(x, y, key.width, key.height);
      }
    }
  }
  
  // Draw trace points
  ctx.fillStyle = '#764ba2';
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  
  trace.forEach((point, i) => {
    const x = normalizeX(point.x);
    const y = normalizeY(point.y);
    
    const radius = i === 0 ? 6 : i === trace.length - 1 ? 5 : 2;
    
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    
    if (i === 0 || i === trace.length - 1) {
      ctx.stroke();
    }
  });
}

// Show trace details
function showTraceDetails(traceData) {
  const detailsDiv = document.getElementById('trace-details');
  
  detailsDiv.innerHTML = `
    <h3>Trace Details: ${traceData.word}</h3>
    <table>
      <tr>
        <td>Word</td>
        <td>${traceData.word}</td>
      </tr>
      <tr>
        <td>Points</td>
        <td>${traceData.trace.length}</td>
      </tr>
      <tr>
        <td>Duration</td>
        <td>${((traceData.metadata?.duration || 0) / 1000).toFixed(2)}s</td>
      </tr>
      <tr>
        <td>Generator</td>
        <td>${traceData.metadata?.generator || 'Unknown'}</td>
      </tr>
      <tr>
        <td>Average speed</td>
        <td>${calculateSpeed(traceData.trace).toFixed(1)} px/s</td>
      </tr>
    </table>
  `;
  
  detailsDiv.classList.add('show');
}

// Calculate average speed
function calculateSpeed(trace) {
  if (trace.length < 2) return 0;
  
  let totalDistance = 0;
  let totalTime = 0;
  
  for (let i = 1; i < trace.length; i++) {
    const dx = trace[i].x - trace[i - 1].x;
    const dy = trace[i].y - trace[i - 1].y;
    const dt = (trace[i].t - trace[i - 1].t) / 1000;
    
    totalDistance += Math.sqrt(dx * dx + dy * dy);
    totalTime += dt;
  }
  
  return totalTime > 0 ? totalDistance / totalTime : 0;
}

// Show status message
function showStatus(message, type = 'info') {
  // Try to show in current tab's status area
  const activeTab = document.querySelector('.tab.active')?.dataset.tab;
  let statusDiv;
  
  if (activeTab === 'viewer') {
    // Create or find status div in viewer
    statusDiv = document.getElementById('viewer-status');
    if (!statusDiv) {
      statusDiv = document.createElement('div');
      statusDiv.id = 'viewer-status';
      statusDiv.className = 'status';
      const viewerControls = document.querySelector('#viewer-tab .controls');
      if (viewerControls) {
        viewerControls.parentNode.insertBefore(statusDiv, viewerControls.nextSibling);
      }
    }
  } else {
    statusDiv = document.getElementById('generation-status');
  }
  
  if (statusDiv) {
    statusDiv.textContent = message;
    statusDiv.className = 'status show ' + type;
    
    setTimeout(() => {
      statusDiv.classList.remove('show');
    }, 5000);
  }
}

// Show preview
function showPreview(dataset) {
  const previewDiv = document.getElementById('generation-preview');
  
  const sample = dataset.traces.slice(0, 5);
  previewDiv.innerHTML = '<h3>Preview</h3><pre>' + 
    JSON.stringify(sample, null, 2) + '</pre>';
  
  previewDiv.classList.add('show');
}