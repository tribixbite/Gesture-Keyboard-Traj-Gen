// Client-side only version of the app for GitHub Pages deployment

// App state
let currentDataset = null;
let generatedDataset = null;
let selectedTrace = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupGenerator();
  setupViewer();
  
  // Show static deployment notice
  showStatus('🌐 Static version - Limited functionality. For full features, run the local server.', 'info');
  
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
        const fallbackWords = ['hello', 'world', 'test', 'example', 'demo', 'sample', 'keyboard', 'swipe', 'gesture', 'trace', 'the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they'];
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
  
  // Handle wordlist file upload (client-side parsing)
  wordlistFile.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const content = await file.text();
    const format = document.getElementById('wordlist-format').value;
    
    try {
      const words = parseWordlistClientSide(content, format);
      wordlistFile.dataset.words = JSON.stringify(words);
      showStatus('Loaded ' + words.length + ' words', 'success');
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
  
  // Generate dataset (client-side simulation)
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
    showStatus('Generating dataset (client-side simulation)...', 'info');
    
    try {
      // Simulate dataset generation
      generatedDataset = await generateDatasetClientSide(config);
      
      showStatus('Generated ' + generatedDataset.traces.length + ' traces (simulated)', 'success');
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

// Client-side wordlist parsing
function parseWordlistClientSide(content, format) {
  const words = [];
  
  switch (format) {
    case "plain":
    case "auto":
    default:
      // Split by lines and filter empty
      content.split(/\r?\n/).forEach(line => {
        const word = line.trim();
        if (word) {
          words.push(word);
        }
      });
      break;
      
    case "frequency":
      // Format: word\tfrequency or word frequency
      content.split(/\r?\n/).forEach(line => {
        const parts = line.split(/\s+/);
        if (parts.length >= 1 && parts[0].trim()) {
          words.push(parts[0].trim());
        }
      });
      break;
      
    case "csv":
      // CSV with word in first column
      content.split(/\r?\n/).forEach(line => {
        const [word] = line.split(/,/);
        if (word && word.trim()) {
          words.push(word.trim().replace(/^["']|["']$/g, ''));
        }
      });
      break;
  }
  
  return words.filter(w => w.length > 0);
}

// Load wordlist from URL (client-side)
async function loadWordlistFromUrl(url) {
  if (!url) return;
  
  try {
    showStatus('Loading wordlist from URL...', 'info');
    
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(\`HTTP \${response.status}: \${response.statusText}\`);
    }
    
    const content = await response.text();
    const format = document.getElementById('wordlist-format').value;
    const words = parseWordlistClientSide(content, format);
    
    const wordlistFile = document.getElementById('wordlist-file');
    wordlistFile.dataset.words = JSON.stringify(words);
    showStatus('Loaded ' + words.length + ' words from URL', 'success');
    
  } catch (error) {
    console.error('URL loading error:', error);
    showStatus('Error loading wordlist from URL: ' + error.message, 'error');
    throw error;
  }
}

// Client-side dataset generation (simplified simulation)
async function generateDatasetClientSide(config) {
  const { words, options, generator } = config;
  
  // Filter words by length
  let filteredWords = words.filter(word => 
    word.length >= options.minLength && word.length <= options.maxLength
  );
  
  // Remove duplicates if requested
  if (options.noDuplicates) {
    filteredWords = [...new Set(filteredWords)];
  }
  
  // Randomize if requested
  if (options.random) {
    filteredWords = filteredWords.sort(() => Math.random() - 0.5);
  }
  
  // Limit quantity
  filteredWords = filteredWords.slice(0, options.maxQuantity);
  
  // Generate traces (simplified simulation)
  const traces = filteredWords.map(word => {
    const trace = generateSimulatedTrace(word, generator);
    return {
      word: word,
      trace: trace,
      metadata: {
        generator: generator.type + ' (simulated)',
        duration: trace.length * 50, // 50ms per point
      }
    };
  });
  
  return {
    traces: traces,
    generator: generator.type + ' (client-side simulation)',
    timestamp: new Date().toISOString(),
    words: filteredWords.length
  };
}

// Generate a simulated trace for a word
function generateSimulatedTrace(word, generator) {
  const trace = [];
  const keys = getKeyboardCoordinates(700, 400); // Standard canvas size
  
  let currentTime = 0;
  
  for (let i = 0; i < word.length; i++) {
    const char = word[i].toUpperCase();
    const key = keys[char];
    
    if (key) {
      // Add some random variation based on generator type
      const variation = getGeneratorVariation(generator.type);
      const x = key.x + (Math.random() - 0.5) * variation;
      const y = key.y + (Math.random() - 0.5) * variation;
      
      // Add intermediate points for smoother curves
      if (i > 0) {
        const prevTrace = trace[trace.length - 1];
        const steps = 3 + Math.floor(Math.random() * 3);
        
        for (let j = 1; j < steps; j++) {
          const t = j / steps;
          const interpX = prevTrace.x + t * (x - prevTrace.x) + (Math.random() - 0.5) * 10;
          const interpY = prevTrace.y + t * (y - prevTrace.y) + (Math.random() - 0.5) * 10;
          
          trace.push({
            x: interpX,
            y: interpY,
            t: currentTime + (j * 30),
            p: 0 // Pen down
          });
        }
        currentTime += steps * 30;
      }
      
      // Add the key point
      trace.push({
        x: x,
        y: y,
        t: currentTime,
        p: 0 // Pen down
      });
      
      currentTime += 50 + Math.random() * 50; // Dwell time
    }
  }
  
  return trace;
}

function getGeneratorVariation(type) {
  switch (type) {
    case 'rnn': return 15;
    case 'gan': return 20;
    case 'transformer': return 10;
    case 'rule-based': return 5;
    default: return 10;
  }
}

// The rest of the functions remain the same as in the original server version...
// (I'll include the essential ones for the viewer functionality)

function setupViewer() {
  const datasetFile = document.getElementById('dataset-file');
  const filterInput = document.getElementById('filter-input');
  const sortBy = document.getElementById('sort-by');
  
  // Handle dataset upload
  datasetFile.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    showStatus('Loading dataset...', 'info');
    
    try {
      if (!file.name.toLowerCase().endsWith('.json')) {
        throw new Error('Please select a JSON file (.json)');
      }
      
      if (file.size > 10 * 1024 * 1024) {
        throw new Error('File is too large. Maximum size is 10MB.');
      }
      
      const content = await file.text();
      let dataset;
      try {
        dataset = JSON.parse(content);
      } catch (parseError) {
        throw new Error('Invalid JSON format. Please check your file.');
      }
      
      if (!validateDatasetStructure(dataset)) {
        throw new Error('Invalid dataset format. Expected structure: {traces: [{word, trace: [{x, y, t, p}]}]}');
      }
      
      currentDataset = dataset;
      displayDataset(currentDataset);
      showStatus(\`Successfully loaded \${dataset.traces?.length || 0} traces\`, 'success');
      
    } catch (error) {
      console.error('Dataset loading error:', error);
      showStatus('Error loading dataset: ' + error.message, 'error');
      datasetFile.value = '';
    }
  });
  
  filterInput.addEventListener('input', () => displayDataset(currentDataset));
  sortBy.addEventListener('change', () => displayDataset(currentDataset));
}

// Include essential viewer functions...
function validateDatasetStructure(dataset) {
  try {
    if (!dataset || typeof dataset !== 'object') return false;
    if (!Array.isArray(dataset.traces)) return false;
    if (dataset.traces.length === 0) return false;
    
    const samplesToCheck = Math.min(5, dataset.traces.length);
    for (let i = 0; i < samplesToCheck; i++) {
      const trace = dataset.traces[i];
      if (!trace || typeof trace !== 'object') return false;
      if (typeof trace.word !== 'string' || !Array.isArray(trace.trace)) return false;
      if (trace.trace.length > 0) {
        const point = trace.trace[0];
        if (!point || typeof point !== 'object') return false;
        if (typeof point.x !== 'number' || typeof point.y !== 'number') return false;
      }
    }
    return true;
  } catch (error) {
    return false;
  }
}

function displayDataset(dataset) {
  if (!dataset) return;
  
  const infoDiv = document.getElementById('dataset-info');
  infoDiv.innerHTML = \`
    <div class="info-item">
      <div class="value">\${dataset.traces.length}</div>
      <div class="label">Traces</div>
    </div>
    <div class="info-item">
      <div class="value">\${dataset.words || dataset.traces.length}</div>
      <div class="label">Words</div>
    </div>
    <div class="info-item">
      <div class="value">\${dataset.generator || 'Unknown'}</div>
      <div class="label">Generator</div>
    </div>
  \`;
  
  const filterText = document.getElementById('filter-input').value.toLowerCase();
  const sortKey = document.getElementById('sort-by').value;
  
  let traces = dataset.traces.filter(t => 
    !filterText || t.word.toLowerCase().includes(filterText)
  );
  
  traces.sort((a, b) => {
    switch (sortKey) {
      case 'word': return a.word.localeCompare(b.word);
      case 'length': return a.word.length - b.word.length;
      case 'points': return a.trace.length - b.trace.length;
      default: return 0;
    }
  });
  
  const listDiv = document.getElementById('trace-list');
  listDiv.innerHTML = traces.slice(0, 50).map((trace, i) => \`
    <div class="trace-item" data-index="\${i}">
      <div class="word">\${trace.word}</div>
      <div class="meta">
        \${trace.trace.length} points · 
        \${((trace.metadata?.duration || 0) / 1000).toFixed(1)}s
      </div>
    </div>
  \`).join('');
  
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
  
  if (traces.length > 0) {
    document.querySelector('.trace-item').click();
  }
}

// Include the full QWERTY keyboard and drawing functions from the original...
// (copying the essential functions)

const QWERTY_LAYOUT = [
  ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
  ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
  ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
];

function getKeyboardCoordinates(width, height) {
  const keys = {};
  const keyWidth = width / 10;
  const keyHeight = height / 4;
  const startY = keyHeight * 0.2;
  
  QWERTY_LAYOUT[0].forEach((key, i) => {
    keys[key] = {
      x: i * keyWidth + keyWidth / 2,
      y: startY + keyHeight / 2,
      width: keyWidth * 0.9,
      height: keyHeight * 0.8
    };
  });
  
  const middleOffset = keyWidth * 0.25;
  QWERTY_LAYOUT[1].forEach((key, i) => {
    keys[key] = {
      x: middleOffset + i * keyWidth + keyWidth / 2,
      y: startY + keyHeight * 1.2 + keyHeight / 2,
      width: keyWidth * 0.9,
      height: keyHeight * 0.8
    };
  });
  
  const bottomOffset = keyWidth * 0.75;
  QWERTY_LAYOUT[2].forEach((key, i) => {
    keys[key] = {
      x: bottomOffset + i * keyWidth + keyWidth / 2,
      y: startY + keyHeight * 2.4 + keyHeight / 2,
      width: keyWidth * 0.9,
      height: keyHeight * 0.8
    };
  });
  
  keys[' '] = {
    x: width / 2,
    y: startY + keyHeight * 3.6 + keyHeight / 2,
    width: keyWidth * 4,
    height: keyHeight * 0.6
  };
  
  return keys;
}

function drawKeyboard(canvas, ctx) {
  const width = canvas.width;
  const height = canvas.height;
  const keys = getKeyboardCoordinates(width, height);
  
  ctx.fillStyle = '#1a1a1a';
  ctx.fillRect(0, 0, width, height);
  
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#2a2a2a';
  ctx.font = '12px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  
  Object.entries(keys).forEach(([key, pos]) => {
    const x = pos.x - pos.width / 2;
    const y = pos.y - pos.height / 2;
    
    ctx.fillStyle = '#2a2a2a';
    ctx.fillRect(x, y, pos.width, pos.height);
    ctx.strokeRect(x, y, pos.width, pos.height);
    
    ctx.fillStyle = '#888';
    ctx.fillText(key === ' ' ? 'SPACE' : key, pos.x, pos.y);
  });
  
  return keys;
}

function drawTrace(traceData) {
  const canvas = document.getElementById('trace-canvas');
  if (!canvas) {
    console.error('Canvas not found!');
    return;
  }
  const ctx = canvas.getContext('2d');
  const trace = traceData.trace;
  
  const displayWidth = canvas.clientWidth;
  const displayHeight = displayWidth / 1.4;
  
  if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
    canvas.width = displayWidth;
    canvas.height = displayHeight;
  }
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const keys = drawKeyboard(canvas, ctx);
  
  if (trace.length < 2) return;
  
  const minX = Math.min(...trace.map(p => p.x));
  const maxX = Math.max(...trace.map(p => p.x));
  const minY = Math.min(...trace.map(p => p.y));
  const maxY = Math.max(...trace.map(p => p.y));
  const rangeX = maxX - minX;
  const rangeY = maxY - minY;
  
  let normalizeX, normalizeY;
  
  if (maxX <= canvas.width && maxY <= canvas.height && minX >= 0 && minY >= 0) {
    normalizeX = (x) => x;
    normalizeY = (y) => y;
  } else if (maxX <= 1 && maxY <= 1 && minX >= 0 && minY >= 0) {
    normalizeX = (x) => x * canvas.width;
    normalizeY = (y) => y * canvas.height;
  } else {
    const padding = 40;
    normalizeX = (x) => padding + ((x - minX) / rangeX) * (canvas.width - 2 * padding);
    normalizeY = (y) => padding + ((y - minY) / rangeY) * (canvas.height - 2 * padding);
  }
  
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
    const isPenDown = point.p === undefined || point.p === 0;
    
    if (i === 0 || (wasUp && isPenDown)) {
      if (pathStarted) {
        ctx.stroke();
        ctx.beginPath();
      }
      ctx.moveTo(x, y);
      pathStarted = true;
    } else if (isPenDown) {
      ctx.lineTo(x, y);
    } else {
      ctx.moveTo(x, y);
    }
    
    wasUp = !isPenDown;
  });
  
  if (pathStarted) {
    ctx.stroke();
  }
  
  ctx.shadowBlur = 0;
  
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

function showTraceDetails(traceData) {
  const detailsDiv = document.getElementById('trace-details');
  
  detailsDiv.innerHTML = \`
    <h3>Trace Details: \${traceData.word}</h3>
    <table>
      <tr><td>Word</td><td>\${traceData.word}</td></tr>
      <tr><td>Points</td><td>\${traceData.trace.length}</td></tr>
      <tr><td>Duration</td><td>\${((traceData.metadata?.duration || 0) / 1000).toFixed(2)}s</td></tr>
      <tr><td>Generator</td><td>\${traceData.metadata?.generator || 'Unknown'}</td></tr>
    </table>
  \`;
  
  detailsDiv.classList.add('show');
}

// Utility functions
function updateGeneratorSettings() {
  const type = document.getElementById('generator-type').value;
  const settingsDiv = document.getElementById('generator-settings');
  
  let html = '<label>Settings</label><div class="options">';
  
  switch (type) {
    case 'rnn':
      html += \`
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
      \`;
      break;
    case 'gan':
      html += \`
        <div class="input-group">
          <label>Noise dimension</label>
          <input type="number" id="gan-noise-dim" value="100" min="50" max="200">
        </div>
        <div class="input-group">
          <label>Style</label>
          <select id="gan-style">
            <option value="natural">Natural</option>
            <option value="smooth">Smooth</option>
            <option value="sharp">Sharp</option>
          </select>
        </div>
      \`;
      break;
    case 'transformer':
      html += \`
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
      \`;
      break;
    case 'rule-based':
      html += \`
        <div class="input-group">
          <label>Smoothness</label>
          <input type="range" id="rule-smoothness" min="0" max="1" step="0.1" value="0.7">
        </div>
        <div class="input-group">
          <label>Noise level</label>
          <input type="range" id="rule-noise" min="0" max="1" step="0.1" value="0.2">
        </div>
      \`;
      break;
  }
  
  html += '</div>';
  settingsDiv.innerHTML = html;
}

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

function showStatus(message, type = 'info') {
  const activeTab = document.querySelector('.tab.active')?.dataset.tab;
  let statusDiv;
  
  if (activeTab === 'viewer') {
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

function showPreview(dataset) {
  const previewDiv = document.getElementById('generation-preview');
  
  const sample = dataset.traces.slice(0, 5);
  previewDiv.innerHTML = '<h3>Preview</h3><pre>' + 
    JSON.stringify(sample, null, 2) + '</pre>';
  
  previewDiv.classList.add('show');
}