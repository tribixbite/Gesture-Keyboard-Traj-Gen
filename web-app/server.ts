import { serve } from "bun";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
// Let system choose an available port
const PORT = parseInt(process.env.PORT || "0");

// Ensure directories exist
if (!existsSync(join(__dirname, "datasets"))) {
  mkdirSync(join(__dirname, "datasets"), { recursive: true });
}

// Serve the app
const server = serve({
  port: PORT,
  
  async fetch(req) {
    const url = new URL(req.url);
    
    // API Routes
    if (url.pathname.startsWith("/api")) {
      return handleAPI(req, url);
    }
    
    // Static files
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(getIndexHTML(), {
        headers: { "Content-Type": "text/html" }
      });
    }
    
    if (url.pathname === "/app.js") {
      return new Response(getAppJS(), {
        headers: { "Content-Type": "application/javascript" }
      });
    }
    
    if (url.pathname === "/styles.css") {
      return new Response(getStyles(), {
        headers: { "Content-Type": "text/css" }
      });
    }
    
    return new Response("Not Found", { status: 404 });
  }
});

// API Handler
async function handleAPI(req: Request, url: URL) {
  const headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
  };
  
  if (req.method === "OPTIONS") {
    return new Response(null, { headers });
  }
  
  // Generate dataset endpoint
  if (url.pathname === "/api/generate" && req.method === "POST") {
    try {
      const body = await req.json();
      const dataset = await generateDataset(body);
      return new Response(JSON.stringify(dataset), { headers });
    } catch (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers
      });
    }
  }
  
  // Parse wordlist endpoint
  if (url.pathname === "/api/parse-wordlist" && req.method === "POST") {
    try {
      const body = await req.json();
      const words = await parseWordlist(body.content, body.format);
      return new Response(JSON.stringify({ words }), { headers });
    } catch (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers
      });
    }
  }
  
  // Save dataset endpoint
  if (url.pathname === "/api/save-dataset" && req.method === "POST") {
    try {
      const body = await req.json();
      const filename = `dataset_${Date.now()}.json`;
      const filepath = join(__dirname, "datasets", filename);
      writeFileSync(filepath, JSON.stringify(body.dataset, null, 2));
      return new Response(JSON.stringify({ filename, success: true }), { headers });
    } catch (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers
      });
    }
  }
  
  return new Response(JSON.stringify({ error: "Not Found" }), {
    status: 404,
    headers
  });
}

// Dataset generation logic
async function generateDataset(config: any) {
  const { generator, words, options } = config;
  
  // Filter words based on options
  let filteredWords = words;
  if (options.minLength) {
    filteredWords = filteredWords.filter((w: string) => w.length >= options.minLength);
  }
  if (options.maxLength) {
    filteredWords = filteredWords.filter((w: string) => w.length <= options.maxLength);
  }
  if (options.noDuplicates) {
    filteredWords = [...new Set(filteredWords)];
  }
  
  // Apply randomization if requested
  if (options.random) {
    filteredWords = filteredWords.sort(() => Math.random() - 0.5);
  }
  
  // Apply max quantity limit (regardless of random option)
  if (options.maxQuantity && options.maxQuantity > 0) {
    filteredWords = filteredWords.slice(0, options.maxQuantity);
  }
  
  // Generate traces based on generator type
  const traces = [];
  for (const word of filteredWords) {
    const trace = generateTrace(word, generator);
    traces.push({
      word,
      trace,
      metadata: {
        generator: generator.type,
        timestamp: Date.now(),
        duration: trace.length * 16.67, // Approximate ms
        points: trace.length
      }
    });
  }
  
  return {
    version: "1.0",
    generator: generator.type,
    timestamp: Date.now(),
    words: filteredWords.length,
    traces
  };
}

// Simple trace generation (replace with actual model inference)
function generateTrace(word: string, generator: any) {
  const points = [];
  const numPoints = word.length * 10 + Math.random() * 20;
  
  // Simple path through keyboard positions
  for (let i = 0; i < numPoints; i++) {
    const t = i / numPoints;
    points.push({
      x: Math.sin(t * Math.PI * 2) * 100 + 180,
      y: Math.cos(t * Math.PI * word.length) * 50 + 400,
      t: i * 16.67, // Timestamp in ms
      p: i === numPoints - 1 ? 0 : 1 // Pen state
    });
  }
  
  return points;
}

// Parse wordlist from various formats
async function parseWordlist(content: string, format: string) {
  const words: string[] = [];
  
  switch (format) {
    case "plain":
      // One word per line
      words.push(...content.split(/\r?\n/).filter(w => w.trim()));
      break;
      
    case "frequency":
      // Word<tab>frequency format
      content.split(/\r?\n/).forEach(line => {
        const [word] = line.split(/\t/);
        if (word && word.trim()) {
          words.push(word.trim());
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
      
    default:
      // Try to auto-detect
      if (content.includes('\t')) {
        return parseWordlist(content, "frequency");
      } else if (content.includes(',')) {
        return parseWordlist(content, "csv");
      } else {
        return parseWordlist(content, "plain");
      }
  }
  
  return words.filter(w => w.length > 0);
}

// HTML Template
function getIndexHTML() {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Swipe Dataset Studio</title>
  <link rel="stylesheet" href="/styles.css">
</head>
<body>
  <div id="app">
    <header>
      <h1>🎯 Swipe Dataset Studio</h1>
      <div class="tabs">
        <button class="tab active" data-tab="generator">Generator</button>
        <button class="tab" data-tab="viewer">Viewer</button>
      </div>
    </header>
    
    <main>
      <!-- Generator Tab -->
      <div id="generator-tab" class="tab-content active">
        <div class="controls">
          <div class="control-group">
            <label>Generator Type</label>
            <select id="generator-type">
              <option value="rnn">RNN (LSTM)</option>
              <option value="gan">GAN (WGAN-GP)</option>
              <option value="transformer">Transformer.js</option>
              <option value="rule-based">Rule-based</option>
            </select>
          </div>
          
          <div id="generator-settings" class="control-group">
            <!-- Dynamic settings based on generator -->
          </div>
          
          <div class="control-group">
            <label>Wordlist</label>
            <div class="file-input">
              <input type="file" id="wordlist-file" accept=".txt,.csv,.tsv">
              <span>Choose file or enter URL</span>
            </div>
            <input type="url" id="wordlist-url" placeholder="https://example.com/wordlist.txt" value="https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/google-10000-english.txt">
            <select id="wordlist-format">
              <option value="auto">Auto-detect</option>
              <option value="plain">Plain text</option>
              <option value="frequency">Word + Frequency</option>
              <option value="csv">CSV</option>
            </select>
          </div>
          
          <div class="control-group">
            <label>Output Options</label>
            <div class="options">
              <label class="checkbox">
                <input type="checkbox" id="no-duplicates" checked>
                <span>No duplicates</span>
              </label>
              <label class="checkbox">
                <input type="checkbox" id="random-order">
                <span>Random order</span>
              </label>
              <div class="input-group">
                <label>Min length</label>
                <input type="number" id="min-length" value="2" min="1">
              </div>
              <div class="input-group">
                <label>Max length</label>
                <input type="number" id="max-length" value="10" min="1">
              </div>
              <div class="input-group">
                <label>Max quantity</label>
                <input type="number" id="max-quantity" value="100" min="1">
              </div>
            </div>
          </div>
          
          <div class="control-group">
            <label>Keyboard Layout</label>
            <select id="keyboard-layout">
              <option value="qwerty">QWERTY</option>
              <option value="qwertz">QWERTZ</option>
              <option value="azerty">AZERTY</option>
              <option value="custom">Custom</option>
            </select>
          </div>
          
          <div class="actions">
            <button id="generate-btn" class="primary">Generate Dataset</button>
            <button id="export-btn" disabled>Export</button>
            <button id="view-generated-btn" disabled>View Generated</button>
          </div>
        </div>
        
        <div id="generation-status" class="status"></div>
        <div id="generation-preview" class="preview"></div>
      </div>
      
      <!-- Viewer Tab -->
      <div id="viewer-tab" class="tab-content">
        <div class="controls">
          <div class="control-group">
            <label>Upload Dataset</label>
            <div class="file-input">
              <input type="file" id="dataset-file" accept=".json">
              <span>Choose dataset file</span>
            </div>
          </div>
          
          <div class="control-group">
            <label>Filter</label>
            <input type="text" id="filter-input" placeholder="Filter by word...">
            <select id="sort-by">
              <option value="word">Sort by word</option>
              <option value="length">Sort by length</option>
              <option value="points">Sort by points</option>
              <option value="duration">Sort by duration</option>
            </select>
          </div>
        </div>
        
        <div id="dataset-info" class="info"></div>
        <div id="trace-list" class="trace-list"></div>
        <canvas id="trace-canvas" width="560" height="400"></canvas>
        <div id="trace-details" class="details"></div>
      </div>
    </main>
  </div>
  
  <script src="/app.js"></script>
</body>
</html>`;
}

// CSS Styles
function getStyles() {
  return `
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #0a0a0a;
  color: #e0e0e0;
  min-height: 100vh;
}

#app {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}

header {
  margin-bottom: 30px;
  border-bottom: 1px solid #2a2a2a;
  padding-bottom: 20px;
}

h1 {
  font-size: 28px;
  font-weight: 600;
  margin-bottom: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.tabs {
  display: flex;
  gap: 10px;
}

.tab {
  padding: 10px 20px;
  background: #1a1a1a;
  border: 1px solid #2a2a2a;
  border-radius: 8px;
  color: #888;
  cursor: pointer;
  transition: all 0.3s;
  font-size: 14px;
}

.tab:hover {
  background: #222;
  color: #aaa;
}

.tab.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-color: transparent;
}

.tab-content {
  display: none;
  animation: fadeIn 0.3s;
}

.tab-content.active {
  display: block;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.controls {
  background: #1a1a1a;
  padding: 25px;
  border-radius: 12px;
  margin-bottom: 20px;
  border: 1px solid #2a2a2a;
}

.control-group {
  margin-bottom: 20px;
}

.control-group label {
  display: block;
  margin-bottom: 8px;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #888;
  font-weight: 500;
}

select, input[type="text"], input[type="url"], input[type="number"] {
  width: 100%;
  padding: 10px;
  background: #0a0a0a;
  border: 1px solid #333;
  border-radius: 6px;
  color: #e0e0e0;
  font-size: 14px;
  transition: border-color 0.3s;
}

select:focus, input:focus {
  outline: none;
  border-color: #667eea;
}

.file-input {
  position: relative;
  display: inline-block;
  width: 100%;
  margin-bottom: 10px;
}

.file-input input[type="file"] {
  position: absolute;
  opacity: 0;
  width: 100%;
  height: 100%;
  cursor: pointer;
}

.file-input span {
  display: block;
  padding: 10px;
  background: #0a0a0a;
  border: 2px dashed #333;
  border-radius: 6px;
  text-align: center;
  color: #888;
  cursor: pointer;
  transition: all 0.3s;
}

.file-input:hover span {
  border-color: #667eea;
  color: #667eea;
}

.options {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
  margin-top: 10px;
}

.checkbox {
  display: flex;
  align-items: center;
  cursor: pointer;
  font-size: 14px;
}

.checkbox input {
  margin-right: 8px;
  width: auto;
}

.checkbox span {
  color: #aaa;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.input-group label {
  font-size: 12px;
  margin: 0;
}

.input-group input {
  width: 100%;
}

.actions {
  display: flex;
  gap: 10px;
  margin-top: 25px;
}

button {
  padding: 12px 24px;
  background: #2a2a2a;
  border: none;
  border-radius: 6px;
  color: #e0e0e0;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s;
}

button:hover:not(:disabled) {
  background: #333;
}

button.primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

button.primary:hover {
  opacity: 0.9;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.status {
  padding: 15px;
  background: #1a1a1a;
  border-radius: 8px;
  margin-bottom: 20px;
  border-left: 3px solid #667eea;
  display: none;
}

.status.show {
  display: block;
}

.preview {
  background: #1a1a1a;
  padding: 20px;
  border-radius: 8px;
  max-height: 400px;
  overflow-y: auto;
  display: none;
}

.preview.show {
  display: block;
}

.preview pre {
  background: #0a0a0a;
  padding: 15px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 12px;
  line-height: 1.5;
  color: #aaa;
}

.info {
  background: #1a1a1a;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
}

.info-item {
  text-align: center;
}

.info-item .value {
  font-size: 24px;
  font-weight: bold;
  color: #667eea;
}

.info-item .label {
  font-size: 12px;
  color: #888;
  text-transform: uppercase;
  margin-top: 5px;
}

.trace-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
}

.trace-item {
  background: #1a1a1a;
  padding: 15px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s;
  border: 1px solid #2a2a2a;
}

.trace-item:hover {
  border-color: #667eea;
  transform: translateY(-2px);
}

.trace-item.selected {
  border-color: #667eea;
  background: #222;
}

.trace-item .word {
  font-size: 18px;
  font-weight: 500;
  margin-bottom: 8px;
}

.trace-item .meta {
  font-size: 12px;
  color: #888;
}

#trace-canvas {
  width: 100%;
  max-width: 700px;
  height: auto;
  aspect-ratio: 1.4;
  background: #0a0a0a;
  border-radius: 8px;
  border: 1px solid #2a2a2a;
  margin: 20px auto;
  display: block;
}

.details {
  background: #1a1a1a;
  padding: 20px;
  border-radius: 8px;
  display: none;
}

.details.show {
  display: block;
}

.details h3 {
  margin-bottom: 15px;
  color: #667eea;
}

.details table {
  width: 100%;
  border-collapse: collapse;
}

.details td {
  padding: 8px;
  border-bottom: 1px solid #2a2a2a;
}

.details td:first-child {
  color: #888;
  width: 150px;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #1a1a1a;
}

::-webkit-scrollbar-thumb {
  background: #333;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #444;
}`;
}

// JavaScript Application
function getAppJS() {
  return `
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
    setTimeout(() => loadWordlistFromUrl(defaultUrl), 100);
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
      html += \`
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
      \`;
      break;
      
    case 'gan':
      html += \`
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
      \`;
      break;
      
    case 'transformer':
      html += \`
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
      \`;
      break;
      
    case 'rule-based':
      html += \`
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
      \`;
      break;
  }
  
  settingsDiv.innerHTML = html;
}

// Load wordlist from URL
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
      showStatus(\`Successfully loaded \${dataset.traces?.length || 0} traces\`, 'success');
      
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
    <div class="info-item">
      <div class="value">\${new Date(dataset.timestamp).toLocaleDateString()}</div>
      <div class="label">Generated</div>
    </div>
  \`;
  
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
  listDiv.innerHTML = traces.slice(0, 50).map((trace, i) => \`
    <div class="trace-item" data-index="\${i}">
      <div class="word">\${trace.word}</div>
      <div class="meta">
        \${trace.trace.length} points · 
        \${((trace.metadata?.duration || 0) / 1000).toFixed(1)}s
      </div>
    </div>
  \`).join('');
  
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
  
  detailsDiv.innerHTML = \`
    <h3>Trace Details: \${traceData.word}</h3>
    <table>
      <tr>
        <td>Word</td>
        <td>\${traceData.word}</td>
      </tr>
      <tr>
        <td>Points</td>
        <td>\${traceData.trace.length}</td>
      </tr>
      <tr>
        <td>Duration</td>
        <td>\${((traceData.metadata?.duration || 0) / 1000).toFixed(2)}s</td>
      </tr>
      <tr>
        <td>Generator</td>
        <td>\${traceData.metadata?.generator || 'Unknown'}</td>
      </tr>
      <tr>
        <td>Average speed</td>
        <td>\${calculateSpeed(traceData.trace).toFixed(1)} px/s</td>
      </tr>
    </table>
  \`;
  
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
}`;
}

console.log(`
🚀 Swipe Dataset Studio Server
📡 Running on http://localhost:${server.port}
🎯 Features:
  - Generate datasets with RNN/GAN/Transformer
  - Parse various wordlist formats
  - Visualize swipe trajectories
  - Export/import datasets
  
Press Ctrl+C to stop the server
`);