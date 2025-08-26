#!/usr/bin/env node
/**
 * Node.js-based web server for Swipe Dataset Studio
 * Alternative to Bun server due to compatibility issues
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

// Try to find an available port
const PORT = process.env.PORT || findAvailablePort();

function findAvailablePort(startPort = 3000) {
  const net = require('net');
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(startPort, (err) => {
      if (err) {
        server.listen(startPort + 1, (err) => {
          if (err) {
            server.listen(9999, (err) => {
              if (err) {
                resolve(8080); // Last resort
              } else {
                const port = server.address().port;
                server.close();
                resolve(port);
              }
            });
          } else {
            const port = server.address().port;
            server.close();
            resolve(port);
          }
        });
      } else {
        const port = server.address().port;
        server.close();
        resolve(port);
      }
    });
  });
}

// Ensure directories exist
const datasetsDir = path.join(__dirname, "datasets");
if (!fs.existsSync(datasetsDir)) {
  fs.mkdirSync(datasetsDir, { recursive: true });
}

// Create server
const server = http.createServer(async (req, res) => {
  const parsedUrl = url.parse(req.url, true);
  
  // Set CORS headers
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  
  if (req.method === "OPTIONS") {
    res.writeHead(200);
    res.end();
    return;
  }
  
  // API Routes
  if (parsedUrl.pathname.startsWith("/api")) {
    await handleAPI(req, res, parsedUrl);
    return;
  }
  
  // Static files
  if (parsedUrl.pathname === "/" || parsedUrl.pathname === "/index.html") {
    res.writeHead(200, { "Content-Type": "text/html" });
    res.end(getIndexHTML());
    return;
  }
  
  if (parsedUrl.pathname === "/app.js") {
    res.writeHead(200, { "Content-Type": "application/javascript" });
    res.end(getAppJS());
    return;
  }
  
  if (parsedUrl.pathname === "/styles.css") {
    res.writeHead(200, { "Content-Type": "text/css" });
    res.end(getStyles());
    return;
  }
  
  res.writeHead(404);
  res.end("Not Found");
});

// API Handler
async function handleAPI(req, res, parsedUrl) {
  const headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
  };
  
  try {
    let body = '';
    if (req.method === 'POST') {
      for await (const chunk of req) {
        body += chunk.toString();
      }
      body = JSON.parse(body);
    }
    
    // Generate dataset endpoint
    if (parsedUrl.pathname === "/api/generate" && req.method === "POST") {
      const dataset = await generateDataset(body);
      res.writeHead(200, headers);
      res.end(JSON.stringify(dataset));
      return;
    }
    
    // Parse wordlist endpoint
    if (parsedUrl.pathname === "/api/parse-wordlist" && req.method === "POST") {
      const words = await parseWordlist(body.content, body.format);
      res.writeHead(200, headers);
      res.end(JSON.stringify({ words }));
      return;
    }
    
    // Save dataset endpoint
    if (parsedUrl.pathname === "/api/save-dataset" && req.method === "POST") {
      const filename = `dataset_${Date.now()}.json`;
      const filepath = path.join(datasetsDir, filename);
      fs.writeFileSync(filepath, JSON.stringify(body.dataset, null, 2));
      res.writeHead(200, headers);
      res.end(JSON.stringify({ filename, success: true }));
      return;
    }
    
    res.writeHead(404, headers);
    res.end(JSON.stringify({ error: "Not Found" }));
  } catch (error) {
    res.writeHead(500, headers);
    res.end(JSON.stringify({ error: error.message }));
  }
}

// Dataset generation logic
async function generateDataset(config) {
  const { generator, words, options } = config;
  
  // Filter words based on options
  let filteredWords = words;
  if (options.minLength) {
    filteredWords = filteredWords.filter(w => w.length >= options.minLength);
  }
  if (options.maxLength) {
    filteredWords = filteredWords.filter(w => w.length <= options.maxLength);
  }
  if (options.noDuplicates) {
    filteredWords = [...new Set(filteredWords)];
  }
  if (options.random && options.maxQuantity) {
    filteredWords = filteredWords
      .sort(() => Math.random() - 0.5)
      .slice(0, options.maxQuantity);
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
function generateTrace(word, generator) {
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
async function parseWordlist(content, format) {
  const words = [];
  
  switch (format) {
    case "plain":
      words.push(...content.split(/\r?\n/).filter(w => w.trim()));
      break;
      
    case "frequency":
      content.split(/\r?\n/).forEach(line => {
        const [word] = line.split(/\t/);
        if (word && word.trim()) {
          words.push(word.trim());
        }
      });
      break;
      
    case "csv":
      content.split(/\r?\n/).forEach(line => {
        const [word] = line.split(/,/);
        if (word && word.trim()) {
          words.push(word.trim().replace(/^["']|["']$/g, ''));
        }
      });
      break;
      
    default:
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

// HTML/CSS/JS content (reuse from original server.ts)
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
          
          <div class="control-group">
            <label>Status</label>
            <div id="port-status">Running on port ${PORT || 3000}</div>
          </div>
          
          <div class="actions">
            <button id="test-btn" class="primary">Test Connection</button>
          </div>
        </div>
        
        <div id="generation-status" class="status"></div>
      </div>
      
      <!-- Viewer Tab -->
      <div id="viewer-tab" class="tab-content">
        <p>Dataset viewer functionality</p>
      </div>
    </main>
  </div>
  
  <script src="/app.js"></script>
</body>
</html>`;
}

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

.tab.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-color: transparent;
}

.tab-content {
  display: none;
}

.tab-content.active {
  display: block;
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
  color: #888;
  font-weight: 500;
}

select, input {
  width: 100%;
  padding: 10px;
  background: #0a0a0a;
  border: 1px solid #333;
  border-radius: 6px;
  color: #e0e0e0;
  font-size: 14px;
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
  transition: all 0.3s;
}

button.primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

#port-status {
  padding: 10px;
  background: #0a0a0a;
  border: 1px solid #333;
  border-radius: 6px;
  color: #667eea;
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
`;
}

function getAppJS() {
  return `
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupTestButton();
});

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

function setupTestButton() {
  document.getElementById('test-btn').addEventListener('click', async () => {
    const statusDiv = document.getElementById('generation-status');
    statusDiv.textContent = 'Testing server connection...';
    statusDiv.className = 'status show';
    
    try {
      const response = await fetch('/api/parse-wordlist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: 'test\\nhello\\nworld', format: 'plain' })
      });
      
      const data = await response.json();
      statusDiv.textContent = 'Server connected successfully! Parsed ' + data.words.length + ' words.';
      statusDiv.className = 'status show success';
    } catch (error) {
      statusDiv.textContent = 'Connection failed: ' + error.message;
      statusDiv.className = 'status show error';
    }
  });
}
`;
}

// Start server
async function startServer() {
  const actualPort = typeof PORT === 'number' ? PORT : await PORT;
  server.listen(actualPort, () => {
    console.log(`
🚀 Swipe Dataset Studio Server (Node.js)
📡 Running on http://localhost:${actualPort}
🎯 Server resolved port conflicts and is ready!

Press Ctrl+C to stop the server
`);
  });
}

startServer();