#!/usr/bin/env bun

import { writeFileSync, mkdirSync, existsSync, copyFileSync } from "fs";
import { join } from "path";

console.log("🏗️  Building static site for GitHub Pages...");

const srcDir = "src";
const distDir = "dist";

// Create directories
if (!existsSync(srcDir)) mkdirSync(srcDir, { recursive: true });
if (!existsSync(distDir)) mkdirSync(distDir, { recursive: true });

try {
  console.log("✅ Step 1: Copy existing assets from src/");
  
  // Copy pre-built files from src if they exist
  const files = ['index.html', 'styles.css', 'client-main.js'];
  
  for (const file of files) {
    const srcPath = join(srcDir, file);
    const destName = file === 'client-main.js' ? 'main.js' : file;
    const destPath = join(distDir, destName);
    
    if (existsSync(srcPath)) {
      copyFileSync(srcPath, destPath);
      console.log(`✅ Copied ${file} -> ${destName}`);
    } else {
      console.log(`⚠️ Source file not found: ${srcPath}, will generate it`);
      
      // Generate missing files
      if (file === 'styles.css') {
        generateCSS();
      } else if (file === 'client-main.js') {
        generateClientJS();
      } else if (file === 'index.html') {
        generateHTML();
      }
    }
  }

  console.log("✅ Step 2: Fix HTML references");
  
  // Fix the HTML file
  let html = require('fs').readFileSync(join(distDir, 'index.html'), 'utf8');
  
  // Fix script and CSS references
  html = html.replace('./client-main.js', './main.js');
  html = html.replace('src="./main.js"', 'src="./main.js"'); // Ensure it's relative
  html = html.replace('href="./styles.css"', 'href="./styles.css"'); // Ensure it's relative
  
  // Add deployment notice
  const notice = `<!--
  🌐 Static version of Swipe Dataset Studio deployed on GitHub Pages
  🚀 Live Demo: https://tribixbite.github.io/Gesture-Keyboard-Traj-Gen/
  
  This is a client-side only version with simulated trajectory generation.
  For full server functionality, run: bun web-app/server.ts
-->
`;
  
  html = html.replace('<!DOCTYPE html>', notice + '\n<!DOCTYPE html>');
  writeFileSync(join(distDir, 'index.html'), html);
  
  console.log("✅ Fixed HTML references and added deployment notice");
  console.log("🚀 Static build complete! Files ready in dist/ folder:");
  console.log("  📄 index.html");
  console.log("  🎨 styles.css"); 
  console.log("  ⚡ main.js");
  console.log("");
  console.log("🌐 Ready for GitHub Pages deployment!");
  
} catch (error) {
  console.error("❌ Build failed:", error);
  process.exit(1);
}

function generateHTML() {
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Swipe Dataset Studio</title>
  <link rel="stylesheet" href="./styles.css">
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
                <input type="number" id="max-quantity" value="100" min="1" max="10000">
              </div>
            </div>
          </div>
          
          <div class="actions">
            <button id="generate-btn" class="primary">Generate Dataset</button>
            <button id="export-btn" class="secondary" disabled>Export JSON</button>
            <button id="view-generated-btn" class="secondary" disabled>View Results</button>
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
  
  <script src="./main.js"></script>
</body>
</html>`;

  writeFileSync(join(distDir, 'index.html'), html);
  console.log("✅ Generated index.html");
}

function generateCSS() {
  const css = `* {
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
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border-color: #667eea;
}

.tab-content {
  display: none;
}

.tab-content.active {
  display: block;
}

.controls {
  background: #111;
  padding: 25px;
  border-radius: 12px;
  margin-bottom: 20px;
  border: 1px solid #2a2a2a;
}

.control-group {
  margin-bottom: 20px;
}

.control-group:last-child {
  margin-bottom: 0;
}

.control-group > label {
  display: block;
  font-weight: 500;
  margin-bottom: 8px;
  color: #ccc;
}

select, input[type="text"], input[type="number"], input[type="url"] {
  width: 100%;
  padding: 10px 12px;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 6px;
  color: #e0e0e0;
  font-size: 14px;
  transition: all 0.3s;
}

select:focus, input:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.file-input {
  position: relative;
  display: inline-block;
  width: 100%;
  margin-bottom: 10px;
}

.file-input input[type="file"] {
  width: 100%;
  padding: 10px 12px;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 6px;
  color: #e0e0e0;
  cursor: pointer;
}

.file-input span {
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: #888;
  pointer-events: none;
  font-size: 14px;
}

.options {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-top: 10px;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.input-group label {
  font-size: 13px;
  color: #aaa;
}

.checkbox {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  cursor: pointer;
}

.checkbox input[type="checkbox"] {
  width: auto;
  margin: 0;
}

.actions {
  display: flex;
  gap: 10px;
  margin-top: 25px;
}

button {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.primary {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
}

.primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
}

.secondary {
  background: #1a1a1a;
  color: #ccc;
  border: 1px solid #333;
}

.secondary:hover:not(:disabled) {
  background: #222;
  border-color: #667eea;
  color: #fff;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none !important;
}

.status {
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 20px;
  font-size: 14px;
  display: none;
}

.status.show {
  display: block;
}

.status.success {
  background: rgba(34, 197, 94, 0.1);
  border: 1px solid rgba(34, 197, 94, 0.3);
  color: #22c55e;
}

.status.error {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: #ef4444;
}

.status.info {
  background: rgba(59, 130, 246, 0.1);
  border: 1px solid rgba(59, 130, 246, 0.3);
  color: #3b82f6;
}

.preview {
  background: #111;
  padding: 20px;
  border-radius: 8px;
  border: 1px solid #2a2a2a;
  display: none;
}

.preview.show {
  display: block;
}

.preview h3 {
  margin-bottom: 15px;
  color: #667eea;
}

.preview pre {
  background: #0a0a0a;
  padding: 15px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 12px;
  border: 1px solid #333;
}

.info {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
  padding: 20px;
  background: #111;
  border-radius: 8px;
  border: 1px solid #2a2a2a;
}

.info-item {
  text-align: center;
}

.info-item .value {
  font-size: 24px;
  font-weight: 600;
  color: #667eea;
  margin-bottom: 5px;
}

.info-item .label {
  font-size: 12px;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.5px;
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

  writeFileSync(join(distDir, 'styles.css'), css);
  console.log("✅ Generated styles.css");
}

function generateClientJS() {
  // Just copy the existing client-main.js if it exists, otherwise create a minimal version
  if (existsSync(join(srcDir, 'client-main.js'))) {
    copyFileSync(join(srcDir, 'client-main.js'), join(distDir, 'main.js'));
    console.log("✅ Copied client-main.js -> main.js");
  } else {
    console.log("⚠️ client-main.js not found, generating minimal version");
    const js = `console.log('🎯 Swipe Dataset Studio - Static Version');

// Basic app initialization
document.addEventListener('DOMContentLoaded', () => {
  console.log('App loaded - minimal functionality');
  
  // Show notice
  const statusDiv = document.getElementById('generation-status');
  if (statusDiv) {
    statusDiv.innerHTML = '🌐 Static version loaded. Upload a dataset to view traces.';
    statusDiv.className = 'status show info';
  }
});`;

    writeFileSync(join(distDir, 'main.js'), js);
    console.log("✅ Generated minimal main.js");
  }
}