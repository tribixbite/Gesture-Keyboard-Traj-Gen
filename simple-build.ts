#!/usr/bin/env bun

import { writeFileSync, mkdirSync, existsSync, copyFileSync } from "fs";
import { join } from "path";

console.log("🏗️  Building static site for GitHub Pages...");

const distDir = "dist";

// Create dist directory
if (!existsSync(distDir)) {
  mkdirSync(distDir, { recursive: true });
}

try {
  // Copy pre-built files from src
  const files = ['index.html', 'styles.css', 'client-main.js'];
  
  for (const file of files) {
    const srcPath = join('src', file);
    const destPath = join(distDir, file === 'client-main.js' ? 'main.js' : file);
    
    if (existsSync(srcPath)) {
      copyFileSync(srcPath, destPath);
      console.log(`✅ Copied ${file} -> ${file === 'client-main.js' ? 'main.js' : file}`);
    } else {
      console.log(`⚠️ Source file not found: ${srcPath}`);
    }
  }

  // Fix the script reference in HTML
  let html = require('fs').readFileSync(join(distDir, 'index.html'), 'utf8');
  html = html.replace('./client-main.js', './main.js');
  html = html.replace('/Gesture-Keyboard-Traj-Gen/', './'); // Fix base path for GitHub Pages
  
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
  console.log("🚀 Static build complete! Ready for GitHub Pages deployment.");
  
} catch (error) {
  console.error("❌ Build failed:", error);
  process.exit(1);
}