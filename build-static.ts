#!/usr/bin/env bun

import { writeFileSync, mkdirSync, existsSync } from "fs";
import { join } from "path";

console.log("🏗️  Building static site for GitHub Pages...");

const distDir = "dist";

// Create dist directory
if (!existsSync(distDir)) {
  mkdirSync(distDir, { recursive: true });
}

// Extract the functions from our server
// We'll import and call the HTML, CSS, and JS generation functions
const { getIndexHTML, getStyles, getAppJS } = await import("./web-app/server.ts");

try {
  // Generate static files
  const html = getIndexHTML();
  const css = getStyles();  
  const js = getAppJS();

  // Write files to dist
  writeFileSync(join(distDir, "index.html"), html);
  writeFileSync(join(distDir, "styles.css"), css);
  writeFileSync(join(distDir, "app.js"), js);

  console.log("✅ Generated static files:");
  console.log("  📄 index.html");
  console.log("  🎨 styles.css");
  console.log("  ⚡ app.js");
  
  // Create a simple server-less version notice
  const notice = `
<!--
  This is a static version of Swipe Dataset Studio deployed on GitHub Pages.
  Some server-side features are limited, but the core functionality works client-side.
  
  For full functionality, run the Bun server locally:
  bun web-app/server.ts
-->`;

  // Prepend notice to HTML
  const htmlWithNotice = html.replace('<!DOCTYPE html>', notice + '\n<!DOCTYPE html>');
  writeFileSync(join(distDir, "index.html"), htmlWithNotice);

  console.log("🚀 Static build complete! Ready for GitHub Pages deployment.");
  
} catch (error) {
  console.error("❌ Build failed:", error);
  console.error("Make sure the server exports the required functions:");
  console.error("  - getIndexHTML()");
  console.error("  - getStyles()"); 
  console.error("  - getAppJS()");
  process.exit(1);
}