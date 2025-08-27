#!/usr/bin/env bun

import { writeFileSync } from "fs";
import { getStyles } from "./web-app/server.ts";

const css = getStyles();
writeFileSync("src/styles.css", css);
console.log("✅ Extracted CSS to src/styles.css");