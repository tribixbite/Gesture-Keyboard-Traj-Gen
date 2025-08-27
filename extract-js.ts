#!/usr/bin/env bun

import { writeFileSync } from "fs";
import { getAppJS } from "./web-app/server.ts";

const js = getAppJS();
writeFileSync("src/main.js", js);
console.log("✅ Extracted JS to src/main.js");