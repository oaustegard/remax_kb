// Emit js/kb-reader.js's tokenizeQuery() output for a list of strings, so
// tests/gates/gate_tokenizer_parity.py can compare it against bm25s itself.
//
//   echo '["response_model", "café"]' | node js_tokenize.mjs [--legacy]
//
// --legacy runs the pre-fix `[a-z0-9]+` regex instead — the gate's known-bad.
// It is defined here and nowhere else; the shipped reader has one pattern.

import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const { tokenizeQuery } = await import(
  pathToFileURL(resolve(HERE, "../../js/kb-reader.js")).href
);

const legacy = process.argv.includes("--legacy");
const legacyTokenize = (text) => text.toLowerCase().match(/[a-z0-9]+/g) || [];
const fn = legacy ? legacyTokenize : tokenizeQuery;

let input = "";
for await (const chunk of process.stdin) input += chunk;
process.stdout.write(JSON.stringify(JSON.parse(input).map(fn)));
