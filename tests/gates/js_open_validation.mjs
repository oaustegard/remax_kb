// Open a .kbi with the REAL js/kb-reader.js and report whether the constructor
// accepted it. Used by tests/gates/gate_open_validation.py to check the two
// readers agree about which artifacts are refusable — js/kb-reader.js already
// throws on the bm25 row-count mismatch (SPEC_v2 validation step 7) that the
// Python reader used to accept, so it is a genuine second implementation of
// this claim, not a transcription of the Python one.
//
//   node js_open_validation.mjs --kbi <path.kbi> [--reader <kb-reader.js>]
//
// stdout: {"opened": bool, "error": string|null, "live_count": number|null}
//
// --reader defaults to the shipped js/kb-reader.js. The gate points it at a
// mutated copy to prove the readers-agree assertion can actually go red.

import { readFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));

function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0) {
    if (dflt === undefined) throw new Error(`missing --${name}`);
    return dflt;
  }
  return process.argv[i + 1];
}

const readerPath = resolve(arg("reader", resolve(HERE, "../../js/kb-reader.js")));
const { KBReader } = await import(pathToFileURL(readerPath).href);

const kbiPath = resolve(arg("kbi"));
const out = { opened: false, error: null, live_count: null };

try {
  const kbi = new Uint8Array(await readFile(kbiPath));
  const reader = new KBReader(kbi, "file:///nonexistent/");
  out.opened = true;
  out.live_count = reader.liveCount;
} catch (e) {
  out.error = String((e && e.message) || e);
}

process.stdout.write(JSON.stringify(out));
