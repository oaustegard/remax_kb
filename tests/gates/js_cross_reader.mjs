// Drive the REAL js/kb-reader.js over a committed fixture and emit its
// results as JSON on stdout, for tests/gates/gate_cross_reader.py to compare
// against the Python reader.
//
// This file exists because every "JS compat" assertion in the Python test
// suite was a Python re-implementation of the JS source, which cannot catch a
// divergence between the two languages — the one thing the format promises.
//
//   node js_cross_reader.mjs --kbi <path.kbi> --kbc <dir> --queries <q.json> \
//        [--k 5] [--over-fetch N] [--rrf-c N] [--min-sim auto|off|<float>] \
//        [--reader <kb-reader.js>]
//
// --over-fetch / --rrf-c / --min-sim are OMITTED by default, so the reader's
// own defaults apply and the gate can compare them against Python's instead of
// pinning both sides to a value neither reader would have chosen.
//
// Nothing here re-implements reader logic: it imports KBReader, encodeQueryCode,
// tokenizeQuery, defaultOverFetch and RRF_C_DEFAULT from js/kb-reader.js and
// only marshals I/O. In particular the reported knob values come from the
// reader's own exports, not from a copy of its expressions.

import { readFile, open } from "node:fs/promises";
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
const { KBReader, encodeQueryCode, tokenizeQuery, defaultOverFetch,
        RRF_C_DEFAULT } = await import(pathToFileURL(readerPath).href);

const kbiPath = resolve(arg("kbi"));
const kbcDir = resolve(arg("kbc"));
const queriesPath = resolve(arg("queries"));
const topK = Number(arg("k", "5"));

// null everywhere means "not supplied", i.e. let the reader default apply.
const rawOverFetch = arg("over-fetch", null);
const overFetch = rawOverFetch === null ? null : Number(rawOverFetch);
const rawRrfC = arg("rrf-c", null);
const rrfC = rawRrfC === null ? RRF_C_DEFAULT : Number(rawRrfC);
const rawMinSim = arg("min-sim", null);
const minSim = rawMinSim === null ? null
  : (Number.isNaN(Number(rawMinSim)) ? rawMinSim : Number(rawMinSim));

// A file:// fetch with Range support, so fetchChunks() exercises the real
// chunk_map offsets instead of being stubbed out.
async function fileFetch(url, opts) {
  const m = /bytes=(\d+)-(\d+)/.exec(opts.headers.range);
  const start = Number(m[1]), end = Number(m[2]);
  const path = url.startsWith("file://") ? fileURLToPath(url) : url;
  const fh = await open(path, "r");
  try {
    const buf = Buffer.alloc(end - start + 1);
    await fh.read(buf, 0, buf.length, start);
    return {
      ok: true,
      status: 206,
      arrayBuffer: async () =>
        buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
    };
  } finally {
    await fh.close();
  }
}

const kbi = new Uint8Array(await readFile(kbiPath));
const spec = JSON.parse(await readFile(queriesPath, "utf8"));
const reader = new KBReader(kbi, pathToFileURL(kbcDir + "/").href);

const hex = (u8) => [...u8].map((b) => b.toString(16).padStart(2, "0")).join("");

const out = {
  live_count: reader.liveCount,
  total_rows: reader._totalRows,
  row_bytes: reader._rowBytes,
  // Knob values as the READER resolves them, for default-parity checks.
  default_over_fetch: defaultOverFetch(topK),
  rrf_c_default: RRF_C_DEFAULT,
  resolved_min_sim: reader.resolveMinSim(minSim),
  has_rotation_entry: reader._zipNames
    ? reader._zipNames.some((n) => n.startsWith("binarizer/rotations"))
    : null,
  queries: [],
};

for (const q of spec.queries) {
  const qvec = Float32Array.from(q.embedding);
  const code = encodeQueryCode(
    qvec, reader._mean, reader._rotations, reader._dim, reader._k
  );
  const hits = await reader.searchAndFetch(
    q.text, qvec, { k: topK, alpha: null, overFetch, rrfC, minSim }, fileFetch
  );
  out.queries.push({
    text: q.text,
    tokens: tokenizeQuery(q.text),
    query_code_hex: hex(code),
    hits: hits.map((h) => ({
      chunk_id: h.chunk_id,
      row: h.row,
      dense_dist: h.dense_dist ?? null,
      bm25_score: h.bm25_score ?? null,
      fused: h.fused ?? null,
      text: h.text,
      sha256: h.sha256,
      verified: h.verified,
    })),
  });
}

process.stdout.write(JSON.stringify(out));
