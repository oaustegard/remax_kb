// kb-reader.js — Pure JS reader for remax_kb v2 (.kbi + .kbc/) artifacts.
//
// Vendorable into Cloudflare Workers, Pages Functions, browsers, or Node.
// Zero external dependencies: rolls its own ZIP_STORED reader, NPY parser,
// Hamming popcount, BM25 scoring, and stacked-SimHash query encoder.
//
// Requires the .kbi to ship its rotations — either `binarizer/rotations.f32`
// or `binarizer/rotations.i8` + `binarizer/rotations.scale.f32` when
// `binarizer.rotations_quant == "int8"` (see SPEC_v2 §binarizer/rotations).
// Throws on absence — bit-fidelity with NumPy's QR is impractical without
// the shipped rotations.

// ─────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────

const SPEC_VERSION = "2";
const KIND = "split-index";
const ROW_BYTES_CHUNK_MAP = 24;
const FLAG_TOMBSTONE = 0x01;

// ─────────────────────────────────────────────────────────────────────────
// ZIP_STORED reader (no inflation; central-directory walk)
// ─────────────────────────────────────────────────────────────────────────

const SIG_LFH = 0x04034b50;
const SIG_CFH = 0x02014b50;
const SIG_EOCD = 0x06054b50;

export class ZipStored {
  /** @param {ArrayBuffer | Uint8Array} buffer */
  constructor(buffer) {
    const ab = buffer instanceof Uint8Array ? buffer.buffer : buffer;
    const off = buffer instanceof Uint8Array ? buffer.byteOffset : 0;
    const len = buffer instanceof Uint8Array ? buffer.byteLength : buffer.byteLength;
    this._view = new DataView(ab, off, len);
    this._base = off;
    this._buf = ab;
    this._entries = this._parse();
  }

  _parse() {
    const v = this._view;
    const N = v.byteLength;
    // EOCD is at least 22 bytes; comment can extend it up to 65557
    let eocd = -1;
    for (let i = N - 22; i >= Math.max(0, N - 65557); i--) {
      if (v.getUint32(i, true) === SIG_EOCD) { eocd = i; break; }
    }
    if (eocd < 0) throw new Error("kb-reader: EOCD not found in .kbi");
    const total = v.getUint16(eocd + 10, true);
    const cdOffset = v.getUint32(eocd + 16, true);

    const entries = new Map();
    let p = cdOffset;
    for (let i = 0; i < total; i++) {
      if (v.getUint32(p, true) !== SIG_CFH) {
        throw new Error(`kb-reader: bad central-directory header at ${p}`);
      }
      const method = v.getUint16(p + 10, true);
      if (method !== 0) {
        throw new Error(
          `kb-reader: zip entry uses compression method ${method}; ` +
          `only STORED (0) is supported. SPEC_v2 mandates ZIP_STORED.`
        );
      }
      const size = v.getUint32(p + 24, true);
      const nameLen = v.getUint16(p + 28, true);
      const extraLen = v.getUint16(p + 30, true);
      const commentLen = v.getUint16(p + 32, true);
      const lfh = v.getUint32(p + 42, true);
      const name = new TextDecoder().decode(
        new Uint8Array(this._buf, this._base + p + 46, nameLen)
      );
      // Walk local file header to find data offset
      if (v.getUint32(lfh, true) !== SIG_LFH) {
        throw new Error(`kb-reader: bad local-file header for ${name}`);
      }
      const lfhNameLen = v.getUint16(lfh + 26, true);
      const lfhExtraLen = v.getUint16(lfh + 28, true);
      const dataOffset = lfh + 30 + lfhNameLen + lfhExtraLen;
      entries.set(name, { offset: dataOffset, size });
      p += 46 + nameLen + extraLen + commentLen;
    }
    return entries;
  }

  has(name) { return this._entries.has(name); }
  list() { return [...this._entries.keys()]; }

  /** @returns {Uint8Array} view (no copy) */
  read(name) {
    const e = this._entries.get(name);
    if (!e) throw new Error(`kb-reader: entry not found: ${name}`);
    return new Uint8Array(this._buf, this._base + e.offset, e.size);
  }

  readText(name) {
    return new TextDecoder().decode(this.read(name));
  }
}

// ─────────────────────────────────────────────────────────────────────────
// NPY parser (numpy binary array format)
// ─────────────────────────────────────────────────────────────────────────

const NPY_MAGIC = "\x93NUMPY";

const NPY_DTYPE_CTORS = {
  "<f4": Float32Array, "<f8": Float64Array,
  "<i4": Int32Array,   "<i8": BigInt64Array,
  "<u4": Uint32Array,
  "|i1": Int8Array,    "|u1": Uint8Array,
};

export function parseNpy(bytes) {
  for (let i = 0; i < NPY_MAGIC.length; i++) {
    if (bytes[i] !== NPY_MAGIC.charCodeAt(i)) {
      throw new Error("kb-reader: not an NPY file");
    }
  }
  const verMajor = bytes[6];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let headerLen, headerStart;
  if (verMajor === 1) {
    headerLen = view.getUint16(8, true);
    headerStart = 10;
  } else {
    headerLen = view.getUint32(8, true);
    headerStart = 12;
  }
  const header = new TextDecoder().decode(
    bytes.subarray(headerStart, headerStart + headerLen)
  );
  // Header is a Python dict literal, e.g.:
  //   {'descr': '<f4', 'fortran_order': False, 'shape': (3, 32, 32), }
  const dtype = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1] === "True";
  const shapeStr = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "";
  const shape = shapeStr
    .split(",")
    .map(s => s.trim())
    .filter(s => s.length > 0)
    .map(Number);
  if (fortran) throw new Error("kb-reader: fortran_order NPY not supported");

  const Ctor = NPY_DTYPE_CTORS[dtype];
  if (!Ctor) throw new Error(`kb-reader: unsupported NPY dtype ${dtype}`);

  const dataStart = headerStart + headerLen;
  // Copy to align — typed arrays require alignment matching their elem size
  const slice = bytes.subarray(dataStart).slice();
  const elemCount = slice.byteLength / Ctor.BYTES_PER_ELEMENT;
  const array = new Ctor(slice.buffer, slice.byteOffset, elemCount);
  return { array, shape, dtype };
}

// ─────────────────────────────────────────────────────────────────────────
// Hamming popcount LUT
// ─────────────────────────────────────────────────────────────────────────

const POPCOUNT = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  let v = i, c = 0;
  while (v) { c += v & 1; v >>= 1; }
  POPCOUNT[i] = c;
}

function hammingDistance(a, b, len) {
  // a, b: Uint8Array of length >= len
  let d = 0;
  for (let i = 0; i < len; i++) d += POPCOUNT[a[i] ^ b[i]];
  return d;
}

// ─────────────────────────────────────────────────────────────────────────
// Binarizer (shipped rotations → query code)
// ─────────────────────────────────────────────────────────────────────────

/**
 * Encode a query embedding into a v2 chunk_map code.
 *
 * Matches `remax.StackedSignBitQuantizer.encode()` bit-for-bit when given:
 *   - `qVec`: Float32Array of length `fullDim`
 *   - `mean`: Float32Array of length `fullDim` (binarizer.mean_vector)
 *   - `rotations`: Float32Array of length `k * dim * dim` (binarizer/rotations.f32)
 *
 * Bit-pack convention matches `numpy.packbits(..., bitorder='big')`:
 * within a byte, bit 0 of the projection lands at 0x80, bit 7 at 0x01.
 * Rotation outputs are concatenated in stack-order across the codeword.
 *
 * @returns {Uint8Array} of length `(dim * k) / 8`
 */
export function encodeQueryCode(qVec, mean, rotations, dim, k) {
  const fullDim = mean.length;
  if (qVec.length !== fullDim) {
    throw new Error(`encodeQueryCode: qVec length ${qVec.length} != fullDim ${fullDim}`);
  }
  if (rotations.length !== k * dim * dim) {
    throw new Error(
      `encodeQueryCode: rotations length ${rotations.length} != k*dim*dim ${k * dim * dim}`
    );
  }
  // 1) Center
  const centered = new Float32Array(fullDim);
  for (let i = 0; i < fullDim; i++) centered[i] = qVec[i] - mean[i];
  // 2) Truncate to dim
  const x = centered.subarray(0, dim);

  // 3) For each rotation, compute x @ Q (length dim), then sign-pack
  const rowBytes = (dim * k) / 8;
  const code = new Uint8Array(rowBytes);
  for (let j = 0; j < k; j++) {
    const qOff = j * dim * dim;
    const bitBase = j * dim;
    for (let col = 0; col < dim; col++) {
      let sum = 0;
      const colBase = qOff + col;
      for (let row = 0; row < dim; row++) {
        sum += x[row] * rotations[colBase + row * dim];
      }
      if (sum >= 0) {
        // big-endian bitorder: bit `i` of byte `B` is mask (1 << (7 - i & 7))
        const bitIdx = bitBase + col;
        code[bitIdx >>> 3] |= 1 << (7 - (bitIdx & 7));
      }
    }
  }
  return code;
}

// ─────────────────────────────────────────────────────────────────────────
// Base64
// ─────────────────────────────────────────────────────────────────────────

function base64ToFloat32(b64) {
  // Workers/browsers: atob; Node: Buffer.from
  const binStr = typeof atob === "function"
    ? atob(b64)
    : Buffer.from(b64, "base64").toString("binary");
  const bytes = new Uint8Array(binStr.length);
  for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
  // Re-slice to align to 4-byte boundary
  const aligned = bytes.slice();
  return new Float32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4);
}

// ─────────────────────────────────────────────────────────────────────────
// BM25 scoring from CSC (data, indices, indptr arrays)
// ─────────────────────────────────────────────────────────────────────────

/**
 * Score a tokenized query against a BM25 corpus stored as CSC sparse matrix.
 * Returns Float32Array of scores (length = num_docs = live_count).
 */
function bm25Scores(queryTokens, bm25) {
  const { data, indices, indptr, vocab, numDocs } = bm25;
  const scores = new Float32Array(numDocs);
  for (const tok of queryTokens) {
    const col = vocab[tok];
    if (col == null) continue;
    const start = indptr[col], end = indptr[col + 1];
    for (let p = start; p < end; p++) {
      scores[indices[p]] += data[p];
    }
  }
  return scores;
}

function tokenizeQuery(text) {
  // Matches the writer's bm25s.tokenize default tokenization (lowercase
  // + alphanumeric runs).
  return text.toLowerCase().match(/[a-z0-9]+/g) || [];
}

// ─────────────────────────────────────────────────────────────────────────
// chunk_map.bin row decoder
// ─────────────────────────────────────────────────────────────────────────

function readChunkMapRow(view, rowIdx) {
  const o = rowIdx * ROW_BYTES_CHUNK_MAP;
  return {
    shardId: view.getUint16(o, true),
    flags: view.getUint8(o + 2),
    byteOffset: Number(view.getBigUint64(o + 4, true)),
    byteLength: view.getUint32(o + 12, true),
    chunkIdOffset: Number(view.getBigUint64(o + 16, true)),
  };
}

function readChunkId(chunkIds, offset) {
  let end = offset;
  while (end < chunkIds.length && chunkIds[end] !== 0) end++;
  return new TextDecoder().decode(chunkIds.subarray(offset, end));
}

// ─────────────────────────────────────────────────────────────────────────
// Main reader
// ─────────────────────────────────────────────────────────────────────────

export class KBReader {
  /**
   * Parse a .kbi byte buffer. Throws on validation failure.
   * @param {Uint8Array | ArrayBuffer} kbiBytes
   * @param {string} chunksBaseUri - absolute URL of the .kbc/ directory
   */
  constructor(kbiBytes, chunksBaseUri) {
    const zip = new ZipStored(kbiBytes);

    // Required entries (the rotation sidecar is validated below, per
    // binarizer.rotations_quant — f32 vs int8 ship different files).
    for (const name of ["manifest.json", "vectors.bin", "chunk_map.bin",
                        "chunk_ids.bin"]) {
      if (!zip.has(name)) {
        throw new Error(`kb-reader: missing required entry ${name}`);
      }
    }

    this.manifest = JSON.parse(zip.readText("manifest.json"));
    if (this.manifest.spec_version !== SPEC_VERSION) {
      throw new Error(
        `kb-reader: unsupported spec_version ${this.manifest.spec_version}`
      );
    }
    if (this.manifest.kind !== KIND) {
      throw new Error(`kb-reader: unsupported kind ${this.manifest.kind}`);
    }
    const bin = this.manifest.binarizer;
    this._dim = bin.dim;
    this._k = bin.k;
    this._seed = bin.seed;
    this._fullDim = this.manifest.embedder.full_dim;
    this._rowBytes = (this._dim * this._k) / 8;
    this._totalBits = this._rowBytes * 8;

    this._mean = base64ToFloat32(bin.mean_vector_b64);
    if (this._mean.length !== this._fullDim) {
      throw new Error(
        `kb-reader: mean length ${this._mean.length} != full_dim ${this._fullDim}`
      );
    }

    // Rotations — f32 sidecar, or int8 + per-output-column scale.
    // See SPEC_v2 §binarizer/rotations.i8. The corpus is packed against the
    // dequantized rotations, so we dequantize and use these exactly (never
    // re-derive from seed).
    const nRot = this._k * this._dim * this._dim;
    const rotQuant = bin.rotations_quant || "float32";
    if (rotQuant === "int8") {
      if (!zip.has("binarizer/rotations.i8") ||
          !zip.has("binarizer/rotations.scale.f32")) {
        throw new Error(
          "kb-reader: rotations_quant=int8 but rotations.i8/scale.f32 missing"
        );
      }
      const i8u = zip.read("binarizer/rotations.i8").slice();
      const i8 = new Int8Array(i8u.buffer, i8u.byteOffset, i8u.length);
      const scAligned = zip.read("binarizer/rotations.scale.f32").slice();
      const scale = new Float32Array(
        scAligned.buffer, scAligned.byteOffset, this._k * this._dim
      );
      if (i8.length !== nRot) {
        throw new Error("kb-reader: rotations.i8 size mismatch");
      }
      if (scale.length !== this._k * this._dim) {
        throw new Error("kb-reader: rotations.scale.f32 size mismatch");
      }
      // Dequant: Q[j, row, col] = i8[j, row, col] * scale[j, col]
      const rot = new Float32Array(nRot);
      const d = this._dim;
      for (let j = 0; j < this._k; j++) {
        const base = j * d * d;
        const sBase = j * d;
        for (let row = 0; row < d; row++) {
          const rBase = base + row * d;
          for (let col = 0; col < d; col++) {
            rot[rBase + col] = i8[rBase + col] * scale[sBase + col];
          }
        }
      }
      this._rotations = rot;
    } else {
      if (!zip.has("binarizer/rotations.f32")) {
        throw new Error("kb-reader: missing required entry binarizer/rotations.f32");
      }
      const rotAligned = zip.read("binarizer/rotations.f32").slice();
      this._rotations = new Float32Array(
        rotAligned.buffer, rotAligned.byteOffset, nRot
      );
      if (this._rotations.length !== nRot) {
        throw new Error("kb-reader: rotations size mismatch");
      }
    }

    // Vectors
    const vecBytes = zip.read("vectors.bin");
    const total = this.manifest.chunks.total_rows;
    if (vecBytes.length !== total * this._rowBytes) {
      throw new Error(
        `kb-reader: vectors.bin size ${vecBytes.length} != ${total} * ${this._rowBytes}`
      );
    }
    this._vectors = vecBytes.slice();  // own copy
    this._totalRows = total;

    // chunk_map
    const cmBytes = zip.read("chunk_map.bin");
    if (cmBytes.length !== total * ROW_BYTES_CHUNK_MAP) {
      throw new Error("kb-reader: chunk_map.bin size mismatch");
    }
    const cmAligned = cmBytes.slice();
    this._chunkMap = cmAligned;
    this._chunkMapView = new DataView(
      cmAligned.buffer, cmAligned.byteOffset, cmAligned.byteLength
    );

    // chunk_ids
    this._chunkIds = zip.read("chunk_ids.bin").slice();

    // bm25 (optional)
    if (zip.has("bm25/data.csc.index.npy")) {
      const data = parseNpy(zip.read("bm25/data.csc.index.npy")).array;
      const indices = parseNpy(zip.read("bm25/indices.csc.index.npy")).array;
      const indptr = parseNpy(zip.read("bm25/indptr.csc.index.npy")).array;
      const vocab = JSON.parse(zip.readText("bm25/vocab.index.json"));
      const params = JSON.parse(zip.readText("bm25/params.index.json"));
      this._bm25 = { data, indices, indptr, vocab, numDocs: params.num_docs };
    } else {
      this._bm25 = null;
    }

    // Tombstone mask + live→absolute row mapping
    this._tomb = new Uint8Array(total);
    this._rowOfLive = [];
    for (let i = 0; i < total; i++) {
      const f = this._chunkMapView.getUint8(i * ROW_BYTES_CHUNK_MAP + 2);
      if (f & FLAG_TOMBSTONE) {
        this._tomb[i] = 1;
      } else {
        this._rowOfLive.push(i);
      }
    }
    if (this._bm25 && this._rowOfLive.length !== this._bm25.numDocs) {
      throw new Error(
        `kb-reader: bm25 num_docs ${this._bm25.numDocs} != live rows ${this._rowOfLive.length}`
      );
    }

    this._chunksUri = chunksBaseUri.endsWith("/") ? chunksBaseUri : chunksBaseUri + "/";
  }

  get liveCount() { return this._rowOfLive.length; }

  // ───── Query path ─────

  /**
   * Run hybrid search.
   * @param {string} query
   * @param {Float32Array} queryEmbedding - already-embedded query vector
   * @param {number} k - top-K to return
   * @param {number|null} alpha - null → RRF; number → weighted
   * @returns array of hits, NOT yet enriched with text/meta.
   */
  search(query, queryEmbedding, { k = 5, alpha = null, overFetch = null } = {}) {
    const dense = this._denseSearch(queryEmbedding);
    const lex = this._bm25 ? this._bm25Search(query) : null;
    const N = overFetch ?? Math.max(k * 4, 20);

    if (!lex) return dense.slice(0, k).map(h => this._withChunkId(h));

    const fused = fuseRanks(dense, lex, N, alpha);
    return fused.slice(0, k).map(h => this._withChunkId(h));
  }

  _denseSearch(queryEmbedding) {
    const qCode = encodeQueryCode(
      queryEmbedding, this._mean, this._rotations, this._dim, this._k
    );
    const hits = [];
    for (let i = 0; i < this._totalRows; i++) {
      if (this._tomb[i]) continue;
      const d = hammingDistance(
        this._vectors.subarray(i * this._rowBytes, (i + 1) * this._rowBytes),
        qCode, this._rowBytes
      );
      hits.push({
        row: i,
        dense_dist: d,
        dense_sim: 1 - d / this._totalBits,
      });
    }
    hits.sort((a, b) => a.dense_dist - b.dense_dist);
    return hits;
  }

  _bm25Search(query) {
    const toks = tokenizeQuery(query);
    if (!toks.length) return [];
    const scores = bm25Scores(toks, this._bm25);
    const hits = [];
    for (let liveIdx = 0; liveIdx < scores.length; liveIdx++) {
      if (scores[liveIdx] <= 0) continue;
      hits.push({
        row: this._rowOfLive[liveIdx],
        bm25_score: scores[liveIdx],
      });
    }
    hits.sort((a, b) => b.bm25_score - a.bm25_score);
    return hits;
  }

  _withChunkId(hit) {
    const row = readChunkMapRow(this._chunkMapView, hit.row);
    return { ...hit, chunk_id: readChunkId(this._chunkIds, row.chunkIdOffset) };
  }

  // ───── Chunk fetch via HTTP Range ─────

  async fetchChunks(hits, fetchImpl = globalThis.fetch) {
    return await Promise.all(hits.map(async (hit) => {
      const row = readChunkMapRow(this._chunkMapView, hit.row);
      const url = this._chunksUri + `shard-${String(row.shardId).padStart(4, "0")}.bin`;
      const end = row.byteOffset + row.byteLength - 1;
      const resp = await fetchImpl(url, {
        headers: { range: `bytes=${row.byteOffset}-${end}` },
        cf: { cacheTtl: 3600, cacheEverything: true },
      });
      if (!resp.ok && resp.status !== 206) {
        throw new Error(`kb-reader: chunk fetch failed: ${resp.status}`);
      }
      const data = new Uint8Array(await resp.arrayBuffer());
      const nl = data.indexOf(0x0a);
      const headerJson = new TextDecoder().decode(data.subarray(0, nl));
      const text = new TextDecoder().decode(data.subarray(nl + 1));
      const header = JSON.parse(headerJson);
      const verified = (await sha256Hex(text)) === header.sha256;
      return {
        ...hit,
        text,
        meta: header.meta || {},
        sha256: header.sha256,
        verified,
      };
    }));
  }

  async searchAndFetch(query, queryEmbedding, opts = {}, fetchImpl) {
    const hits = this.search(query, queryEmbedding, opts);
    return await this.fetchChunks(hits, fetchImpl);
  }
}

async function sha256Hex(s) {
  const buf = new TextEncoder().encode(s);
  const hash = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2, "0")).join("");
}

// ─────────────────────────────────────────────────────────────────────────
// Fusion
// ─────────────────────────────────────────────────────────────────────────

export function fuseRanks(dense, lex, overFetch, alpha) {
  const denseN = dense.slice(0, overFetch);
  const lexN = lex.slice(0, overFetch);

  if (alpha == null) {
    // RRF
    const C = 60;
    const merged = new Map();
    denseN.forEach((h, idx) => {
      merged.set(h.row, { ...h, fused: 1 / (C + idx + 1) });
    });
    lexN.forEach((h, idx) => {
      const prev = merged.get(h.row);
      const add = 1 / (C + idx + 1);
      if (prev) {
        prev.bm25_score = h.bm25_score;
        prev.fused += add;
      } else {
        merged.set(h.row, { ...h, fused: add });
      }
    });
    return [...merged.values()].sort((a, b) => b.fused - a.fused);
  }

  // Weighted with min-max norm
  const dDists = denseN.map(h => h.dense_dist);
  const lScores = lexN.map(h => h.bm25_score);
  const dMin = Math.min(...dDists), dMax = Math.max(...dDists);
  const lMin = Math.min(...lScores), lMax = Math.max(...lScores);
  const nd = d => dMax === dMin ? 1 : (dMax - d) / (dMax - dMin);
  const nl = s => lMax === lMin ? 1 : (s - lMin) / (lMax - lMin);

  const merged = new Map();
  denseN.forEach(h => {
    merged.set(h.row, { ...h, fused: alpha * nd(h.dense_dist) });
  });
  lexN.forEach(h => {
    const add = (1 - alpha) * nl(h.bm25_score);
    const prev = merged.get(h.row);
    if (prev) {
      prev.bm25_score = h.bm25_score;
      prev.fused += add;
    } else {
      merged.set(h.row, { ...h, fused: add });
    }
  });
  return [...merged.values()].sort((a, b) => b.fused - a.fused);
}
