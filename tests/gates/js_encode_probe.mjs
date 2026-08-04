// Drive js/kb-reader.js's encodeQueryCode() over caller-supplied planes and
// vectors, with no .kbi involved.
//
// gate_cross_reader.py's main arm can only compare the two readers on the
// projections a committed fixture happens to produce, and those all sit far
// from zero. This harness exists so the gate can hand BOTH readers a
// deliberately-constructed exact-zero projection — the case where the sign
// convention (`> 0` vs `>= 0`) decides the bit on its own, with no float
// rounding involved.
//
//   node js_encode_probe.mjs --spec <probe.json> [--reader <path/kb-reader.js>]
//
// probe.json: {"dim": int, "k": int, "mean": [f...], "rotations": [f...],
//              "vectors": [[f...], ...]}
// stdout:     {"codes": ["<hex>", ...]}
//
// --reader defaults to the shipped js/kb-reader.js; the gate points it at a
// mutated copy to prove the exact-zero check can go red.

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
const { encodeQueryCode } = await import(pathToFileURL(readerPath).href);

const spec = JSON.parse(await readFile(resolve(arg("spec")), "utf8"));
const mean = Float32Array.from(spec.mean);
const rotations = Float32Array.from(spec.rotations);

const hex = (u8) => [...u8].map((b) => b.toString(16).padStart(2, "0")).join("");

const codes = spec.vectors.map((v) =>
  hex(encodeQueryCode(Float32Array.from(v), mean, rotations, spec.dim, spec.k))
);

process.stdout.write(JSON.stringify({ codes }));
