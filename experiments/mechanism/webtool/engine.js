"use strict";
const DIG = "0123456789ABCDEF", D = 64, NH = 4, DH = 16, NM = 256;
const BASE_NAME = { 2: "binary", 8: "octal", 10: "decimal", 16: "hexadecimal" };
const THRESH = { 2: 0.5, 8: 4.5, 10: 5.5, 16: 8.5 };
const WINDOW = { 2: [170, 190], 8: [70, 80], 10: [60, 110], 16: [60, 110] };
const TIERS = [["0, 1", "01"], ["2 to 7", "234567"], ["8, 9", "89"], ["A to F", "ABCDEF"]];
const TIER_BASE = ["binary", "octal", "decimal", "hexadecimal"];

// ---------- weights ----------
const T = {};
for (const [k, v] of Object.entries(MODEL_WEIGHTS.tensors)) {
  const bin = atob(v.b64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  T[k] = new Float32Array(u.buffer);
}
const VOC = MODEL_WEIGHTS.vocab, ITOS = {};
for (const [k, v] of Object.entries(VOC)) ITOS[v] = k;
const NV = Object.keys(VOC).length, EOS = VOC["<eos>"], NL = VOC["\n"];
const WE = T["wte.weight"], PE = T["wpe.weight"];

function ln(x, w, b) {
  let m = 0; for (let i = 0; i < D; i++) m += x[i]; m /= D;
  let v = 0; for (let i = 0; i < D; i++) { const d = x[i] - m; v += d * d; } v /= D;
  const s = Math.sqrt(v + 1e-5), o = new Float64Array(D);
  for (let i = 0; i < D; i++) o[i] = (x[i] - m) / s * w[i] + b[i];
  return o;
}
function mv(x, W, b, nin, nout) {
  const o = new Float64Array(nout);
  if (b) for (let j = 0; j < nout; j++) o[j] = b[j];
  for (let i = 0; i < nin; i++) { const xi = x[i], off = i * nout; for (let j = 0; j < nout; j++) o[j] += xi * W[off + j]; }
  return o;
}
const gelu = x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));

// Full forward pass; returns logits at the last position and the values the trace needs there.
function forward(ids) {
  const n = ids.length, X = [], cap = {};
  for (let i = 0; i < n; i++) { const x = new Float64Array(D); for (let d = 0; d < D; d++) x[d] = WE[ids[i] * D + d] + PE[i * D + d]; X.push(x); }
  for (let L = 0; L < 2; L++) {
    const p = `h.${L}.`;
    if (L === 1) cap.resid_pre1 = X[n - 1].slice();
    const QKV = X.map(x => mv(ln(x, T[p + "ln_1.weight"], T[p + "ln_1.bias"]), T[p + "attn.c_attn.weight"], T[p + "attn.c_attn.bias"], D, 3 * D));
    const pats = [];
    const Z = [];
    for (let i = 0; i < n; i++) {
      const z = new Float64Array(D);
      for (let h = 0; h < NH; h++) {
        const sc = new Float64Array(i + 1); let mx = -Infinity;
        for (let j = 0; j <= i; j++) { let s = 0; for (let d = 0; d < DH; d++) s += QKV[i][h * DH + d] * QKV[j][D + h * DH + d]; s /= 4; sc[j] = s; if (s > mx) mx = s; }
        let tot = 0; for (let j = 0; j <= i; j++) { sc[j] = Math.exp(sc[j] - mx); tot += sc[j]; }
        for (let j = 0; j <= i; j++) { sc[j] /= tot; const pj = sc[j]; for (let d = 0; d < DH; d++) z[h * DH + d] += pj * QKV[j][2 * D + h * DH + d]; }
        if (i === n - 1) pats.push(sc);
      }
      Z.push(z);
    }
    cap["pattern" + L] = pats;
    if (L === 1) cap.z1 = Z[n - 1];
    for (let i = 0; i < n; i++) {
      const a = mv(Z[i], T[p + "attn.c_proj.weight"], T[p + "attn.c_proj.bias"], D, D);
      for (let d = 0; d < D; d++) X[i][d] += a[d];
      if (L === 1 && i < n - 1) continue; // only the last position matters after layer 1 attention
      const pre = mv(ln(X[i], T[p + "ln_2.weight"], T[p + "ln_2.bias"]), T[p + "mlp.c_fc.weight"], T[p + "mlp.c_fc.bias"], D, NM);
      const post = pre.map(gelu);
      const mo = mv(post, T[p + "mlp.c_proj.weight"], T[p + "mlp.c_proj.bias"], NM, D);
      for (let d = 0; d < D; d++) X[i][d] += mo[d];
      if (L === 1) { cap.pre1 = pre; cap.post1 = post; }
    }
  }
  cap.resid_post1 = X[n - 1];
  const f = ln(X[n - 1], T["ln_f.weight"], T["ln_f.bias"]), logits = new Float64Array(NV);
  for (let t = 0; t < NV; t++) { let s = 0; for (let d = 0; d < D; d++) s += f[d] * WE[t * D + d]; logits[t] = s; }
  return { logits, cap };
}

// Direct logit attribution helpers: P[n][t] = centred output weights of neuron n . (embedding of t * ln_f gain)
const Wo1 = T["h.1.mlp.c_proj.weight"], bo1 = T["h.1.mlp.c_proj.bias"], Wp1 = T["h.1.attn.c_proj.weight"], gF = T["ln_f.weight"];
function centDotU(vec, t) { let m = 0; for (let d = 0; d < D; d++) m += vec[d]; m /= D; let s = 0; for (let d = 0; d < D; d++) s += (vec[d] - m) * WE[t * D + d] * gF[d]; return s; }
const P = []; for (let n = 0; n < NM; n++) { const row = Wo1.subarray(n * D, n * D + D); P.push(Array.from({ length: NV }, (_, t) => centDotU(row, t))); }
const PB = Array.from({ length: NV }, (_, t) => centDotU(bo1, t));

// ---------- prompts ----------
function toBase(n, b) { if (n === 0) return "0"; let s = ""; while (n) { s = DIG[n % b] + s; n = Math.floor(n / b); } return s; }
const rev = s => s.split("").reverse().join("");
function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function shots(seed, a, b) {
  const r = rng(seed), seen = new Set([a + "," + b]), out = [];
  while (out.length < 5) { const x = Math.floor(r() * 1000), y = Math.floor(r() * 1000); if (!seen.has(x + "," + y)) { seen.add(x + "," + y); out.push([x, y]); } }
  return out;
}

// ---------- trace ----------
function trace(a, b, base, ctxBase, seed) {
  const sh = shots(seed, a, b);
  const lines = sh.map(([x, y]) => `${toBase(x, ctxBase)}+${toBase(y, ctxBase)}=${rev(toBase(x + y, ctxBase))}`);
  const sa = toBase(a, base), sb = toBase(b, base);
  const prompt = lines.join("\n") + "\n" + `${sa}+${sb}=`;
  const ids = [...prompt].map(c => VOC[c]);
  const start = prompt.lastIndexOf("\n") + 1, plus = start + sa.length, eq = plus + 1 + sb.length;
  const digOf = (s, k) => (k < s.length ? DIG.indexOf(s[s.length - 1 - k]) : null);
  const truth = rev(toBase(a + b, base));
  const steps = [];
  let first = null;
  for (let k = 0; k < 16 && ids.length < 256; k++) {
    const { logits, cap } = forward(ids);
    if (!first) first = cap;
    let best = 0; for (let t = 1; t < NV; t++) if (logits[t] > logits[best]) best = t;
    const R = cap.resid_post1; let m = 0; for (const v of R) m += v; m /= D; let vv = 0; for (const v of R) vv += (v - m) ** 2; const scale = Math.sqrt(vv / D + 1e-5);
    steps.push({ k, q: ids.length - 1, token: best, logits, cap, scale });
    ids.push(best);
    if (best === EOS || best === NL) break;
  }
  return { a, b, base, ctxBase, sa, sb, prompt, ids, start, plus, eq, steps, truth, digOf, first, lines };
}

