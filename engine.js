/* ============================================================
   DL-Kern: kleine Autograd-Engine (Reverse Mode) für den Browser.
   Jede Operation speichert ihre Eltern und eine back()-Funktion,
   die den Gradienten nach hinten weitergibt (Kettenregel).
   ============================================================ */
const DL = (() => {
  // ---------- Zufall (mit Startwert, damit alles reproduzierbar ist)
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  let _rand = mulberry32(7);
  const seed = (s) => { _rand = mulberry32(s); };
  const rnd = () => _rand();
  function randn() {
    let u = 0;
    while (u === 0) u = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
  }
  function shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }

  // ---------- Graph aufzeichnen oder nicht
  let recording = true;
  function noGrad(fn) { const p = recording; recording = false; try { return fn(); } finally { recording = p; } }

  class V {
    constructor(d, shape, parents, op) {
      this.d = d; this.shape = shape; this.n = d.length; this.g = null;
      this.p = recording ? (parents || []) : [];
      this.back = null; this.op = op || '';
    }
    grad() { if (!this.g) this.g = new Float64Array(this.n); return this.g; }
  }
  const tensor = (arr, shape) => new V(arr instanceof Float64Array ? arr : Float64Array.from(arr), shape || [arr.length], [], 'input');

  // Parameter = Tensor, der beim Training verändert wird
  function param(shape, init, opts = {}) {
    const n = shape.reduce((a, b) => a * b, 1);
    const d = new Float64Array(n);
    for (let k = 0; k < n; k++) d[k] = init(k);
    const v = new V(d, shape, [], 'param');
    v.isParam = true; v.decay = opts.decay !== false; v.name = opts.name || '';
    v.g = new Float64Array(n); v.delta = new Float64Array(n);
    return v;
  }

  function backward(root, seedGrad = 1) {
    const topo = [], seen = new Set(), stack = [[root, 0]];
    while (stack.length) {
      const top = stack[stack.length - 1];
      const v = top[0];
      if (top[1] === 0) {
        if (seen.has(v)) { stack.pop(); continue; }
        seen.add(v);
      }
      if (top[1] < v.p.length) {
        const c = v.p[top[1]++];
        if (!seen.has(c)) stack.push([c, 0]);
      } else { stack.pop(); topo.push(v); }
    }
    root.grad()[0] += seedGrad;
    for (let k = topo.length - 1; k >= 0; k--) { const v = topo[k]; if (v.back && v.g) v.back(); }
  }

  // ---------- Aktivierungsfunktionen (wie in L2)
  function erf(x) { // Abramowitz-Stegun 7.1.26
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const PHI = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  const phi = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  const ACT = {
    heaviside: { f: (z) => (z >= 0 ? 1 : 0), d: () => 0 },
    sign:      { f: (z) => (z > 0 ? 1 : z < 0 ? -1 : 0), d: () => 0 },
    linear:    { f: (z) => z, d: () => 1 },
    sigmoid:   { f: sig, d: (z) => { const s = sig(z); return s * (1 - s); } },
    tanh:      { f: Math.tanh, d: (z) => { const t = Math.tanh(z); return 1 - t * t; } },
    relu:      { f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0) },
    leaky:     { f: (z) => (z > 0 ? z : 0.01 * z), d: (z) => (z > 0 ? 1 : 0.01) },
    swish:     { f: (z) => z * sig(z), d: (z) => { const s = sig(z); return s + z * s * (1 - s); } },
    gelu:      { f: (z) => z * PHI(z), d: (z) => PHI(z) + z * phi(z) },
  };

  // ---------- Operationen
  // Fully Connected: z_j = b_j + sum_i W[j,i] x_i   (Zeile j von W = alle Gewichte in Neuron j)
  function linear(W, b, x) {
    const [o, i] = W.shape, Wd = W.d, xd = x.d, out = new Float64Array(o);
    for (let j = 0; j < o; j++) {
      let s = b ? b.d[j] : 0; const r = j * i;
      for (let k = 0; k < i; k++) s += Wd[r + k] * xd[k];
      out[j] = s;
    }
    const y = new V(out, [o], b ? [W, b, x] : [W, x], 'linear');
    if (recording) y.back = () => {
      const g = y.g, gW = W.grad(), gx = x.grad(), gb = b ? b.grad() : null;
      for (let j = 0; j < o; j++) {
        const gj = g[j]; if (gj === 0) continue;
        if (gb) gb[j] += gj;
        const r = j * i;
        for (let k = 0; k < i; k++) { gW[r + k] += gj * xd[k]; gx[k] += gj * Wd[r + k]; }
      }
    };
    return y;
  }

  function act(x, key) {
    const A = ACT[key], n = x.n, out = new Float64Array(n);
    for (let k = 0; k < n; k++) out[k] = A.f(x.d[k]);
    const y = new V(out, x.shape, [x], 'act');
    y.key = key;
    if (recording) y.back = () => { const gx = x.grad(), g = y.g; for (let k = 0; k < n; k++) gx[k] += g[k] * A.d(x.d[k]); };
    return y;
  }

  // Dropout wie in L3: x'_i = 0 mit Wahrscheinlichkeit p, sonst x_i / (1 - p)
  function dropout(x, p, train) {
    if (!train || !(p > 0)) return x;
    const n = x.n, mask = new Float64Array(n), out = new Float64Array(n), keep = 1 - p;
    for (let k = 0; k < n; k++) { mask[k] = rnd() < p ? 0 : 1 / keep; out[k] = x.d[k] * mask[k]; }
    const y = new V(out, x.shape, [x], 'dropout');
    y.mask = mask;
    if (recording) y.back = () => { const gx = x.grad(), g = y.g; for (let k = 0; k < n; k++) gx[k] += g[k] * mask[k]; };
    return y;
  }

  function softmax(arr) {
    let m = -Infinity; for (const v of arr) if (v > m) m = v;
    const p = new Float64Array(arr.length); let s = 0;
    for (let k = 0; k < arr.length; k++) { p[k] = Math.exp(arr[k] - m); s += p[k]; }
    for (let k = 0; k < arr.length; k++) p[k] /= s;
    return p;
  }

  // Softmax + Cross-Entropy: L = -log(ŷ_k) für die richtige Klasse k; dL/dz = ŷ - y
  function softmaxCE(z, label) {
    const p = softmax(z.d);
    const L = -Math.log(Math.max(p[label], 1e-12));
    const y = new V(new Float64Array([L]), [1], [z], 'ce');
    y.probs = p; y.label = label;
    if (recording) y.back = () => { const g = y.g[0], gz = z.grad(); for (let k = 0; k < z.n; k++) gz[k] += g * (p[k] - (k === label ? 1 : 0)); };
    return y;
  }

  // Summe skalarer Werte mal c (für Mittelwerte über Zeitschritte)
  function sumScalars(list, c = 1) {
    let s = 0; for (const v of list) s += v.d[0];
    const y = new V(new Float64Array([s * c]), [1], list, 'sum');
    if (recording) y.back = () => { const g = y.g[0] * c; for (const v of list) v.grad()[0] += g; };
    return y;
  }

  // Faltung (wie in PyTorch eine Kreuzkorrelation), x: [S, X, Y], K: [H, S, M, N]
  function conv2d(x, K, b, pad = 1, stride = 1) {
    const [S, X, Y] = x.shape, [H, S2, M, N] = K.shape;
    if (S !== S2) throw new Error('conv2d: Kanäle passen nicht');
    const Xo = Math.floor((X + 2 * pad - M) / stride) + 1, Yo = Math.floor((Y + 2 * pad - N) / stride) + 1;
    const xd = x.d, Kd = K.d, out = new Float64Array(H * Xo * Yo);
    for (let h = 0; h < H; h++) for (let u = 0; u < Xo; u++) for (let v = 0; v < Yo; v++) {
      let s = b.d[h];
      for (let c = 0; c < S; c++) for (let m = 0; m < M; m++) {
        const iu = u * stride + m - pad; if (iu < 0 || iu >= X) continue;
        for (let n = 0; n < N; n++) {
          const iv = v * stride + n - pad; if (iv < 0 || iv >= Y) continue;
          s += xd[(c * X + iu) * Y + iv] * Kd[((h * S + c) * M + m) * N + n];
        }
      }
      out[(h * Xo + u) * Yo + v] = s;
    }
    const y = new V(out, [H, Xo, Yo], [x, K, b], 'conv');
    y.pad = pad; y.stride = stride;
    if (recording) y.back = () => {
      const g = y.g, gx = x.grad(), gK = K.grad(), gb = b.grad();
      for (let h = 0; h < H; h++) for (let u = 0; u < Xo; u++) for (let v = 0; v < Yo; v++) {
        const go = g[(h * Xo + u) * Yo + v]; if (go === 0) continue;
        gb[h] += go;
        for (let c = 0; c < S; c++) for (let m = 0; m < M; m++) {
          const iu = u * stride + m - pad; if (iu < 0 || iu >= X) continue;
          for (let n = 0; n < N; n++) {
            const iv = v * stride + n - pad; if (iv < 0 || iv >= Y) continue;
            const xi = (c * X + iu) * Y + iv, ki = ((h * S + c) * M + m) * N + n;
            gK[ki] += go * xd[xi]; gx[xi] += go * Kd[ki];
          }
        }
      }
    };
    return y;
  }

  // Max Pooling: nur der größte Wert im Fenster wird weitergegeben (und bekommt den Gradienten)
  function maxpool(x, size = 2) {
    const [C, X, Y] = x.shape, Xo = Math.floor(X / size), Yo = Math.floor(Y / size);
    const out = new Float64Array(C * Xo * Yo), arg = new Int32Array(C * Xo * Yo), xd = x.d;
    for (let c = 0; c < C; c++) for (let u = 0; u < Xo; u++) for (let v = 0; v < Yo; v++) {
      let best = -Infinity, bi = 0;
      for (let a = 0; a < size; a++) for (let bb = 0; bb < size; bb++) {
        const i = (c * X + u * size + a) * Y + v * size + bb;
        if (xd[i] > best) { best = xd[i]; bi = i; }
      }
      const o = (c * Xo + u) * Yo + v; out[o] = best; arg[o] = bi;
    }
    const y = new V(out, [C, Xo, Yo], [x], 'pool');
    y.arg = arg; y.size = size;
    if (recording) y.back = () => { const gx = x.grad(), g = y.g; for (let o = 0; o < g.length; o++) gx[arg[o]] += g[o]; };
    return y;
  }

  function flatten(x) {
    const y = new V(x.d, [x.n], [x], 'flatten');
    if (recording) y.back = () => { const gx = x.grad(), g = y.g; for (let k = 0; k < x.n; k++) gx[k] += g[k]; };
    return y;
  }

  function concat(a, b) {
    const out = new Float64Array(a.n + b.n); out.set(a.d, 0); out.set(b.d, a.n);
    const y = new V(out, [a.n + b.n], [a, b], 'concat');
    if (recording) y.back = () => {
      const g = y.g, ga = a.grad(), gb = b.grad();
      for (let k = 0; k < a.n; k++) ga[k] += g[k];
      for (let k = 0; k < b.n; k++) gb[k] += g[a.n + k];
    };
    return y;
  }
  function add(a, b) {
    const out = new Float64Array(a.n); for (let k = 0; k < a.n; k++) out[k] = a.d[k] + b.d[k];
    const y = new V(out, a.shape, [a, b], 'add');
    if (recording) y.back = () => { const g = y.g, ga = a.grad(), gb = b.grad(); for (let k = 0; k < a.n; k++) { ga[k] += g[k]; gb[k] += g[k]; } };
    return y;
  }
  function mul(a, b) { // elementweise (⊙)
    const out = new Float64Array(a.n); for (let k = 0; k < a.n; k++) out[k] = a.d[k] * b.d[k];
    const y = new V(out, a.shape, [a, b], 'mul');
    if (recording) y.back = () => { const g = y.g, ga = a.grad(), gb = b.grad(); for (let k = 0; k < a.n; k++) { ga[k] += g[k] * b.d[k]; gb[k] += g[k] * a.d[k]; } };
    return y;
  }
  function oneMinus(a) {
    const out = new Float64Array(a.n); for (let k = 0; k < a.n; k++) out[k] = 1 - a.d[k];
    const y = new V(out, a.shape, [a], 'oneminus');
    if (recording) y.back = () => { const g = y.g, ga = a.grad(); for (let k = 0; k < a.n; k++) ga[k] -= g[k]; };
    return y;
  }

  // ---------- Initialisierung (L4 / Übung 3)
  const INIT = {
    he:     (fanIn) => { const s = Math.sqrt(2 / fanIn); return () => randn() * s; },
    xavier: (fanIn, fanOut) => { const s = Math.sqrt(2 / (fanIn + fanOut)); return () => randn() * s; },
    gleich: () => () => 0.3,
  };

  // ---------- Optimierer (L3 / Übung 2)
  class Optimizer {
    constructor(params, cfg) { this.params = params; this.t = 0; this.set(cfg); }
    set(cfg) { this.cfg = Object.assign({ type: 'adam', lr: 0.01, l2: 0, mu: 0.9, b1: 0.9, b2: 0.999, eps: 1e-8, clip: 0 }, this.cfg || {}, cfg || {}); }
    zero() { for (const p of this.params) p.g.fill(0); }
    gradNorm(scale = 1) { let s = 0; for (const p of this.params) for (let k = 0; k < p.n; k++) s += (p.g[k] * scale) ** 2; return Math.sqrt(s); }
    step(scale = 1) {
      const c = this.cfg; this.t++;
      let clipF = 1;
      if (c.clip > 0) { const nrm = this.gradNorm(scale); if (nrm > c.clip) clipF = c.clip / nrm; }
      for (const p of this.params) {
        if (c.type !== 'sgd' && !p.m) { p.m = new Float64Array(p.n); p.v = new Float64Array(p.n); }
        for (let k = 0; k < p.n; k++) {
          let gk = p.g[k] * scale * clipF;
          if (c.l2 > 0 && p.decay) gk += 2 * c.l2 * p.d[k]; // J_R = J + λ‖θ‖²
          let step;
          if (c.type === 'sgd') step = -c.lr * gk;
          else if (c.type === 'momentum') { p.m[k] = c.mu * p.m[k] - c.lr * gk; step = p.m[k]; }
          else {
            p.m[k] = c.b1 * p.m[k] + (1 - c.b1) * gk;
            p.v[k] = c.b2 * p.v[k] + (1 - c.b2) * gk * gk;
            const mh = p.m[k] / (1 - Math.pow(c.b1, this.t)), vh = p.v[k] / (1 - Math.pow(c.b2, this.t));
            step = -c.lr * mh / (Math.sqrt(vh) + c.eps);
          }
          p.d[k] += step; p.delta[k] = step;
        }
        p.g.fill(0);
      }
    }
    reset() { this.t = 0; for (const p of this.params) { p.m = null; p.v = null; } }
  }

  return { seed, rnd, randn, shuffle, noGrad, V, tensor, param, backward, ACT, erf, sig, linear, act, dropout, softmax, softmaxCE, sumScalars, conv2d, maxpool, flatten, concat, add, mul, oneMinus, INIT, Optimizer };
})();
if (typeof module !== 'undefined') module.exports = DL;
