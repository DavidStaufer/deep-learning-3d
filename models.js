/* ============================================================
   Modelle und Daten: MLP (L2/L3), CNN (L4), RNN/LSTM/GRU (L5)
   ============================================================ */
const Models = (() => {
  const { param, tensor, linear, act, dropout, softmaxCE, softmax, conv2d, maxpool, flatten, concat, add, mul, oneMinus, sumScalars, backward, noGrad, INIT, Optimizer, rnd, randn, shuffle } = DL;

  // ======================= Daten für das MLP =======================
  const DATASETS = {
    xor:    { name: 'XOR', K: 2, text: 'Vier Wolken wie beim XOR-Problem aus L2: gleiche Vorzeichen von x₁ und x₂ ergeben Klasse 0, verschiedene Vorzeichen Klasse 1. Keine einzelne Gerade trennt das.' },
    kreise: { name: 'Kreis im Ring', K: 2, text: 'Innen Klasse 0, außen ein Ring aus Klasse 1. Die Grenze ist ein Kreis, also nicht linear.' },
    spirale:{ name: 'Spirale', K: 3, text: 'Drei Spiralarme, drei Klassen. Schwer: braucht mehrere versteckte Schichten oder viele Neuronen.' },
    wolken: { name: 'Zwei Wolken', K: 2, text: 'Zwei Wolken, die eine Gerade trennt (linear trennbar). Hier reicht sogar ein Netz ohne versteckte Schicht.' },
  };
  function makeData(key, N = 200) {
    const X = [], Y = [];
    const push = (a, b, c) => { X.push([Math.max(-1, Math.min(1, a)), Math.max(-1, Math.min(1, b))]); Y.push(c); };
    if (key === 'xor') {
      for (let i = 0; i < N; i++) {
        const sx = i % 2 ? 1 : -1, sy = (i >> 1) % 2 ? 1 : -1;
        push(sx * 0.5 + randn() * 0.16, sy * 0.5 + randn() * 0.16, sx * sy > 0 ? 0 : 1);
      }
    } else if (key === 'kreise') {
      for (let i = 0; i < N; i++) {
        const inner = i % 2 === 0, a = rnd() * Math.PI * 2;
        const r = inner ? Math.sqrt(rnd()) * 0.38 : 0.62 + rnd() * 0.3;
        push(r * Math.cos(a), r * Math.sin(a), inner ? 0 : 1);
      }
    } else if (key === 'spirale') {
      const per = Math.floor(N / 3);
      for (let c = 0; c < 3; c++) for (let i = 0; i < per; i++) {
        const r = 0.08 + 0.9 * (i / per), t = c * 2 * Math.PI / 3 + 3.6 * (i / per) + randn() * 0.18;
        push(r * Math.cos(t), r * Math.sin(t), c);
      }
    } else {
      for (let i = 0; i < N; i++) {
        const c = i % 2;
        push((c ? 0.42 : -0.42) + randn() * 0.22, (c ? 0.32 : -0.32) + randn() * 0.22, c);
      }
    }
    return { X, Y, K: DATASETS[key].K, key };
  }

  // ======================= MLP =======================
  class MLP {
    constructor(cfg) { this.build(cfg); }
    build(cfg) {
      this.cfg = Object.assign({ hidden: [6, 6], acts: null, act: 'relu', K: 2, init: 'he', opt: { type: 'adam', lr: 0.03 }, dropout: 0, batch: 16 }, cfg);
      const c = this.cfg;
      if (!c.acts) c.acts = c.hidden.map(() => c.act);
      this.sizes = [2, ...c.hidden, c.K];
      this.W = []; this.b = [];
      for (let l = 0; l < this.sizes.length - 1; l++) {
        const fi = this.sizes[l], fo = this.sizes[l + 1];
        const ini = c.init === 'xavier' ? INIT.xavier(fi, fo) : c.init === 'gleich' ? INIT.gleich() : INIT.he(fi);
        this.W.push(param([fo, fi], ini, { name: `W${l + 1}` }));
        this.b.push(param([fo], () => 0, { decay: false, name: `b${l + 1}` }));
      }
      this.params = [...this.W, ...this.b];
      this.opt = new Optimizer(this.params, c.opt);
      this.steps = 0; this.epoch = 0; this.order = []; this.pos = 0;
    }
    get L() { return this.W.length; }
    // Vorwärtsdurchlauf mit Graph: gibt alle Zwischenwerte zurück
    forward(x, label, train = false) {
      const xs = tensor(x), zs = [], hs = [xs], drops = [];
      let h = xs;
      for (let l = 0; l < this.L; l++) {
        const z = linear(this.W[l], this.b[l], h); zs.push(z);
        if (l < this.L - 1) {
          let a = act(z, this.cfg.acts[l]);
          hs.push(a);
          const d = dropout(a, this.cfg.dropout, train); drops.push(d.mask || null);
          h = d;
        }
      }
      const loss = label == null ? null : softmaxCE(zs[zs.length - 1], label);
      return { x: xs, zs, hs, drops, loss, probs: loss ? loss.probs : softmax(zs[zs.length - 1].d) };
    }
    // schnell, ohne Graph (für Karten über den Eingaberaum)
    plain(x) {
      let h = x; const zs = [], hs = [x];
      for (let l = 0; l < this.L; l++) {
        const W = this.W[l], [o, i] = W.shape, z = new Float64Array(o);
        for (let j = 0; j < o; j++) { let s = this.b[l].d[j]; for (let k = 0; k < i; k++) s += W.d[j * i + k] * h[k]; z[j] = s; }
        zs.push(z);
        if (l < this.L - 1) { const A = DL.ACT[this.cfg.acts[l]].f; const a = new Float64Array(o); for (let j = 0; j < o; j++) a[j] = A(z[j]); hs.push(a); h = a; }
      }
      return { zs, hs, probs: softmax(zs[zs.length - 1]) };
    }
    trainStep(data) {
      const N = data.X.length, B = Math.min(this.cfg.batch, N);
      let lossSum = 0;
      for (let b = 0; b < B; b++) {
        if (this.pos >= this.order.length) { this.order = shuffle([...Array(N).keys()]); this.pos = 0; if (this.steps > 0) this.epoch++; }
        const i = this.order[this.pos++];
        const r = this.forward(data.X[i], data.Y[i], true);
        backward(r.loss); lossSum += r.loss.d[0];
      }
      this.opt.step(1 / B);
      this.steps++;
      return lossSum / B;
    }
    evaluate(data) {
      let L = 0, ok = 0;
      for (let i = 0; i < data.X.length; i++) {
        const p = this.plain(data.X[i]).probs; L += -Math.log(Math.max(p[data.Y[i]], 1e-12));
        let best = 0; for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
        if (best === data.Y[i]) ok++;
      }
      return { loss: L / data.X.length, acc: ok / data.X.length };
    }
  }

  // ======================= CNN: Formen erkennen =======================
  const SHAPES = ['Senkrecht', 'Waagerecht', 'Kreuz', 'Ring'];
  const IMG = 16;
  function drawLine(img, vertical, pos, from, to, thick, val) {
    for (let t = from; t <= to; t++) for (let w = 0; w < thick; w++) {
      const r = vertical ? t : pos + w, c = vertical ? pos + w : t;
      if (r >= 0 && r < IMG && c >= 0 && c < IMG) img[r * IMG + c] = Math.max(img[r * IMG + c], val);
    }
  }
  function makeShape(cls) {
    const img = new Float64Array(IMG * IMG), val = 0.75 + rnd() * 0.25, thick = 1 + Math.floor(rnd() * 2.2);
    const ri = (a, b) => a + Math.floor(rnd() * (b - a + 1));
    if (cls === 0 || cls === 1) {
      const pos = ri(2, 13 - thick), len = ri(8, 14), from = ri(0, IMG - len);
      drawLine(img, cls === 0, pos, from, from + len - 1, thick, val);
    } else if (cls === 2) {
      const cr = ri(5, 10), cc = ri(5, 10), a = ri(3, 6), b = ri(3, 6);
      drawLine(img, true, cc, cr - a, cr + a, thick, val);
      drawLine(img, false, cr, cc - b, cc + b, thick, val);
    } else {
      const cr = 7.5 + (rnd() - 0.5) * 4, cc = 7.5 + (rnd() - 0.5) * 4, R = 3 + rnd() * 2.8, w = 0.55 + thick * 0.35;
      for (let r = 0; r < IMG; r++) for (let c = 0; c < IMG; c++) {
        const d = Math.abs(Math.hypot(r - cr, c - cc) - R);
        if (d < w) img[r * IMG + c] = val * Math.min(1, (w - d) * 1.6);
      }
    }
    for (let k = 0; k < img.length; k++) img[k] = Math.max(0, Math.min(1, img[k] + randn() * 0.05));
    return img;
  }

  class CNN {
    constructor() { this.reset(); }
    reset() {
      const he = INIT.he;
      this.K1 = param([4, 1, 3, 3], he(9), { name: 'K1' }); this.b1 = param([4], () => 0, { decay: false });
      this.K2 = param([8, 4, 3, 3], he(36), { name: 'K2' }); this.b2 = param([8], () => 0, { decay: false });
      this.W3 = param([16, 128], he(128), { name: 'W3' }); this.b3 = param([16], () => 0, { decay: false });
      this.W4 = param([4, 16], he(16), { name: 'W4' }); this.b4 = param([4], () => 0, { decay: false });
      this.params = [this.K1, this.b1, this.K2, this.b2, this.W3, this.b3, this.W4, this.b4];
      this.opt = new Optimizer(this.params, { type: 'adam', lr: 0.004 });
      this.p = 0.3; this.steps = 0; this.hist = [];
    }
    forward(img, label, train = false) {
      const x = tensor(img, [1, IMG, IMG]);
      const c1 = conv2d(x, this.K1, this.b1, 1), r1 = act(c1, 'relu'), p1 = maxpool(r1, 2);
      const c2 = conv2d(p1, this.K2, this.b2, 1), r2 = act(c2, 'relu'), p2 = maxpool(r2, 2);
      const f = flatten(p2);
      const z3 = linear(this.W3, this.b3, f), h3 = act(z3, 'relu'), d3 = dropout(h3, this.p, train);
      const z4 = linear(this.W4, this.b4, d3);
      const loss = label == null ? null : softmaxCE(z4, label);
      return { x, c1, r1, p1, c2, r2, p2, f, z3, h3, d3, z4, loss, probs: loss ? loss.probs : softmax(z4.d) };
    }
    trainStep(B = 16) {
      let L = 0, ok = 0;
      for (let b = 0; b < B; b++) {
        const cls = Math.floor(rnd() * 4), r = this.forward(makeShape(cls), cls, true);
        backward(r.loss); L += r.loss.d[0];
        let best = 0; for (let k = 1; k < 4; k++) if (r.probs[k] > r.probs[best]) best = k;
        if (best === cls) ok++;
      }
      this.opt.step(1 / B); this.steps++;
      return { loss: L / B, acc: ok / B };
    }
  }

  // ======================= RNN / LSTM / GRU =======================
  class RNN {
    constructor(type, text) { this.setText(text || 'deep learning '); this.build(type || 'elman'); }
    setText(text) {
      this.text = text; this.vocab = [...new Set(text.split(''))].sort();
      this.V = this.vocab.length; this.idx = Object.fromEntries(this.vocab.map((c, i) => [c, i]));
    }
    build(type) {
      this.type = type; const V = this.V, H = this.H = type === 'elman' ? 16 : 12;
      const P = (o, i, name) => param([o, i], INIT.xavier(i, o), { name });
      const B = (n, name) => param([n], () => 0, { decay: false, name });
      if (type === 'elman') {
        this.Whh = P(H, H, 'Whh'); this.Wxh = P(H, V, 'Wxh'); this.bh = B(H, 'bh');
        this.params = [this.Whh, this.Wxh, this.bh];
      } else if (type === 'lstm') {
        this.Wf = P(H, H + V, 'Wf'); this.bf = B(H, 'bf'); this.Wi = P(H, H + V, 'Wi'); this.bi = B(H, 'bi');
        this.WC = P(H, H + V, 'WC'); this.bC = B(H, 'bC'); this.Wo = P(H, H + V, 'Wo'); this.bo = B(H, 'bo');
        this.params = [this.Wf, this.bf, this.Wi, this.bi, this.WC, this.bC, this.Wo, this.bo];
      } else {
        this.Wr = P(H, H + V, 'Wr'); this.br = B(H, 'br'); this.Wz = P(H, H + V, 'Wz'); this.bz = B(H, 'bz');
        this.Wh = P(H, H + V, 'Wh'); this.bh = B(H, 'bh');
        this.params = [this.Wr, this.br, this.Wz, this.bz, this.Wh, this.bh];
      }
      this.Why = P(V, H, 'Why'); this.by = B(V, 'by');
      this.params.push(this.Why, this.by);
      this.opt = new Optimizer(this.params, { type: 'adam', lr: 0.02, clip: 5 });
      this.steps = 0;
    }
    onehot(ch) { const a = new Float64Array(this.V); a[this.idx[ch]] = 1; return tensor(a); }
    // eine Zelle: gibt alle Zwischenwerte zurück
    cell(xt, hPrev, CPrev) {
      if (this.type === 'elman') {
        const u = add(linear(this.Whh, this.bh, hPrev), linear(this.Wxh, null, xt));
        return { u, h: act(u, 'tanh') };
      }
      const hx = concat(hPrev, xt);
      if (this.type === 'lstm') {
        const f = act(linear(this.Wf, this.bf, hx), 'sigmoid');
        const i = act(linear(this.Wi, this.bi, hx), 'sigmoid');
        const Ct_ = act(linear(this.WC, this.bC, hx), 'tanh');
        const C = add(mul(f, CPrev), mul(i, Ct_));
        const o = act(linear(this.Wo, this.bo, hx), 'sigmoid');
        const h = mul(o, act(C, 'tanh'));
        return { hx, f, i, Ct_, C, o, h };
      }
      const r = act(linear(this.Wr, this.br, hx), 'sigmoid');
      const z = act(linear(this.Wz, this.bz, hx), 'sigmoid');
      const ht_ = act(linear(this.Wh, this.bh, concat(mul(r, hPrev), xt)), 'tanh');
      const h = add(mul(oneMinus(z), hPrev), mul(z, ht_));
      return { hx, r, z, ht_, h };
    }
    // Sequenz ab Position s, Länge T. lossMode: 'mean' (alle Schritte) oder 'last' (nur letzter)
    run(s, T, lossMode = 'mean') {
      const N = this.text.length, zero = () => tensor(new Float64Array(this.H));
      let h = zero(), C = zero();
      const h0 = h, steps = [], losses = [];
      for (let t = 0; t < T; t++) {
        const ch = this.text[(s + t) % N], nx = this.text[(s + t + 1) % N];
        const xt = this.onehot(ch), c = this.cell(xt, h, C);
        const logits = linear(this.Why, this.by, c.h), L = softmaxCE(logits, this.idx[nx]);
        steps.push(Object.assign({ ch, nx, xt, logits, L, probs: L.probs, hPrev: h, CPrev: C }, c));
        losses.push(L);
        h = c.h; if (c.C) C = c.C;
      }
      const loss = lossMode === 'last' ? sumScalars([losses[T - 1]], 1) : sumScalars(losses, 1 / T);
      return { steps, loss, h0 };
    }
    trainStep(B = 4, T = 10) {
      let L = 0;
      for (let b = 0; b < B; b++) { const r = this.run(Math.floor(rnd() * this.text.length), T); backward(r.loss); L += r.loss.d[0]; }
      this.opt.step(1 / B); this.steps++;
      return L / B;
    }
    evaluate(T = 10) {
      return noGrad(() => {
        let ok = 0, n = 0, L = 0;
        for (let s = 0; s < this.text.length; s++) {
          const r = this.run(s, T);
          for (const st of r.steps) { n++; L += st.L.d[0]; let best = 0; for (let k = 1; k < this.V; k++) if (st.probs[k] > st.probs[best]) best = k; if (best === this.idx[st.nx]) ok++; }
        }
        return { acc: ok / n, loss: L / n };
      });
    }
    generate(n = 40) {
      return noGrad(() => {
        let h = tensor(new Float64Array(this.H)), C = tensor(new Float64Array(this.H)), ch = this.text[0], out = ch;
        for (let k = 0; k < n; k++) {
          const c = this.cell(this.onehot(ch), h, C);
          const p = softmax(linear(this.Why, this.by, c.h).d);
          let best = 0; for (let q = 1; q < p.length; q++) if (p[q] > p[best]) best = q;
          ch = this.vocab[best]; out += ch; h = c.h; if (c.C) C = c.C;
        }
        return out;
      });
    }
  }

  return { DATASETS, makeData, MLP, SHAPES, IMG, makeShape, CNN, RNN };
})();
if (typeof module !== 'undefined') module.exports = Models;
