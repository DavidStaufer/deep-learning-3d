/* ============================================================
   Modus 3: Rekurrente Netze – Elman-RNN, LSTM, GRU (L5)
   ============================================================ */
const RNN_DX = 2.9, RNN_DZ = 0.24;
const RNN_ROWS = {
  elman: [{ k: 'x', y: 0 }, { k: 'h', y: 2.3 }, { k: 'y', y: 4.6 }],
  lstm: [{ k: 'x', y: 0 }, { k: 'f', y: 1.7, dx: -0.9 }, { k: 'i', y: 1.7, dx: -0.3 }, { k: 'Ct', y: 1.7, dx: 0.3 }, { k: 'o', y: 1.7, dx: 0.9 }, { k: 'C', y: 3.1 }, { k: 'h', y: 4.3 }, { k: 'y', y: 6.1 }],
  gru: [{ k: 'x', y: 0 }, { k: 'r', y: 1.7, dx: -0.65 }, { k: 'z', y: 1.7, dx: 0 }, { k: 'ht', y: 1.7, dx: 0.65 }, { k: 'h', y: 3.3 }, { k: 'y', y: 5.1 }],
};
const ROWINFO = {
  x: { sym: 'x', html: 'x<sub>t</sub>', name: 'Eingabe', text: 'Das Zeichen zum Zeitpunkt \\(t\\) als One-hot-Vektor: eine \\(1\\) an der Stelle des Zeichens, sonst \\(0\\).' },
  h: { sym: 'h', html: 'h<sub>t</sub>', name: 'Versteckter Zustand', text: 'Der Hidden State ist das Gedächtnis des Netzes. Er wird an den nächsten Zeitschritt weitergegeben.' },
  y: { sym: '\\hat{y}', html: 'ŷ<sub>t</sub>', name: 'Vorhersage', text: 'Wahrscheinlichkeit für jedes mögliche nächste Zeichen (Softmax).' },
  f: { sym: 'f', html: 'f<sub>t</sub>', name: 'Forget Gate', text: 'Wie viel vom alten Zellzustand \\(C_{t-1}\\) bleibt: \\(1\\) = alles behalten, \\(0\\) = vergessen.' },
  i: { sym: 'i', html: 'i<sub>t</sub>', name: 'Input Gate', text: 'Wie viel vom neuen Vorschlag \\(\\tilde{C}_t\\) in den Zellzustand kommt.' },
  Ct: { sym: '\\tilde{C}', html: 'C̃<sub>t</sub>', name: 'Vorschlag', text: 'Neuer Inhalt für den Zellzustand, zwischen \\(-1\\) und \\(1\\) (tanh).' },
  o: { sym: 'o', html: 'o<sub>t</sub>', name: 'Output Gate', text: 'Wie viel vom Zellzustand als \\(h_t\\) nach außen geht.' },
  C: { sym: 'C', html: 'C<sub>t</sub>', name: 'Zellzustand', text: 'Das Langzeitgedächtnis. Es läuft wie ein Förderband durch die Zeit und wird nur mit \\(\\odot\\) und \\(+\\) verändert. Darum wird der Gradient hier langsamer klein als im einfachen RNN.' },
  r: { sym: 'r', html: 'r<sub>t</sub>', name: 'Reset Gate', text: 'Wie stark der alte Zustand \\(h_{t-1}\\) in den Vorschlag \\(\\tilde{h}_t\\) eingeht.' },
  z: { sym: 'z', html: 'z<sub>t</sub>', name: 'Update Gate', text: 'Mischt alt und neu: Anteil \\(1 - z_t\\) vom alten \\(h_{t-1}\\), Anteil \\(z_t\\) vom Vorschlag \\(\\tilde{h}_t\\).' },
  ht: { sym: '\\tilde{h}', html: 'h̃<sub>t</sub>', name: 'Vorschlag', text: 'Vorschlag für den neuen Hidden State, zwischen \\(-1\\) und \\(1\\).' },
};
const RNN_EQ = {
  elman: ['\\mathbf{h}_t = \\tanh(\\mathbf{u}_t) = \\tanh(\\mathbf{W}_{hh}\\mathbf{h}_{t-1} + \\mathbf{W}_{xh}\\mathbf{x}_t + \\mathbf{b}_h)', '\\hat{\\mathbf{y}}_t = \\sigma(\\mathbf{W}_{hy}\\mathbf{h}_t + \\mathbf{b}_y)'],
  lstm: ['\\mathbf{f}_t = \\sigma(\\mathbf{W}_f \\cdot [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_f)', '\\mathbf{i}_t = \\sigma(\\mathbf{W}_i \\cdot [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_i)', '\\tilde{\\mathbf{C}}_t = \\tanh(\\mathbf{W}_C \\cdot [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_C)', '\\mathbf{C}_t = \\mathbf{f}_t \\odot \\mathbf{C}_{t-1} + \\mathbf{i}_t \\odot \\tilde{\\mathbf{C}}_t', '\\mathbf{o}_t = \\sigma(\\mathbf{W}_o \\cdot [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_o)', '\\mathbf{h}_t = \\mathbf{o}_t \\odot \\tanh(\\mathbf{C}_t)', '\\hat{\\mathbf{y}}_t = \\sigma(\\mathbf{W}_{hy}\\mathbf{h}_t + \\mathbf{b}_y)'],
  gru: ['\\mathbf{r}_t = \\sigma(\\mathbf{W}_r \\cdot [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_r)', '\\mathbf{z}_t = \\sigma(\\mathbf{W}_z \\cdot [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_z)', '\\tilde{\\mathbf{h}}_t = \\tanh(\\mathbf{W}_h \\cdot [\\mathbf{r}_t \\odot \\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_h)', '\\mathbf{h}_t = (1 - \\mathbf{z}_t) \\odot \\mathbf{h}_{t-1} + \\mathbf{z}_t \\odot \\tilde{\\mathbf{h}}_t', '\\hat{\\mathbf{y}}_t = \\sigma(\\mathbf{W}_{hy}\\mathbf{h}_t + \\mathbf{b}_y)'],
};
const CELLNAME = { elman: 'Elman-RNN', lstm: 'LSTM', gru: 'GRU' };
const chr = (c) => (c === ' ' ? '␣' : esc(c));

class RNNMode {
  constructor() {
    this.key = 'rnn'; this.group = new THREE.Group(); this.built = false;
    this.type = 'lstm'; this.text = 'deep learning '; this.T = 8; this.s = 0; this.playing = false; this.anim = null;
    this.hist = []; this.ev = null; this.lossEMA = null; this.flowView = false; this.gen = ''; this.focusT = 7; this.seedN = 31;
    this.revealT = Infinity; this.gradT = Infinity; this.autoStop = 0; this.lastLines = 0; this.lastLive = 0;
  }
  init() { this.newNet(); this.build(); this.trace(); this.playing = true; this.autoStop = 450; }
  enter() { if (!this.rnn.steps && !this.playing) { this.playing = true; this.autoStop = 450; } }
  newNet() {
    DL.seed(this.seedN); this.rnn = new Models.RNN(this.type, this.text);
    this.hist = []; this.lossEMA = null; this.ev = this.rnn.evaluate(); this.gen = ''; this.anim = null; App.caption(null);
  }
  trace() {
    const zero = () => { for (const p of this.rnn.params) p.g.fill(0); };
    const r = this.rnn.run(this.s, this.T); DL.backward(r.loss); this.tr = r; zero();
    const r2 = this.rnn.run(this.s, this.T, 'last'); DL.backward(r2.loss);
    const nrm = (v) => (v && v.g ? Math.sqrt(v.g.reduce((a, b) => a + b * b, 0)) : 0);
    this.flow = r2.steps.map((st) => nrm(st.h)); this.flowC = this.type === 'lstm' ? r2.steps.map((st) => nrm(st.C)) : null;
    this.flowUnits = r2.steps.map((st) => (st.h.g ? Float64Array.from(st.h.g) : new Float64Array(this.rnn.H)));
    zero();
  }
  // Werte einer Zeile zum Zeitpunkt t
  rowVals(t, k) {
    const st = this.tr.steps[t];
    if (k === 'x') return st.xt.d; if (k === 'y') return st.probs; if (k === 'h') return st.h.d; if (k === 'C') return st.C.d;
    if (k === 'Ct') return st.Ct_.d; if (k === 'ht') return st.ht_.d; return st[k].d;
  }
  rowN(k) { return k === 'x' || k === 'y' ? this.rnn.V : this.rnn.H; }
  pos(t, row, k, out = V3()) {
    const R = this.rows.find((r) => r.k === row), n = this.rowN(row);
    return out.set((t - (this.T - 1) / 2) * RNN_DX + (R.dx || 0), R.y, (k - (n - 1) / 2) * RNN_DZ);
  }
  view() {
    const top = this.rows[this.rows.length - 1].y + 1.6, zl = (Math.max(this.rnn.H, this.rnn.V) * RNN_DZ) / 2;
    return viewFor(V3(-(this.T / 2) * RNN_DX - 1, -0.8, -zl), V3((this.T / 2) * RNN_DX, top, zl), App.phone() ? V3(0.75, 0.45, 0.7) : V3(0.32, 0.4, 1));
  }
  build() {
    disposeGroup(this.group);
    const g = this.group, nT = this.T, H = this.rnn.H, V = this.rnn.V;
    this.rows = RNN_ROWS[this.type];
    // Würfel für alle Vektoren außer ŷ
    this.cubes = [];
    for (let t = 0; t < nT; t++) for (const R of this.rows) if (R.k !== 'y') for (let k = 0; k < this.rowN(R.k); k++) this.cubes.push({ t, row: R.k, k });
    this.cMesh = instanced(new THREE.BoxGeometry(0.17, 0.17, 0.2), new THREE.MeshStandardMaterial({ roughness: 0.5 }), this.cubes.length);
    this.cubes.forEach((c, i) => setInst(this.cMesh, i, this.pos(c.t, c.row, c.k), 1));
    this.cMesh.userData.pick = (h) => { const c = this.cubes[h.instanceId]; return c ? { kind: 'unit', t: c.t, row: c.row, k: c.k } : null; };
    this.yMesh = instanced(new THREE.BoxGeometry(0.17, 1, 0.2), new THREE.MeshStandardMaterial({ roughness: 0.45 }), nT * V);
    this.yMesh.userData.pick = (h) => ({ kind: 'unit', t: Math.floor(h.instanceId / V), row: 'y', k: h.instanceId % V });
    g.add(this.cMesh, this.yMesh);
    // Schienen: der Zustand läuft durch die Zeit
    const xl = (nT - 1) * RNN_DX + 1.2;
    const zb = -(H * RNN_DZ) / 2 - 0.32;
    const rail = (y, col) => {
      const mat = new THREE.MeshStandardMaterial({ color: col, roughness: 0.4 });
      const m = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, xl, 12), mat); m.rotation.z = Math.PI / 2; m.position.set(0, y, zb); g.add(m);
      for (let t = 1; t < nT; t++) { const c = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.26, 16), mat); c.rotation.z = -Math.PI / 2; c.position.set((t - 0.5 - (nT - 1) / 2) * RNN_DX, y, zb); g.add(c); }
      return m;
    };
    this.rails = [rail(this.rows.find((r) => r.k === 'h').y, T.c.pos)];
    if (this.type === 'lstm') this.rails.push(rail(this.rows.find((r) => r.k === 'C').y, T.c.c2));
    // Gradientenfluss-Säulen
    this.fMesh = instanced(new THREE.BoxGeometry(0.34, 1, 0.34), new THREE.MeshStandardMaterial({ roughness: 0.4 }), nT);
    this.fMesh.userData.pick = () => ({ kind: 'flow' });
    g.add(this.fMesh);
    // Linien für den gewählten Zeitschritt
    this.lines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true }));
    g.add(this.lines);
    this.box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.26, 0.26, 0.3)), new THREE.LineBasicMaterial({ color: T.c.text }));
    this.box.visible = false; g.add(this.box);
    g.add(shadowDisc(nT * RNN_DX + 4, 7, -0.9));
    // Beschriftungen
    const zl = Math.max(H, V) * RNN_DZ / 2;
    this.stepLbl = []; this.inLbl = []; this.outLbl = []; this.flowLbl = [];
    const topY = this.rows[this.rows.length - 1].y;
    for (let t = 0; t < nT; t++) {
      const x = (t - (nT - 1) / 2) * RNN_DX;
      const s = label(`t = ${t + 1}`, 'lbl', () => App.select({ kind: 'step', t })); s.position.set(x, topY + 1.75, 0); g.add(s); this.stepLbl.push(s);
      const a = label('', 'zeichen', () => App.select({ kind: 'unit', t, row: 'x', k: this.rnn.idx[this.tr.steps[t].ch] })); a.position.set(x, -0.45, V * RNN_DZ / 2 + 0.25); g.add(a); this.inLbl.push(a);
      const b = label('', 'zeichen', () => App.select({ kind: 'unit', t, row: 'y', k: this.argmax(t) })); b.position.set(x, topY + 1.35, V * RNN_DZ / 2 + 0.2); g.add(b); this.outLbl.push(b);
      const f = label('', 'wert'); f.position.set(x, this.rows.find((r) => r.k === 'h').y - 0.5, zl + 0.75); g.add(f); this.flowLbl.push(f);
      if (t > 0) { const w = label(this.type === 'elman' ? 'W<sub>hh</sub>' : 'h<sub>t−1</sub> →', 'wert'); w.position.set(x - RNN_DX / 2, this.rows.find((r) => r.k === 'h').y + 0.26, zb); g.add(w); }
    }
    const x0 = -(nT - 1) / 2 * RNN_DX - 1.35, levels = [];
    for (const R of this.rows) if (!levels.some((l) => l.y === R.y)) levels.push({ y: R.y, rows: this.rows.filter((q) => q.y === R.y) });
    for (const lv of levels) {
      const R = lv.rows[0], gates = lv.rows.length > 1;
      const o = label(gates ? 'Tore <small>Gates</small>' : `${ROWINFO[R.k].html} <small>${ROWINFO[R.k].name}</small>`, 'lbl', () => App.select({ kind: 'row', row: R.k }));
      o.position.set(x0, lv.y, 0); g.add(o);
      if (gates) for (const Q of lv.rows) { const q = label(ROWINFO[Q.k].html, 'lbl', () => App.select({ kind: 'row', row: Q.k })); q.position.copy(this.pos(0, Q.k, this.rowN(Q.k) - 1)).add(V3(0, 0.34, 0.25)); g.add(q); }
    }
    if (this.type === 'lstm') { const o = label('Förderband C<sub>t</sub>', 'wert'); o.position.set(-(nT - 1) / 2 * RNN_DX + RNN_DX / 2, this.rows.find((r) => r.k === 'C').y + 0.3, zb); g.add(o); }
    this.pickables = [this.cMesh, this.yMesh, this.fMesh];
    this.linesFor = -1; this.lastFlowView = null;
  }
  argmax(t) { const p = this.tr.steps[t].probs; let b = 0; for (let k = 1; k < p.length; k++) if (p[k] > p[b]) b = k; return b; }
  buildLines() {
    const t = this.focusT, H = this.rnn.H, V = this.rnn.V, segs = [];
    const P = (row, k, tt = t) => this.pos(tt, row, k);
    const W = (M, k, m, n) => () => M.d[k * n + m];
    const R = this.rnn;
    if (this.type === 'elman') {
      for (let k = 0; k < H; k++) { for (let m = 0; m < V; m++) segs.push([P('x', m), P('h', k), W(R.Wxh, k, m, V)]); if (t > 0) for (let m = 0; m < H; m++) segs.push([P('h', m, t - 1), P('h', k), W(R.Whh, k, m, H)]); }
    } else {
      const gates = this.type === 'lstm' ? [['f', R.Wf], ['i', R.Wi], ['Ct', R.WC], ['o', R.Wo]] : [['r', R.Wr], ['z', R.Wz], ['ht', R.Wh]];
      for (const [gk, M] of gates) for (let k = 0; k < H; k++) {
        for (let m = 0; m < V; m++) segs.push([P('x', m), P(gk, k), W(M, k, H + m, H + V)]);
        if (t > 0) for (let m = 0; m < H; m++) segs.push([P('h', m, t - 1), P(gk, k), W(M, k, m, H + V)]);
      }
      const el = this.type === 'lstm' ? [['f', 'C'], ['i', 'C'], ['Ct', 'C'], ['o', 'h'], ['C', 'h']] : [['z', 'h'], ['ht', 'h']];
      for (const [a, b] of el) for (let k = 0; k < H; k++) segs.push([P(a, k), P(b, k), null]);
      if (t > 0) for (let k = 0; k < H; k++) segs.push([P(this.type === 'lstm' ? 'C' : 'h', k, t - 1), P(this.type === 'lstm' ? 'C' : 'h', k), null]);
    }
    for (let k = 0; k < V; k++) for (let m = 0; m < H; m++) segs.push([P('h', m), P('y', k), W(R.Why, k, m, H)]);
    const pos = new Float32Array(segs.length * 6);
    segs.forEach(([a, b], i) => { pos.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6); });
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(segs.length * 6), 3));
    this.lines.geometry.dispose(); this.lines.geometry = geo; this.segs = segs; this.linesFor = t;
    this.colorLines();
  }
  colorLines() {
    const col = this.lines.geometry.attributes.color, c = new THREE.Color(), sel = App.sel;
    const fu = sel && sel.kind === 'unit' && sel.t === this.focusT ? sel : null;
    this.segs.forEach(([a, b, w], i) => {
      if (w) { const v = w(); valueColor(Math.sign(v) * Math.min(1, Math.abs(v) / 0.8), c); } else c.copy(T.c['text-3']);
      let on = true;
      if (fu) { const p = this.pos(fu.t, fu.row, fu.k); on = (Math.abs(b.x - p.x) < 1e-6 && Math.abs(b.y - p.y) < 1e-6 && Math.abs(b.z - p.z) < 1e-6) || (Math.abs(a.x - p.x) < 1e-6 && Math.abs(a.y - p.y) < 1e-6 && Math.abs(a.z - p.z) < 1e-6); }
      c.lerp(T.c.bg, on ? (fu ? 0.05 : 0.72) : 0.96);
      col.setXYZ(2 * i, c.r, c.g, c.b); col.setXYZ(2 * i + 1, c.r, c.g, c.b);
    });
    col.needsUpdate = true;
  }
  frame(dt, now) {
    if (this.anim) this.runAnim(now);
    else if (this.playing) {
      const t0 = performance.now(); let n = 0;
      while (performance.now() - t0 < 10 && n < 20) { const L = this.rnn.trainStep(4, 10); n++; this.lossEMA = this.lossEMA === null ? L : 0.96 * this.lossEMA + 0.04 * L; if (this.rnn.steps % 5 === 0) { this.hist.push(this.lossEMA); if (this.hist.length > 150) this.hist.shift(); } }
      if (this.rnn.steps % 25 < n) this.ev = this.rnn.evaluate();
      this.trace(); App.liveDirty = true;
      if (this.autoStop && this.rnn.steps >= this.autoStop) { this.playing = false; this.autoStop = 0; this.ev = this.rnn.evaluate(); this.gen = this.rnn.generate(42); App.updateDock(); App.liveDirty = true; }
    }
    this.updateVisuals(now);
  }
  updateVisuals(now) {
    const c = new THREE.Color(), H = this.rnn.H, V = this.rnn.V, pos = V3(), sel = App.sel;
    if (this.linesFor !== this.focusT) this.buildLines();
    if (now - this.lastLines > 0.3) { this.lastLines = now; this.colorLines(); }
    let cm = 1; for (let t = 0; t < this.T; t++) if (this.tr.steps[t].C) for (const v of this.tr.steps[t].C.d) cm = Math.max(cm, Math.abs(v));
    let gmax = 1e-12; if (this.flowView) for (const u of this.flowUnits) for (const v of u) gmax = Math.max(gmax, Math.abs(v));
    this.cubes.forEach((cu, i) => {
      const shown = cu.t < this.revealT, gr = cu.t >= this.gradT && this.gradT < Infinity;
      const v = this.rowVals(cu.t, cu.row)[cu.k];
      if (!shown) c.copy(T.c.null);
      else if (this.flowView && cu.row === 'h') c.copy(T.c.null).lerp(T.c.grad, Math.min(1, Math.sqrt(Math.abs(this.flowUnits[cu.t][cu.k]) / gmax)));
      else if (gr && cu.row === 'h') { const g = this.tr.steps[cu.t].h.g; c.copy(T.c.null).lerp(T.c.grad, g ? Math.min(1, Math.abs(g[cu.k]) * 6) : 0); }
      else if (cu.row === 'x') c.copy(T.c.null).lerp(T.c.tinte, v);
      else if (cu.row === 'f' || cu.row === 'i' || cu.row === 'o' || cu.row === 'r' || cu.row === 'z') c.copy(T.c.null).lerp(T.c.gut, v);
      else valueColor(cu.row === 'C' ? v / cm : v, c);
      this.cMesh.setColorAt(i, c);
    });
    this.cMesh.instanceColor.needsUpdate = true;
    for (let t = 0; t < this.T; t++) {
      const st = this.tr.steps[t], tgt = this.rnn.idx[st.nx], shown = t < this.revealT;
      for (let k = 0; k < V; k++) {
        const h = shown ? 0.03 + st.probs[k] * 1.25 : 0.03; this.pos(t, 'y', k, pos); pos.y += h / 2;
        setInst(this.yMesh, t * V + k, pos, 1, h, 1);
        c.copy(k === tgt ? T.c.gut : T.c.pos); if (!shown) c.copy(T.c.null);
        this.yMesh.setColorAt(t * V + k, c);
      }
      // Gradientenfluss
      const fv = this.flow[t], hgt = this.flowView ? 0.05 + 1.5 * clamp((Math.log10(fv + 1e-12) + 5) / 5, 0, 1) : 0;
      const hy = this.rows.find((r) => r.k === 'h').y, zl = Math.max(H, V) * RNN_DZ / 2;
      pos.set((t - (this.T - 1) / 2) * RNN_DX, hy - 0.2 + hgt / 2, zl + 0.75); setInst(this.fMesh, t, pos, 1, Math.max(0.001, hgt), 1); this.fMesh.setColorAt(t, T.c.grad);
      this.flowLbl[t].visible = this.flowView;
      const ft = this.flowView ? sci(fv) : ''; if (this.flowLbl[t].element.innerHTML !== ft) this.flowLbl[t].element.innerHTML = ft;
      // Zeichen
      const inT = chr(st.ch); if (this.inLbl[t].element.innerHTML !== inT) this.inLbl[t].element.innerHTML = inT;
      const am = this.argmax(t), outT = shown ? `${chr(this.rnn.vocab[am])} <span style="font-weight:500;font-size:11px">${num(st.probs[am])}</span>` : '?';
      if (this.outLbl[t].element.innerHTML !== outT) this.outLbl[t].element.innerHTML = outT;
      this.outLbl[t].element.className = 'zeichen ' + (!shown ? 'leise' : am === tgt ? 'ziel' : 'falsch');
      this.stepLbl[t].element.classList.toggle('aktiv', t === this.focusT);
    }
    this.yMesh.instanceMatrix.needsUpdate = true; this.yMesh.instanceColor.needsUpdate = true;
    this.fMesh.instanceMatrix.needsUpdate = true; this.fMesh.instanceColor.needsUpdate = true;
    if (sel && sel.kind === 'unit') { this.box.visible = true; this.pos(sel.t, sel.row, sel.k, this.box.position); if (sel.row === 'y') this.box.position.y += 0.1; } else this.box.visible = false;
  }
  // ---------------- Animation durch die Zeit
  startAnim(kind) {
    if (this.anim) return;
    this.playing = false;
    const ph = [], nT = this.T;
    if (kind !== 'backward') for (let t = 0; t < nT; t++) ph.push({ type: 'fwd', t, dur: 0.75 });
    if (kind !== 'forward') for (let t = nT - 1; t >= 0; t--) ph.push({ type: 'bwd', t, dur: 0.5 });
    if (kind === 'step') ph.push({ type: 'update', dur: 1.3 });
    let tt = 0; for (const p of ph) { p.t = tt; tt += p.dur; }
    this.anim = { kind, ph, idx: -1, t0: performance.now() / 1000, end: tt + 0.3 };
    this.revealT = kind === 'backward' ? Infinity : 0; this.gradT = Infinity;
    App.updateDock();
  }
  runAnim(now) {
    const a = this.anim, t = now - a.t0;
    while (a.idx + 1 < a.ph.length && t >= a.ph[a.idx + 1].t) { a.idx++; this.enterPhase(a.ph[a.idx]); }
    if (t >= a.end) { this.anim = null; this.revealT = Infinity; this.gradT = Infinity; App.caption(null); App.updateDock(); App.liveDirty = true; }
  }
  enterPhase(p) {
    const st = this.tr.steps[p.t], H = this.rnn.H, V = this.rnn.V, c = new THREE.Color();
    if (p.type === 'fwd') {
      this.revealT = p.t + 1; this.focusT = p.t;
      const m = this.rnn.idx[st.ch], first = this.rows[1].k;
      for (let k = 0; k < H; k++) {
        Pulses.add(this.pos(p.t, 'x', m), this.pos(p.t, first, k), 0, 0.45, T.c.tinte, 0.035);
        if (p.t > 0) Pulses.add(this.pos(p.t - 1, 'h', k), this.pos(p.t, this.type === 'elman' ? 'h' : first, k), 0, 0.45, valueColor(this.tr.steps[p.t - 1].h.d[k], c), 0.035);
      }
      const am = this.argmax(p.t);
      const how = { elman: `h<sub>${p.t + 1}</sub> = tanh(W<sub>hh</sub> h<sub>${p.t}</sub> + W<sub>xh</sub> x<sub>${p.t + 1}</sub> + b<sub>h</sub>)`, lstm: `Tore f, i, o und Vorschlag C̃ aus [h<sub>${p.t}</sub>, x<sub>${p.t + 1}</sub>], dann C<sub>${p.t + 1}</sub> und h<sub>${p.t + 1}</sub>`, gru: `Tore r, z und Vorschlag h̃ aus [h<sub>${p.t}</sub>, x<sub>${p.t + 1}</sub>], dann h<sub>${p.t + 1}</sub>` }[this.type];
      App.caption(`<b>t = ${p.t + 1}</b>: Eingabe „${chr(st.ch)}“. ${how}. Vorhersage: „${chr(this.rnn.vocab[am])}“ (${num(st.probs[am])}), richtig wäre „${chr(st.nx)}“.`);
    } else if (p.type === 'bwd') {
      this.revealT = Infinity; this.gradT = p.t; this.focusT = p.t;
      if (p.t > 0) for (let k = 0; k < H; k++) Pulses.add(this.pos(p.t, 'h', k), this.pos(p.t - 1, 'h', k), 0, 0.42, T.c.grad, 0.03);
      if (p.t === this.T - 1) App.caption('<b>Backpropagation Through Time (BPTT)</b>: der Loss ist der Mittelwert über alle Zeitschritte. Das Fehlersignal fließt von hinten nach vorne durch die Zeit zurück, in jedem Schritt durch dieselben Gewichte.');
    } else if (p.type === 'update') {
      const L = this.rnn.trainStep(4, 10); this.lossEMA = this.lossEMA === null ? L : 0.9 * this.lossEMA + 0.1 * L; this.hist.push(this.lossEMA);
      this.trace(); this.ev = this.rnn.evaluate();
      App.caption('<b>Update</b> (Adam, Gradient-Clipping bei Norm 5): die Gradienten aller Zeitschritte werden für jede Gewichtsmatrix addiert, denn es ist in jedem Schritt dieselbe Matrix. Dann ein Schritt θ ← θ − η · ∇θJ.');
    }
  }
  toggleTrain() { if (this.anim) return; this.playing = !this.playing; this.autoStop = 0; }
  stepOnce() { this.startAnim('step'); }
  animForward() { this.startAnim('forward'); }
  animBackward() { this.startAnim('backward'); }
  reinit() { this.seedN++; this.newNet(); this.build(); this.trace(); App.renderPanel(); }
  stats() {
    const e = this.ev;
    return `<span class="lang">Schritt <b>${this.rnn.steps}</b> · </span>Loss <b>${this.lossEMA === null ? '–' : num(this.lossEMA, 3)}</b> · <span class="lang">nächstes Zeichen richtig </span><b>${e ? pct(e.acc) : '–'}</b>`;
  }
  // ---------------- Inspektor
  onSelect(s) { if (s && (s.kind === 'unit' || s.kind === 'step')) this.focusT = s.t; this.lastLines = 0; }
  where(s) { if (s.kind === 'unit') return this.pos(s.t, s.row, s.k); if (s.kind === 'step') return this.pos(s.t, 'h', 0).setZ(0); return null; }
  tip(s) {
    if (s.kind === 'unit') { const v = this.rowVals(s.t, s.row)[s.k], R = ROWINFO[s.row]; return `<b>${R.html.replace('<sub>t</sub>', `<sub>${s.t + 1}</sub>`)}</b> Eintrag ${s.k + 1}${s.row === 'x' || s.row === 'y' ? ' („' + chr(this.rnn.vocab[s.k]) + '“)' : ''}<br>${R.name}: ${num(v, 3)}`; }
    if (s.kind === 'flow') return '<b>Gradientenfluss</b>: Größe von ∂L<sub>T</sub>/∂h<sub>t</sub>';
    return '';
  }
  panel(s) {
    if (!s) return this.overview();
    if (s.kind === 'unit') return this.pUnit(s);
    if (s.kind === 'step') return this.pStep(s.t);
    if (s.kind === 'row') return this.pRow(s.row);
    if (s.kind === 'flow') return this.pFlow();
    return this.overview();
  }
  eqBlock() { return RNN_EQ[this.type].map((e) => `<div class="tex">$$${e}$$</div>`).join(''); }
  pUnit(s) {
    const t = s.t, k = s.k, st = this.tr.steps[t], R = ROWINFO[s.row], tt = t + 1, H = this.rnn.H;
    const hp = (q) => (t > 0 ? this.tr.steps[t - 1].h.d[q] : 0), Cp = (q) => (t > 0 ? this.tr.steps[t - 1].C.d[q] : 0);
    let body = '';
    const pre = (V) => V.p[0].d[k];
    if (s.row === 'x') body = `<p>Zeichen „${chr(st.ch)}“. Der Vektor hat ${this.rnn.V} Einträge, einen pro Zeichen im Vokabular: ${this.rnn.vocab.map((c, q) => (q === k ? `<b>${chr(c)}</b>` : chr(c))).join(' ')}.</p><div class="tex">$$x_{${tt},${k + 1}} = ${st.xt.d[k]}$$</div>`;
    else if (s.row === 'y') {
      const tgt = this.rnn.idx[st.nx];
      body = `<div class="tex">$$\\hat{y}_{${tt},${k + 1}} = \\frac{e^{(\\mathbf{W}_{hy}\\mathbf{h}_{${tt}} + \\mathbf{b}_y)_{${k + 1}}}}{\\sum_j e^{(\\mathbf{W}_{hy}\\mathbf{h}_{${tt}} + \\mathbf{b}_y)_j}} = ${tn(st.probs[k], 3)}$$</div>
        <p>Wahrscheinlichkeit, dass nach „${chr(st.ch)}“ das Zeichen „${chr(this.rnn.vocab[k])}“ kommt. Richtig ist „${chr(st.nx)}“.</p>
        <p>Loss in diesem Schritt: \\(L_{${tt}} = -\\log \\hat{y}_{${tt},\\text{richtig}} = -\\log(${tn(st.probs[tgt], 3)}) = ${tn(st.L.d[0], 3)}\\).</p>${balkenHTML(this.rnn.vocab.map((c) => '„' + (c === ' ' ? '␣' : c) + '“'), st.probs, this.rnn.vocab.map((c, q) => (q === tgt ? T.gut : T.pos)), tgt)}`;
    } else if (this.type === 'elman' && s.row === 'h') {
      const a = Array.from({ length: H }, (_, q) => this.rnn.Whh.d[k * H + q] * hp(q)).reduce((x, y) => x + y, 0);
      const b = this.rnn.Wxh.d[k * this.rnn.V + this.rnn.idx[st.ch]], bias = this.rnn.bh.d[k];
      body = `<ol class="schritte"><li><span class="nr">1</span><div>Was aus der Vergangenheit kommt: \\((\\mathbf{W}_{hh}\\mathbf{h}_{${t}})_{${k + 1}} = ${tn(a, 3)}\\)${t === 0 ? ' (bei \\(t = 1\\) ist \\(\\mathbf{h}_0 = \\mathbf{0}\\))' : ''}</div></li>
        <li><span class="nr">2</span><div>Was die Eingabe beiträgt: \\((\\mathbf{W}_{xh}\\mathbf{x}_{${tt}})_{${k + 1}} = ${tn(b, 3)}\\). Weil \\(\\mathbf{x}_{${tt}}\\) one-hot ist, ist das genau eine Spalte von \\(\\mathbf{W}_{xh}\\).</div></li>
        <li><span class="nr">3</span><div>Summe mit Bias und tanh:<div class="tex">$$u_{${tt},${k + 1}} = ${tn(a, 3)} + ${tp(b, 3)} + ${tp(bias, 3)} = ${tn(st.u.d[k], 3)}, \\quad h_{${tt},${k + 1}} = \\tanh(${tn(st.u.d[k], 3)}) = ${tn(st.h.d[k], 3)}$$</div></div></li></ol>`;
    } else if (['f', 'i', 'o', 'r', 'z'].includes(s.row)) {
      const v = st[s.row].d[k], Wn = { f: 'f', i: 'i', o: 'o', r: 'r', z: 'z' }[s.row];
      body = `<div class="tex">$$${R.sym}_{${tt},${k + 1}} = \\sigma\\big((\\mathbf{W}_${Wn} \\cdot [\\mathbf{h}_{${t}}, \\mathbf{x}_{${tt}}])_{${k + 1}} + b_{${Wn},${k + 1}}\\big) = \\sigma(${tn(pre(st[s.row]), 3)}) = ${tn(v, 3)}$$</div>
        <p>Sigmoid liefert einen Wert zwischen 0 und 1, also eine Art Regler: ${v > 0.8 ? 'hier fast ganz offen.' : v < 0.2 ? 'hier fast ganz zu.' : 'hier teilweise offen.'}</p>`;
      if (s.row === 'f') body += `<p>Wirkung: von \\(C_{${t},${k + 1}} = ${tn(Cp(k), 3)}\\) bleibt \\(${tn(v, 3)} \\cdot ${tp(Cp(k), 3)} = ${tn(v * Cp(k), 3)}\\).</p>`;
      if (s.row === 'z') body += `<p>Wirkung: \\(h_{${tt},${k + 1}} = (1 - ${tn(v, 3)}) \\cdot ${tp(hp(k), 3)} + ${tn(v, 3)} \\cdot ${tp(st.ht_.d[k], 3)} = ${tn(st.h.d[k], 3)}\\).</p>`;
    } else if (s.row === 'Ct' || s.row === 'ht') {
      const V = s.row === 'Ct' ? st.Ct_ : st.ht_, inner = s.row === 'Ct' ? `(\\mathbf{W}_C \\cdot [\\mathbf{h}_{${t}}, \\mathbf{x}_{${tt}}])_{${k + 1}} + b_{C,${k + 1}}` : `(\\mathbf{W}_h \\cdot [\\mathbf{r}_{${tt}} \\odot \\mathbf{h}_{${t}}, \\mathbf{x}_{${tt}}])_{${k + 1}} + b_{h,${k + 1}}`;
      body = `<div class="tex">$$${R.sym}_{${tt},${k + 1}} = \\tanh\\big(${inner}\\big) = \\tanh(${tn(pre(V), 3)}) = ${tn(V.d[k], 3)}$$</div>`;
    } else if (s.row === 'C') {
      body = `<div class="tex">$$\\begin{aligned} C_{${tt},${k + 1}} &= f_{${tt},${k + 1}} \\cdot C_{${t},${k + 1}} + i_{${tt},${k + 1}} \\cdot \\tilde{C}_{${tt},${k + 1}} \\\\ &= ${tn(st.f.d[k], 3)} \\cdot ${tp(Cp(k), 3)} + ${tn(st.i.d[k], 3)} \\cdot ${tp(st.Ct_.d[k], 3)} = ${tn(st.C.d[k], 3)} \\end{aligned}$$</div>
        <p class="klein">Erster Summand: was vom alten Gedächtnis bleibt. Zweiter: was neu dazukommt.</p>`;
    } else if (s.row === 'h') {
      if (this.type === 'lstm') body = `<div class="tex">$$h_{${tt},${k + 1}} = o_{${tt},${k + 1}} \\cdot \\tanh(C_{${tt},${k + 1}}) = ${tn(st.o.d[k], 3)} \\cdot \\tanh(${tn(st.C.d[k], 3)}) = ${tn(st.h.d[k], 3)}$$</div>`;
      else body = `<div class="tex">$$h_{${tt},${k + 1}} = (1 - z_{${tt},${k + 1}}) \\cdot h_{${t},${k + 1}} + z_{${tt},${k + 1}} \\cdot \\tilde{h}_{${tt},${k + 1}} = (1 - ${tn(st.z.d[k], 3)}) \\cdot ${tp(hp(k), 3)} + ${tn(st.z.d[k], 3)} \\cdot ${tp(st.ht_.d[k], 3)} = ${tn(st.h.d[k], 3)}$$</div>`;
    }
    const g = s.row === 'h' && st.h.g ? st.h.g[k] : null;
    return `${backBtn}<h2>${R.name} \\(${R.sym}_{${tt}}\\)<span class="chip">Eintrag ${k + 1}</span></h2>
      <p class="unter">${R.text}</p>${body}
      ${g !== null ? `<h3>Fehlersignal</h3><div class="tex">$$\\frac{\\partial L}{\\partial h_{${tt},${k + 1}}} = ${tsci(g)}$$</div><p class="klein">Kommt aus zwei Richtungen: von der Vorhersage \\(\\hat{\\mathbf{y}}_{${tt}}\\) und vom nächsten Zeitschritt \\(\\mathbf{h}_{${tt + 1}}\\).</p>` : ''}
      <h3>Alle Formeln der ${CELLNAME[this.type]}-Zelle</h3>${this.eqBlock()}`;
  }
  pStep(t) {
    const st = this.tr.steps[t], tgt = this.rnn.idx[st.nx];
    return `${backBtn}<h2>Zeitschritt \\(t = ${t + 1}\\)</h2>
      <p class="unter">Eingabe „${chr(st.ch)}“, richtig als nächstes Zeichen wäre „${chr(st.nx)}“. Die Linien zeigen die Verbindungen dieses Schritts.</p>
      ${this.eqBlock()}
      <p>Dieselben Gewichte werden in jedem Zeitschritt benutzt (geteilte Gewichte). Das „ausgerollte“ Netz hat ${this.T} Kopien derselben Zelle.</p>
      <h3>Vorhersage</h3>${balkenHTML(this.rnn.vocab.map((c) => '„' + (c === ' ' ? '␣' : c) + '“'), st.probs, this.rnn.vocab.map((c, q) => (q === tgt ? T.gut : T.pos)), tgt)}
      <p>Loss: \\(L_{${t + 1}} = ${tn(st.L.d[0], 3)}\\).</p>`;
  }
  pRow(row) {
    const R = ROWINFO[row];
    return `${backBtn}<h2>${R.name} \\(${R.sym}_t\\)</h2><p class="unter">${R.text}</p>
      <p class="klein">Jeder Würfel ist ein Eintrag des Vektors. ${row === 'x' ? 'Dunkel = 1.' : row === 'y' ? 'Säulenhöhe = Wahrscheinlichkeit, grün = richtiges nächstes Zeichen.' : ['f', 'i', 'o', 'r', 'z'].includes(row) ? 'Grün = Tor offen (nahe 1), grau = zu (nahe 0).' : 'Blau positiv, orange negativ.'}</p>
      <h3>Formeln</h3>${this.eqBlock()}`;
  }
  pFlow() {
    return `${backBtn}<h2>Gradientenfluss durch die Zeit</h2>
      <p class="unter">Wie stark hängt der Loss am <b>letzten</b> Zeitschritt \\(L_{${this.T}}\\) vom Hidden State eines früheren Schritts ab?</p>
      <figure><canvas class="plot" id="pl-flow"></canvas><figcaption>Lila Säulen: \\(\\left\\| \\partial L_{${this.T}} / \\partial \\mathbf{h}_t \\right\\|\\), logarithmische Achse.${this.flowC ? ' Gold: dasselbe für den Zellzustand \\(\\mathbf{C}_t\\).' : ''}</figcaption></figure>
      <p>Beim Zurückgehen wird das Fehlersignal in jedem Schritt mit \\(\\mathbf{W}_{hh}^\\top\\) und der Ableitung von tanh multipliziert. Sind diese Faktoren kleiner als 1, wird es von Schritt zu Schritt kleiner (verschwindende Gradienten), größer als 1 explodiert es.</p>
      <p>Die LSTM hat mit dem Zellzustand einen Weg, auf dem nur mit \\(\\mathbf{f}_t\\) multipliziert und addiert wird. Ist \\(\\mathbf{f}_t \\approx 1\\), kommt das Signal weit zurück. Vergleiche die Zelltypen oben.</p>`;
  }
  overview() {
    const e = this.ev;
    return `<h2>Rekurrentes Netz</h2>
      <p class="unter">Lernt, das <b>nächste Zeichen</b> eines Textes vorherzusagen. Waagerecht läuft die Zeit (\\(t = 1 \\dots ${this.T}\\)), jede Spalte ist dieselbe Zelle mit denselben Gewichten.</p>
      <div class="segment klein">${Object.entries(CELLNAME).map(([k, n]) => `<button type="button" data-act="typ" data-k="${k}" aria-pressed="${this.type === k}">${n}</button>`).join('')}</div>
      ${this.eqBlock()}
      <p class="klein">Hier ist \\(\\sigma\\) in der letzten Formel die Softmax über alle Zeichen. \\(\\odot\\) heißt elementweise multiplizieren, \\([\\mathbf{h}_{t-1}, \\mathbf{x}_t]\\) heißt beide Vektoren hintereinanderhängen.</p>
      <div class="legende">
        <div><span class="quadrat" style="background:var(--tinte)"></span><span>Eingabe \\(\\mathbf{x}_t\\): ein dunkler Würfel = das aktuelle Zeichen.</span></div>
        <div><span class="quadrat" style="background:var(--pos)"></span><span>Zustände: blau positiv, orange negativ.</span></div>
        ${this.type !== 'elman' ? '<div><span class="quadrat" style="background:var(--gut)"></span><span>Tore (Gates): grün = offen (nahe 1), grau = zu.</span></div>' : ''}
        <div><span class="stab" style="background:var(--gut)"></span><span>Grüne Säule in \\(\\hat{\\mathbf{y}}_t\\) = richtiges nächstes Zeichen.</span></div>
      </div>
      <p class="klein">Tipp auf „t = …“ zeigt die Verbindungen dieses Schritts, Tipp auf einen Würfel seine Rechnung.</p>
      <h3>Text zum Lernen</h3>
      <div class="zeile"><input type="text" id="s-text" data-set="text" maxlength="40" value="${esc(this.text)}" aria-label="Trainingstext"></div>
      <div class="zeile"><label for="s-start">Start im Text</label><input id="s-start" type="range" min="0" max="${this.text.length - 1}" step="1" value="${this.s}" data-set="start"><output id="o-start">${this.s + 1}</output></div>
      <p class="klein">Vokabular (${this.rnn.V} Zeichen): ${this.rnn.vocab.map(chr).join(' ')}. Zu sehen sind ${this.T} Zeichen ab der gewählten Stelle; trainiert wird mit Abschnitten von 10 Zeichen, der Hidden State startet jeweils bei \\(\\mathbf{0}\\).</p>
      <h3>Selbst schreiben lassen</h3>
      <div class="knopfgruppe"><button type="button" class="knopf haupt" data-act="gen">Text erzeugen</button></div>
      <div class="ausgabe-text" id="dyn-gen">${this.gen ? esc(this.gen) : 'Noch nichts erzeugt. Das Netz beginnt mit dem ersten Zeichen und gibt sein wahrscheinlichstes nächstes Zeichen immer wieder als neue Eingabe.'}</div>
      ${e ? `<p class="klein">Nächstes Zeichen richtig vorhergesagt: ${pct(e.acc)} über alle Startstellen.</p>` : ''}
      <h3>Ansicht</h3>
      ${schalter('flow', 'Gradientenfluss durch die Zeit (BPTT)', this.flowView)}
      <p class="klein">Zeigt vorne lila Säulen: wie stark der Loss des letzten Schritts vom Hidden State jedes früheren Schritts abhängt. Antippen erklärt es.</p>`;
  }
  afterPanel(el, s) {
    const cv = $('#pl-flow', el); if (!cv) return;
    const W = 320, H = 170, g = prepCanvas(cv, W, H), Tn = this.T, L = 34, B = 22, Tp = 10;
    const Y = (v) => Tp + (1 - clamp((Math.log10(v + 1e-12) + 5) / 5, 0, 1)) * (H - Tp - B);
    g.font = '11px ' + getComputedStyle(document.body).fontFamily; g.fillStyle = T['text-3']; g.strokeStyle = T['flaeche-3'];
    for (let e = 0; e >= -5; e--) { const y = Y(Math.pow(10, e)); g.beginPath(); g.moveTo(L, y); g.lineTo(W - 6, y); g.stroke(); g.fillText(e === 0 ? '1' : `1e${e}`, 2, y + 4); }
    const bw = (W - L - 10) / Tn;
    for (let t = 0; t < Tn; t++) {
      const x = L + t * bw + 4;
      g.fillStyle = T.grad; g.fillRect(x, Y(this.flow[t]), (this.flowC ? bw / 2 : bw) - 6, H - B - Y(this.flow[t]));
      if (this.flowC) { g.fillStyle = T.c2; g.fillRect(x + bw / 2 - 3, Y(this.flowC[t]), bw / 2 - 5, H - B - Y(this.flowC[t])); }
      g.fillStyle = T['text-2']; g.fillText(`t=${t + 1}`, x, H - 6);
    }
  }
  live(el, s) {
    if (!s) { const o = $('#dyn-gen', el); if (o && this.gen) o.textContent = this.gen; return; }
    const now = performance.now(); if (this.playing && now - this.lastLive < 1500) { App.liveDirty = true; return; }
    this.lastLive = now; App.renderPanel(false);
  }
  act(name, el) {
    if (name === 'typ') { this.type = el.dataset.k; this.newNet(); this.build(); this.trace(); this.playing = true; this.autoStop = 450; App.home(); App.select(null); App.updateDock(); return; }
    if (name === 'gen') { this.gen = this.rnn.generate(42); const o = $('#dyn-gen'); if (o) o.textContent = this.gen; return; }
  }
  setOpt(key, val, el, type) {
    if (key === 'start') { this.s = +val; const o = $('#o-start'); if (o) o.textContent = this.s + 1; this.trace(); return; }
    if (key === 'text' && type === 'change') {
      const t = String(val).toLowerCase().slice(0, 40);
      if (t.length < 3 || t === this.text) return;
      this.text = t.endsWith(' ') ? t : t + ' '; this.s = 0; this.newNet(); this.build(); this.trace(); this.playing = true; this.autoStop = 450; App.renderPanel(); App.updateDock(); return;
    }
    if (key === 'flow') { this.flowView = !!val; return; }
  }
  themeChanged() { this.lastLines = 0; this.box.material.color.copy(T.c.text); this.rails[0].material.color.copy(T.c.pos); if (this.rails[1]) this.rails[1].material.color.copy(T.c.c2); }
}
App.modes.rnn = new RNNMode();
