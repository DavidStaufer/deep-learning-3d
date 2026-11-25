/* ============================================================
   Modus 1: Fully Connected Netz (L2 + L3)
   ============================================================ */
const MLP_BASE = { data: 'xor', hidden: [6, 6], acts: ['relu', 'relu'], init: 'he', opt: 'adam', lr: 0.01, batch: 16, l2: 0, dropout: 0, tempo: 2, bias: true, zahlen: window.innerWidth > 860, karten: false };
const EXPERIMENTE = [
  { titel: 'XOR ohne versteckte Schicht', cfg: { data: 'xor', hidden: [], acts: [] },
    frage: 'Trainiere eine Weile. Warum bleibt die Genauigkeit bei etwa 50 %?',
    antwort: 'Ohne versteckte Schicht rechnet jedes Ausgabeneuron nur \\(z_k = w_{0,k} + w_{1,k}x_1 + w_{2,k}x_2\\). Die Grenze zwischen zwei Klassen ist dann eine Gerade, und XOR lässt sich mit keiner Geraden trennen (L2).' },
  { titel: 'Lineare Aktivierung', cfg: { data: 'xor', hidden: [6, 6], acts: ['linear', 'linear'] },
    frage: 'Zwei versteckte Schichten mit je 6 Neuronen, trotzdem ist die Grenze eine Gerade. Warum?',
    antwort: 'Mit \\(f(z) = z\\) ist das ganze Netz \\(\\mathbf{W}_3\\mathbf{W}_2\\mathbf{W}_1\\mathbf{x}\\) (plus Bias), und ein Produkt von Matrizen ist wieder eine Matrix. Die Schichten fallen zu einer linearen Abbildung zusammen (Kollaps). Die Nichtlinearität ist der Grund, warum Tiefe überhaupt etwas bringt.' },
  { titel: 'Heaviside wie beim Perzeptron', cfg: { data: 'kreise', hidden: [6, 6], acts: ['heaviside', 'heaviside'] },
    frage: 'Drück „Rückwärts“. Welche Schichten bekommen eine lila Hülle (ein Fehlersignal)?',
    antwort: 'Nur die Ausgabe. Für Heaviside ist \\(f\'(z) = 0\\), also ist \\(\\partial L / \\partial z^{(l)}_j = f\'(z^{(l)}_j) \\cdot \\partial L / \\partial h^{(l)}_j = 0\\) in jeder versteckten Schicht. Darum lernt nur die letzte Schicht, und die versteckten Gewichte bleiben, wie sie initialisiert wurden.' },
  { titel: 'Alle Gewichte gleich', cfg: { data: 'xor', hidden: [6, 6], acts: ['tanh', 'tanh'], init: 'gleich' },
    frage: 'Klick nacheinander die Neuronen von Schicht 1 an. Was fällt dir an \\(z\\) und den Gewichten auf?',
    antwort: 'Alle Neuronen einer Schicht starten mit denselben Gewichten, rechnen deshalb dasselbe \\(z\\) und bekommen dasselbe Fehlersignal. Sie bleiben für immer gleich (Symmetrie), das Netz hat effektiv nur ein Neuron pro Schicht. Zufällige Initialisierung (He, Xavier) bricht diese Symmetrie.' },
  { titel: 'Tiefes Sigmoid-Netz', cfg: { data: 'kreise', hidden: [6, 6, 6, 6], acts: ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], opt: 'sgd', lr: 0.3, init: 'xavier' },
    frage: 'Drück „Rückwärts“ und vergleich die lila Hüllen hinten (bei der Ausgabe) mit denen vorne (Schicht 1).',
    antwort: 'Beim Rückwärtsgehen wird das Fehlersignal in jeder Schicht mit \\(\\sigma\'(z) \\le 0{,}25\\) multipliziert. Nach vier Schichten ist es winzig: verschwindende Gradienten. Die vorderen Schichten lernen kaum. Stell die Aktivierung auf ReLU um und vergleich.' },
  { titel: 'Lernrate zu groß', cfg: { data: 'spirale', hidden: [8, 8], acts: ['relu', 'relu'], opt: 'sgd', lr: 3 },
    frage: 'Schau auf die kleine Loss-Kurve in der Leiste unten. Wie sieht sie aus?',
    antwort: 'Bei \\(\\boldsymbol{\\theta}_{j+1} = \\boldsymbol{\\theta}_j - \\eta \\nabla_{\\boldsymbol{\\theta}} J\\) ist \\(\\eta\\) so groß, dass jeder Schritt über das Minimum hinausspringt. Der Loss springt hin und her, statt zu sinken. Stell \\(\\eta\\) auf 0,1 und vergleich.' },
  { titel: 'Dropout ansehen', cfg: { data: 'spirale', hidden: [10, 10], acts: ['relu', 'relu'], dropout: 0.3 },
    frage: 'Drück „Ein Schritt“. Welche Kugeln werden klein und grau?',
    antwort: 'Beim Training setzt Dropout jede Ausgabe einer versteckten Schicht mit Wahrscheinlichkeit \\(p\\) auf \\(0\\) (die kleinen grauen Kugeln). Die übrigen Werte werden durch \\(1 - p\\) geteilt, damit der Mittelwert gleich bleibt (L3). In jedem Schritt sind andere Neuronen aus; beim Testen ist Dropout aus.' },
];

class MLPMode {
  constructor() {
    this.key = 'mlp'; this.group = new THREE.Group(); this.built = false;
    this.cfg = JSON.parse(JSON.stringify(MLP_BASE));
    this.seedN = 11; this.hist = []; this.playing = false; this.anim = null; this.showGrad = false;
    this.autoStop = 0; this.dirty = true; this.lastDraw = 0; this.reveal = Infinity; this.gradReveal = Infinity;
    this.flash = null; this.lastLbl = 0; this.lastLive = 0; this.nextHist = 0;
  }
  // ---------------- Daten und Netz
  init() {
    this.data = Models.makeData(this.cfg.data);
    this.newNet(); this.sample = this.nearest(0.45, -0.55);
    this.build(); this.trace();
    this.playing = true; this.autoStop = 700;
  }
  newNet() {
    DL.seed(this.seedN);
    const c = this.cfg;
    this.net = new Models.MLP({ hidden: [...c.hidden], acts: [...c.acts], K: this.data.K, init: c.init, opt: { type: c.opt, lr: +c.lr, l2: +c.l2 }, dropout: +c.dropout, batch: +c.batch });
    this.hist = []; this.nextHist = 0; this.ev = this.net.evaluate(this.data); this.anim = null; this.showGrad = false; this.flash = null;
    App.caption(null);
  }
  nearest(a, b) {
    let best = 0, bd = Infinity;
    this.data.X.forEach(([p, q], i) => { const d = (p - a) ** 2 + (q - b) ** 2; if (d < bd) { bd = d; best = i; } });
    return { i: best, x: this.data.X[best], y: this.data.Y[best] };
  }
  trace(train = false) {
    const r = this.net.forward(this.sample.x, this.sample.y, train);
    DL.backward(r.loss);
    const cp = (v) => Float64Array.from(v.d), cg = (v) => (v.g ? Float64Array.from(v.g) : new Float64Array(v.n));
    const inp = [cp(r.x)];
    for (let l = 1; l < this.net.L; l++) { const h = cp(r.hs[l]), m = r.drops[l - 1]; if (m) for (let k = 0; k < h.length; k++) h[k] *= m[k]; inp.push(h); }
    this.tr = { x: cp(r.x), gx: cg(r.x), z: r.zs.map(cp), gz: r.zs.map(cg), h: r.hs.map(cp), gh: r.hs.map(cg), inp, drops: r.drops, probs: Float64Array.from(r.probs), loss: r.loss.d[0] };
    for (const p of this.net.params) p.g.fill(0);
  }
  // ---------------- Aufbau der Szene
  layout() {
    const sizes = this.net.sizes, L = sizes.length - 1, GAP = 3.3;
    this.X = sizes.map((_, l) => l * GAP);
    this.P = sizes.map((n, l) => {
      const cols = n <= 5 ? 1 : n <= 10 ? 2 : n <= 12 ? 3 : 4, rows = Math.ceil(n / cols), s = 0.95, out = [];
      for (let j = 0; j < n; j++) { const col = Math.floor(j / rows), row = j % rows; out.push(V3(this.X[l] + (col - (cols - 1) / 2) * 0.32, ((rows - 1) / 2 - row) * s, (col - (cols - 1) / 2) * s * 1.15)); }
      return out;
    });
    this.minY = (l) => Math.min(...this.P[l].map((p) => p.y));
    this.maxY = (l) => Math.max(...this.P[l].map((p) => p.y));
    this.BP = []; for (let l = 0; l < L; l++) this.BP.push(V3(this.X[l], this.minY(l) - 0.95, 0));
    this.S = 3.0; this.DP = V3(this.X[0] - 3.3, 0, 0);
    this.LP = V3(this.X[L] + 3.2, 0, 0);
    const lo = Math.min(-this.S / 2, ...this.BP.map((p) => p.y), ...this.P.map((_, l) => this.minY(l))), hi = Math.max(this.S / 2, ...this.P.map((_, l) => this.maxY(l)));
    this.bounds = { min: V3(this.DP.x - this.S / 2 - 0.3, lo - 0.4, -1.2), max: V3(this.LP.x + 0.8, hi + 0.9, 1.2) };
  }
  view() {
    const v = viewFor(this.bounds.min, this.bounds.max, App.phone() ? V3(0.95, 0.42, 0.72) : V3(0.42, 0.3, 1));
    return v;
  }
  build() {
    disposeGroup(this.group); this.layout();
    const g = this.group, sizes = this.net.sizes, L = sizes.length - 1, total = sizes.reduce((a, b) => a + b, 0);
    // Neuronen (Kugeln) und Rahmen für die Karten-Ansicht
    this.nMesh = instanced(new THREE.SphereGeometry(0.24, 28, 18), new THREE.MeshStandardMaterial({ roughness: 0.42, metalness: 0.04 }), total);
    this.fMesh = instanced(new THREE.BoxGeometry(0.72, 0.72, 0.08), new THREE.MeshStandardMaterial({ roughness: 0.5 }), total);
    this.nId = []; this.nInfo = []; let k = 0;
    sizes.forEach((n, l) => { this.nId[l] = []; for (let j = 0; j < n; j++) { this.nId[l][j] = k; this.nInfo[k] = { l, j }; k++; } });
    const pickN = (h) => { const inf = this.nInfo[h.instanceId]; return inf ? { kind: 'neuron', l: inf.l, j: inf.j } : null; };
    this.nMesh.userData.pick = pickN; this.fMesh.userData.pick = pickN;
    this.haloMesh = instanced(new THREE.SphereGeometry(1, 22, 14), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3, depthWrite: false }), total);
    g.add(this.nMesh, this.fMesh, this.haloMesh);
    // Karten (Wert über dem Eingaberaum) für jedes Neuron
    this.maps = [];
    sizes.forEach((n, l) => { for (let j = 0; j < n; j++) {
      const cv = document.createElement('canvas'); cv.width = cv.height = 28;
      const tex = new THREE.CanvasTexture(cv); tex.magFilter = THREE.LinearFilter;
      const m = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.62), new THREE.MeshBasicMaterial({ map: tex }));
      m.position.copy(this.P[l][j]).add(V3(0, 0, 0.045)); m.userData.pick = () => ({ kind: 'neuron', l, j });
      g.add(m); this.maps.push({ l, j, cv, tex, mesh: m });
    } });
    // Bias-Neuronen „1“
    this.bMesh = instanced(new THREE.BoxGeometry(0.34, 0.34, 0.34), new THREE.MeshStandardMaterial({ roughness: 0.5 }), L);
    this.bMesh.userData.pick = (h) => ({ kind: 'bias', l: h.instanceId });
    g.add(this.bMesh);
    // Gewichte als Stäbe: Dicke = |w|, Farbe = Vorzeichen
    this.edges = [];
    for (let l = 0; l < L; l++) for (let j = 0; j < sizes[l + 1]; j++) for (let i = -1; i < sizes[l]; i++) {
      const a = i < 0 ? this.BP[l] : this.P[l][i], b = this.P[l + 1][j];
      this.edges.push(Object.assign({ l, i, j, a, b }, stick(a, b)));
    }
    this.eMesh = instanced(new THREE.CylinderGeometry(1, 1, 1, 8, 1, true), new THREE.MeshStandardMaterial({ roughness: 0.6 }), this.edges.length);
    this.eMesh.userData.pick = (h) => { const e = this.edges[h.instanceId]; return e ? { kind: 'edge', l: e.l, i: e.i, j: e.j } : null; };
    g.add(this.eMesh);
    // Ausgabe-Balken ŷ und Loss
    this.barMesh = instanced(new THREE.BoxGeometry(1, 0.26, 0.26), new THREE.MeshStandardMaterial({ roughness: 0.4 }), sizes[L]);
    this.barMesh.userData.pick = () => ({ kind: 'layer', l: L });
    g.add(this.barMesh);
    this.lossMesh = new THREE.Mesh(new THREE.IcosahedronGeometry(0.34, 3), new THREE.MeshStandardMaterial({ roughness: 0.35 }));
    this.lossMesh.position.copy(this.LP); this.lossMesh.userData.pick = () => ({ kind: 'loss' });
    g.add(this.lossMesh);
    const lp = []; for (let q = 0; q < sizes[L]; q++) { const p = this.P[L][q]; lp.push(this.X[L] + 2.15, p.y, p.z, this.LP.x - 0.3, this.LP.y, this.LP.z); }
    const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3));
    this.lossLines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: T.c['text-3'], transparent: true, opacity: 0.5 }));
    g.add(this.lossLines);
    // Eingaberaum als Tafel
    this.planeCv = document.createElement('canvas'); this.planeCv.width = this.planeCv.height = 256;
    this.planeTex = new THREE.CanvasTexture(this.planeCv);
    this.plane = new THREE.Mesh(new THREE.PlaneGeometry(this.S, this.S), new THREE.MeshBasicMaterial({ map: this.planeTex }));
    this.plane.position.copy(this.DP);
    this.plane.userData.pick = (h) => ({ kind: 'data', u: h.uv.x, v: h.uv.y });
    this.board = new THREE.Mesh(new THREE.BoxGeometry(this.S + 0.16, this.S + 0.16, 0.08), new THREE.MeshStandardMaterial({ color: T.c.flaeche, roughness: 0.7 }));
    this.board.position.copy(this.DP).add(V3(0, 0, -0.05));
    this.marker = new THREE.Mesh(new THREE.TorusGeometry(0.1, 0.022, 8, 32), new THREE.MeshBasicMaterial({ color: T.c.text }));
    const dl = new THREE.BufferGeometry(); dl.setAttribute('position', new THREE.Float32BufferAttribute(new Array(12).fill(0), 3));
    this.dashes = new THREE.LineSegments(dl, new THREE.LineDashedMaterial({ color: T.c['text-2'], dashSize: 0.12, gapSize: 0.08 }));
    g.add(this.plane, this.board, this.marker, this.dashes);
    // Auswahlring
    this.ring = new THREE.Mesh(new THREE.TorusGeometry(0.36, 0.022, 8, 48), new THREE.MeshBasicMaterial({ color: T.c.akzent }));
    this.ring.visible = false; g.add(this.ring);
    g.add(shadowDisc(this.bounds.max.x - this.bounds.min.x + 4, 7, this.bounds.min.y - 0.25));
    // Beschriftungen
    this.layerLbl = [];
    for (let l = 0; l <= L; l++) {
      const t = l === 0 ? 'Eingabe <small>x</small>' : l < L ? `Schicht ${l} <small>${ACTINFO[this.net.cfg.acts[l - 1]].name}</small>` : 'Ausgabe <small>Softmax</small>';
      const o = label(t, 'lbl', () => App.select({ kind: 'layer', l }));
      o.position.set(this.X[l], this.maxY(l) + 0.72, 0); g.add(o); this.layerLbl.push(o);
    }
    const pl = label(`Eingaberaum <small>${Models.DATASETS[this.cfg.data].name}</small>`, 'lbl', () => App.select({ kind: 'data' }));
    pl.position.copy(this.DP).add(V3(0, this.S / 2 + 0.42, 0)); g.add(pl);
    const ax1 = label('x<sub>1</sub> →', 'wert'); ax1.position.copy(this.DP).add(V3(this.S / 2 - 0.3, -this.S / 2 - 0.22, 0)); g.add(ax1);
    const ax2 = label('x<sub>2</sub> ↑', 'wert'); ax2.position.copy(this.DP).add(V3(-this.S / 2 - 0.32, this.S / 2 - 0.25, 0)); g.add(ax2);
    this.lossLbl = label('Loss', 'lbl', () => App.select({ kind: 'loss' })); this.lossLbl.position.copy(this.LP).add(V3(0, 0.78, 0)); g.add(this.lossLbl);
    this.valLbl = [];
    sizes.forEach((n, l) => { for (let j = 0; j < n; j++) {
      const o = label('', 'wert'); o.position.copy(this.P[l][j]).add(l === 0 ? V3(-0.62, 0, 0) : V3(0, -0.4, 0.05)); g.add(o); this.valLbl.push({ o, l, j });
    } });
    this.outLbl = [];
    for (let q = 0; q < sizes[L]; q++) { const o = label('', 'wert'); o.position.set(this.X[L] + 2.6, this.P[L][q].y, this.P[L][q].z); g.add(o); this.outLbl.push(o); }
    this.biasLbl = [];
    for (let l = 0; l < L; l++) { const o = label('1', 'wert'); o.position.copy(this.BP[l]).add(V3(0, -0.36, 0)); g.add(o); this.biasLbl.push(o); }
    this.pickables = [this.nMesh, this.fMesh, ...this.maps.map((m) => m.mesh), this.bMesh, this.eMesh, this.barMesh, this.lossMesh, this.plane];
    this.applyView(); this.dirty = true; this.placeMarker();
  }
  applyView() {
    const k = this.cfg.karten;
    this.nMesh.visible = !k; this.fMesh.visible = k; for (const m of this.maps) m.mesh.visible = k;
    this.bMesh.visible = this.cfg.bias; for (const o of this.biasLbl) o.visible = this.cfg.bias;
    for (const v of this.valLbl) v.o.visible = this.cfg.zahlen && !k;
    for (const o of this.outLbl) o.visible = true;
    if (k) this.computeMaps();
  }
  placeMarker() {
    const [a, b] = this.sample.x, p = this.DP.clone().add(V3(a * this.S / 2, b * this.S / 2, 0.03));
    this.marker.position.copy(p);
    const pos = this.dashes.geometry.attributes.position, q1 = this.P[0][0], q2 = this.P[0][1];
    pos.setXYZ(0, p.x, p.y, p.z); pos.setXYZ(1, q1.x - 0.26, q1.y, q1.z); pos.setXYZ(2, p.x, p.y, p.z); pos.setXYZ(3, q2.x - 0.26, q2.y, q2.z);
    pos.needsUpdate = true; this.dashes.computeLineDistances();
  }
  setSample(s) { this.sample = s; this.placeMarker(); this.trace(); this.dirty = true; this.showGrad = false; }
  // ---------------- Zeichnen
  drawPlane() {
    const cv = this.planeCv, g = cv.getContext('2d'), W = cv.width, G = 48, cell = W / G, c = new THREE.Color();
    const net = this.net;
    for (let r = 0; r < G; r++) for (let q = 0; q < G; q++) {
      const x1 = -1 + (q + 0.5) / G * 2, x2 = 1 - (r + 0.5) / G * 2, p = net.plain([x1, x2]).probs;
      c.setRGB(0, 0, 0); for (let k = 0; k < p.length; k++) { c.r += T.cls[k].r * p[k]; c.g += T.cls[k].g * p[k]; c.b += T.cls[k].b * p[k]; }
      c.lerp(T.c.flaeche, T.dark ? 0.55 : 0.62);
      g.fillStyle = cssRGB(c); g.fillRect(q * cell, r * cell, cell + 0.5, cell + 0.5);
    }
    g.strokeStyle = T['text-3']; g.globalAlpha = 0.35; g.lineWidth = 1;
    g.beginPath(); g.moveTo(W / 2, 0); g.lineTo(W / 2, W); g.moveTo(0, W / 2); g.lineTo(W, W / 2); g.stroke(); g.globalAlpha = 1;
    for (let i = 0; i < this.data.X.length; i++) {
      const [a, b] = this.data.X[i];
      g.fillStyle = T.cls[this.data.Y[i]].getStyle(); g.strokeStyle = T.flaeche; g.lineWidth = 1.2;
      g.beginPath(); g.arc((a + 1) / 2 * W, (1 - (b + 1) / 2) * W, 3.6, 0, 7); g.fill(); g.stroke();
    }
    this.planeTex.needsUpdate = true;
  }
  gridValues(G) { // Werte jedes Neurons über dem Eingaberaum
    const sizes = this.net.sizes, L = sizes.length - 1, out = sizes.map((n) => Array.from({ length: n }, () => new Float32Array(G * G)));
    for (let r = 0; r < G; r++) for (let q = 0; q < G; q++) {
      const x = [-1 + (q + 0.5) / G * 2, -1 + (r + 0.5) / G * 2], res = this.net.plain(x), idx = r * G + q;
      for (let l = 0; l <= L; l++) { const v = l === 0 ? x : l < L ? res.hs[l] : res.probs; for (let j = 0; j < sizes[l]; j++) out[l][j][idx] = v[j]; }
    }
    return out;
  }
  computeMaps() {
    const G = 28, vals = this.gridValues(G), L = this.net.L, c = new THREE.Color();
    for (const m of this.maps) {
      const g = m.cv.getContext('2d'), img = g.createImageData(G, G), v = vals[m.l][m.j];
      let mx = 1e-9; for (const a of v) mx = Math.max(mx, Math.abs(a));
      for (let r = 0; r < G; r++) for (let q = 0; q < G; q++) {
        const a = v[r * G + q];
        if (m.l === L) c.copy(T.c.flaeche).lerp(T.cls[m.j], a * 0.92); else valueColor(a / mx, c);
        const o = ((G - 1 - r) * G + q) * 4; img.data[o] = c.r * 255; img.data[o + 1] = c.g * 255; img.data[o + 2] = c.b * 255; img.data[o + 3] = 255;
      }
      g.putImageData(img, 0, 0); m.tex.needsUpdate = true;
    }
  }
  themeChanged() {
    this.dirty = true; this.board.material.color.copy(T.c.flaeche); this.marker.material.color.copy(T.c.text);
    this.ring.material.color.copy(T.c.akzent); this.lossLines.material.color.copy(T.c['text-3']); this.dashes.material.color.copy(T.c['text-2']);
    if (this.cfg.karten) this.computeMaps();
  }
  // ---------------- Pro Bild
  frame(dt, now) {
    if (this.anim) this.runAnim(now);
    else if (this.playing) {
      const t0 = performance.now(), want = +this.cfg.tempo; let n = 0;
      while (n < want && performance.now() - t0 < 12) { this.net.trainStep(this.data); n++; }
      this.afterTrain(n);
      if (this.autoStop && this.net.steps >= this.autoStop) { this.playing = false; this.autoStop = 0; App.updateDock(); App.liveDirty = true; }
    }
    if (this.dirty && now - this.lastDraw > 0.15) { this.lastDraw = now; this.dirty = false; this.drawPlane(); if (this.cfg.karten) this.computeMaps(); }
    this.updateVisuals(now);
  }
  afterTrain(n) {
    if (!n) return;
    this.showGrad = false; this.trace(); this.dirty = true;
    if (this.net.steps >= this.nextHist) { this.ev = this.net.evaluate(this.data); this.hist.push(this.ev.loss); if (this.hist.length > 150) this.hist.shift(); this.nextHist = this.net.steps + 8; }
    App.liveDirty = true;
  }
  weight(E) { return E.i < 0 ? this.net.b[E.l].d[E.j] : this.net.W[E.l].d[E.j * this.net.W[E.l].shape[1] + E.i]; }
  delta(E) { return E.i < 0 ? this.net.b[E.l].delta[E.j] : this.net.W[E.l].delta[E.j * this.net.W[E.l].shape[1] + E.i]; }
  updateVisuals(now) {
    const tr = this.tr, sizes = this.net.sizes, L = sizes.length - 1, sel = App.sel, c = new THREE.Color();
    const fN = sel && (sel.kind === 'neuron') ? sel : null, fE = sel && sel.kind === 'edge' ? sel : null, fL = sel && sel.kind === 'layer' ? sel.l : null;
    let gref = 0.05; for (const v of tr.gz[L - 1]) gref = Math.max(gref, Math.abs(v));
    for (let l = 0; l <= L; l++) {
      const vals = l === 0 ? tr.x : l < L ? tr.h[l] : tr.probs;
      let mx = 1; if (l > 0 && l < L) for (const v of vals) mx = Math.max(mx, Math.abs(v));
      for (let j = 0; j < sizes[l]; j++) {
        const id = this.nId[l][j], shown = l <= this.reveal;
        if (!shown) c.copy(T.c.null);
        else if (l === L) c.copy(T.c.null).lerp(T.cls[j], 0.2 + 0.8 * vals[j]);
        else valueColor(vals[j] / mx, c);
        const dropped = l > 0 && l < L && tr.drops[l - 1] && tr.drops[l - 1][j] === 0;
        if (dropped) c.copy(T.c.null);
        const dim = (fL !== null && fL !== l) ? 0.55 : 0;
        if (dim) c.lerp(T.c.bg, dim);
        this.nMesh.setColorAt(id, c); this.fMesh.setColorAt(id, c);
        const sc = dropped ? 0.5 : 1;
        setInst(this.nMesh, id, this.P[l][j], sc); setInst(this.fMesh, id, this.P[l][j], sc);
        const gz = l === 0 ? 0 : tr.gz[l - 1][j], showH = this.showGrad && l > 0 && l >= this.gradReveal;
        const gn = Math.min(1, Math.sqrt(Math.abs(gz) / gref));
        setInst(this.haloMesh, id, this.P[l][j], showH ? 0.25 + 0.34 * gn : 0);
        this.haloMesh.setColorAt(id, T.c.grad);
      }
    }
    for (const M of [this.nMesh, this.fMesh, this.haloMesh]) { M.instanceMatrix.needsUpdate = true; M.instanceColor.needsUpdate = true; }
    for (let l = 0; l < L; l++) { setInst(this.bMesh, l, this.BP[l], 1); this.bMesh.setColorAt(l, l <= this.reveal ? T.c.pos : T.c.null); }
    this.bMesh.instanceMatrix.needsUpdate = true; this.bMesh.instanceColor.needsUpdate = true;
    // Gewichte
    let fl = 0, maxD = 1e-12;
    if (this.flash) { fl = Math.max(0, 1 - (now - this.flash.t0) / 1.4); if (fl === 0) this.flash = null; else maxD = this.flash.maxD; }
    for (let e = 0; e < this.edges.length; e++) {
      const E = this.edges[e], w = this.weight(E), aw = Math.abs(w);
      let r = (E.i < 0 && !this.cfg.bias) ? 0 : 0.006 + 0.04 * Math.min(1, aw / 1.6);
      valueColor(Math.sign(w) * (0.3 + 0.7 * Math.min(1, aw / 1.2)), c);
      let conn = true;
      if (fN) conn = (E.l === fN.l - 1 && E.j === fN.j) || (E.l === fN.l && E.i === fN.j);
      else if (fE) conn = E.l === fE.l && E.i === fE.i && E.j === fE.j;
      else if (fL !== null) conn = E.l === fL - 1;
      if (!conn) { c.lerp(T.c.bg, 0.85); r *= 0.6; } else if (fN || fE) r *= 1.35;
      if (E.l >= this.reveal) c.lerp(T.c.bg, 0.6);
      if (fl > 0) c.lerp(T.c.grad, fl * Math.min(1, Math.abs(this.delta(E)) / maxD));
      setInst(this.eMesh, e, E.mid, r, E.len, r, E.quat); this.eMesh.setColorAt(e, c);
    }
    this.eMesh.instanceMatrix.needsUpdate = true; this.eMesh.instanceColor.needsUpdate = true;
    // Ausgabe-Balken und Loss
    for (let k = 0; k < sizes[L]; k++) {
      const p = L <= this.reveal ? tr.probs[k] : 0, len = Math.max(0.02, p * 1.7), P = this.P[L][k];
      _p.set(this.X[L] + 0.36 + len / 2, P.y, P.z); setInst(this.barMesh, k, _p, len, 1, 1); this.barMesh.setColorAt(k, T.cls[k]);
    }
    this.barMesh.instanceMatrix.needsUpdate = true; this.barMesh.instanceColor.needsUpdate = true;
    const lossOn = this.reveal > L;
    this.lossMesh.scale.setScalar(lossOn ? 0.7 + 0.5 * Math.min(1, tr.loss / 1.5) : 0.6);
    this.lossMesh.material.color.copy(T.c.null).lerp(T.c.neg, lossOn ? 0.25 + 0.75 * Math.min(1, tr.loss / 1.2) : 0);
    // Auswahlring
    if (fN) { this.ring.visible = true; this.ring.position.copy(this.P[fN.l][fN.j]); this.ring.quaternion.copy(camera.quaternion); } else this.ring.visible = false;
    if (now - this.lastLbl > 0.12) { this.lastLbl = now; this.updateLabels(); }
  }
  updateLabels() {
    const tr = this.tr, L = this.net.L;
    for (const v of this.valLbl) {
      if (!v.o.visible) continue;
      let t;
      if (v.l > this.reveal) t = '·';
      else if (v.l === 0) t = `x<sub>${v.j + 1}</sub> = ${num(tr.x[v.j])}`;
      else if (v.l < L) t = tr.drops[v.l - 1] && tr.drops[v.l - 1][v.j] === 0 ? 'aus' : num(tr.h[v.l][v.j]);
      else t = num(tr.z[L - 1][v.j]);
      if (v.o.element.innerHTML !== t) v.o.element.innerHTML = t;
    }
    this.outLbl.forEach((o, k) => {
      const t = (L <= this.reveal ? `ŷ<sub>${k + 1}</sub> = ${num(tr.probs[k])}` : `ŷ<sub>${k + 1}</sub>`) + (this.sample.y === k ? ' · y<sub>' + (k + 1) + '</sub> = 1' : '');
      if (o.element.innerHTML !== t) o.element.innerHTML = t;
    });
    const lt = this.reveal > L ? `Loss <small>L = ${num(tr.loss, 3)}</small>` : 'Loss';
    if (this.lossLbl.element.innerHTML !== lt) this.lossLbl.element.innerHTML = lt;
    const sel = App.sel;
    this.layerLbl.forEach((o, l) => o.element.classList.toggle('aktiv', !!sel && sel.kind === 'layer' && sel.l === l));
  }
  // ---------------- Animation: Vorwärts, Loss, Rückwärts, Update
  startAnim(kind) {
    if (this.anim) return;
    this.playing = false;
    const L = this.net.L, D = 0.8, ph = [];
    if (kind !== 'backward') { for (let l = 0; l < L; l++) ph.push({ type: 'fwd', l, dur: D }); ph.push({ type: 'loss', dur: D }); }
    if (kind !== 'forward') { ph.push({ type: 'bwdOut', dur: D }); for (let l = L - 1; l >= 1; l--) ph.push({ type: 'bwd', l, dur: D }); }
    if (kind === 'step') ph.push({ type: 'update', dur: 1.5 });
    let t = 0; for (const p of ph) { p.t = t; t += p.dur; }
    this.trace(kind === 'step' && +this.cfg.dropout > 0);
    this.anim = { kind, ph, idx: -1, t0: performance.now() / 1000, end: t + 0.2 };
    this.reveal = kind === 'backward' ? Infinity : -1;
    this.gradReveal = Infinity; this.showGrad = kind !== 'forward';
    App.updateDock();
  }
  runAnim(now) {
    const a = this.anim, t = now - a.t0;
    while (a.idx + 1 < a.ph.length && t >= a.ph[a.idx + 1].t) { a.idx++; this.enterPhase(a.ph[a.idx]); }
    if (t >= a.end) {
      this.anim = null; this.reveal = Infinity; if (this.showGrad) this.gradReveal = 1;
      App.caption(a.kind === 'backward' || a.kind === 'step' ? null : null);
      if (a.kind === 'step') this.showGrad = false;
      App.updateDock(); App.liveDirty = true;
    }
  }
  enterPhase(p) {
    const tr = this.tr, L = this.net.L, D = p.dur, now = performance.now() / 1000, c = new THREE.Color();
    const actName = (l) => ACTINFO[this.net.cfg.acts[l - 1]].name;
    if (p.type === 'fwd') {
      this.reveal = p.l;
      for (const E of this.edges) if (E.l === p.l) {
        if (E.i < 0 && !this.cfg.bias) continue;
        const inp = E.i < 0 ? 1 : tr.inp[p.l][E.i], w = this.weight(E), prod = w * inp;
        if (Math.abs(prod) < 0.02) continue;
        Pulses.add(E.a, E.b, Math.random() * 0.08, D * 0.9, valueColor(Math.sign(prod) * 0.95, c), 0.045 + 0.05 * Math.min(1, Math.abs(prod)));
      }
      const tgt = p.l + 1;
      const inS = p.l === 0 ? 'x<sub>i</sub>' : `h<sup>(${p.l})</sup><sub>i</sub>`;
      App.caption(tgt < L
        ? `<b>Vorwärts · Schicht ${tgt}</b>: jedes Neuron j bildet z<sup>(${tgt})</sup><sub>j</sub> = w<sub>0,j</sub> + Σ<sub>i</sub> w<sub>i,j</sub> ${inS} und dann h<sup>(${tgt})</sup><sub>j</sub> = ${actName(tgt)}(z<sup>(${tgt})</sup><sub>j</sub>).`
        : `<b>Vorwärts · Ausgabe</b>: jedes Ausgabeneuron k bildet die Summe z<sub>k</sub> (Logit), Softmax macht daraus Wahrscheinlichkeiten ŷ<sub>k</sub>.`);
    } else if (p.type === 'loss') {
      this.reveal = L;
      for (let k = 0; k < this.net.sizes[L]; k++) Pulses.add(V3(this.X[L] + 2.15, this.P[L][k].y, this.P[L][k].z), this.LP, 0, D * 0.9, T.cls[k], 0.06 + 0.06 * tr.probs[k]);
      App.caption(`<b>Loss</b>: richtig ist Klasse ${this.sample.y + 1}, das Netz gibt ihr ŷ<sub>${this.sample.y + 1}</sub> = ${num(tr.probs[this.sample.y])}. Cross-Entropy: L = −log ŷ<sub>${this.sample.y + 1}</sub> = ${num(tr.loss, 3)}.`);
    } else if (p.type === 'bwdOut') {
      this.reveal = Infinity; this.gradReveal = L;
      for (let k = 0; k < this.net.sizes[L]; k++) { const gk = tr.gz[L - 1][k]; Pulses.add(this.LP, this.P[L][k], 0, D * 0.9, T.c.grad, 0.04 + 0.08 * Math.min(1, Math.abs(gk))); }
      App.caption(`<b>Rückwärts · Ausgabe</b>: das Fehlersignal am Logit ist ∂L/∂z<sub>k</sub> = ŷ<sub>k</sub> − y<sub>k</sub>. Lila Hülle = wie groß ∂L/∂z ist.`);
    } else if (p.type === 'bwd') {
      this.gradReveal = p.l + 1;
      let mx = 1e-9; for (const E of this.edges) if (E.l === p.l && E.i >= 0) mx = Math.max(mx, Math.abs(this.weight(E) * tr.gz[p.l][E.j]));
      for (const E of this.edges) if (E.l === p.l && E.i >= 0) {
        const s = Math.abs(this.weight(E) * tr.gz[p.l][E.j]) / mx;
        if (s < 0.04) continue;
        Pulses.add(E.b, E.a, Math.random() * 0.08, D * 0.9, T.c.grad, 0.035 + 0.06 * s);
      }
      setTimeout(() => { if (this.anim) this.gradReveal = p.l; }, D * 900);
      App.caption(`<b>Rückwärts · Schicht ${p.l}</b> (Kettenregel): ∂L/∂z<sup>(${p.l})</sup><sub>j</sub> = f′(z<sup>(${p.l})</sup><sub>j</sub>) · Σ<sub>k</sub> w<sub>j,k</sub> · ∂L/∂z<sub>k</sub>. Ist f′ klein, wird das Signal klein.`);
    } else if (p.type === 'update') {
      this.showGrad = false;
      this.net.trainStep(this.data);
      let maxD = 1e-12; for (const P of this.net.params) for (const d of P.delta) maxD = Math.max(maxD, Math.abs(d));
      this.flash = { t0: now, maxD };
      this.trace(); this.afterTrain(1); this.showGrad = false;
      const oname = { sgd: 'SGD', momentum: 'SGD mit Momentum', adam: 'Adam' }[this.cfg.opt];
      App.caption(`<b>Update</b> (${oname}): jedes Gewicht macht einen Schritt gegen seinen Gradienten, θ ← θ − η · ∇<sub>θ</sub>J mit η = ${num(+this.cfg.lr, 3)}. ∇J ist der Mittelwert über ein Mini-Batch von B = ${this.cfg.batch} Punkten. Lila aufleuchtend = große Änderung.`);
    }
  }
  toggleTrain() { if (this.anim) return; this.playing = !this.playing; this.autoStop = 0; App.caption(null); }
  stepOnce() { this.startAnim('step'); }
  animForward() { this.startAnim('forward'); }
  animBackward() { this.startAnim('backward'); }
  reinit() { this.seedN++; this.newNet(); this.trace(); this.dirty = true; App.renderPanel(); }
  stats() {
    const e = this.ev || { loss: NaN, acc: NaN };
    return `<span class="lang">Schritt <b>${this.net.steps}</b> · Epoche ${this.net.epoch} · </span>Loss <b>${num(e.loss, 3)}</b> · <span class="lang">Genauigkeit </span><b>${pct(e.acc)}</b>`;
  }
  // ---------------- Auswahl und Inspektor
  onSelect(s) {
    if (s && s.kind === 'data' && s.u !== undefined) { this.setSample(this.nearest(s.u * 2 - 1, s.v * 2 - 1)); delete s.u; delete s.v; }
  }
  where(s) {
    if (s.kind === 'neuron') return this.P[s.l][s.j];
    if (s.kind === 'bias') return this.BP[s.l];
    if (s.kind === 'edge') { const E = this.edges.find((e) => e.l === s.l && e.i === s.i && e.j === s.j); return E && E.mid; }
    if (s.kind === 'layer') return V3(this.X[s.l], 0, 0);
    if (s.kind === 'loss') return this.LP;
    if (s.kind === 'data') return this.DP;
    return null;
  }
  tip(s) {
    const tr = this.tr, L = this.net.L;
    if (s.kind === 'neuron') {
      if (s.l === 0) return `<b>Eingabe x<sub>${s.j + 1}</sub></b> = ${num(tr.x[s.j])}`;
      if (s.l === L) return `<b>Ausgabeneuron ${s.j + 1}</b><br>z<sub>${s.j + 1}</sub> = ${num(tr.z[L - 1][s.j])} · ŷ<sub>${s.j + 1}</sub> = ${num(tr.probs[s.j])}`;
      return `<b>Neuron ${s.j + 1} · Schicht ${s.l}</b><br>z = ${num(tr.z[s.l - 1][s.j])} · h = ${num(tr.h[s.l][s.j])}`;
    }
    if (s.kind === 'edge') { const w = this.weight(s); return `<b>Gewicht</b> w = ${num(w, 3)}` + (s.i < 0 ? ' (Bias)' : ''); }
    if (s.kind === 'bias') return '<b>Bias-Neuron</b>: immer 1';
    if (s.kind === 'data') return '<b>Eingaberaum</b>: klicken wählt den nächsten Datenpunkt';
    if (s.kind === 'loss') return `<b>Loss</b> L = ${num(tr.loss, 3)}`;
    if (s.kind === 'layer') return '<b>Ausgabe</b>: Softmax-Wahrscheinlichkeiten';
    return '';
  }
  panel(s) {
    if (!s) return this.overview();
    const L = this.net.L;
    if (s.kind === 'neuron') return s.l === 0 ? this.pInput(s.j) : s.l === L ? this.pOutputNeuron(s.j) : this.pNeuron(s.l, s.j);
    if (s.kind === 'edge') return this.pEdge(s);
    if (s.kind === 'bias') return this.pBias(s.l);
    if (s.kind === 'layer') return s.l === 0 ? this.pInput(null) : s.l === L ? this.pSoftmax() : this.pLayer(s.l);
    if (s.kind === 'loss') return this.pLoss();
    if (s.kind === 'data') return this.pData();
    return this.overview();
  }
  inSym(l, i) { return l === 0 ? `x_{${i}}` : `h^{(${l})}_{${i}}`; }
  sumTex(l, j) { // l = Schicht des Neurons (1..L)
    const tr = this.tr, W = this.net.W[l - 1], n = W.shape[1], b = this.net.b[l - 1].d[j], inp = tr.inp[l - 1], L = this.net.L;
    const zs = l === L ? `z_{${j + 1}}` : `z^{(${l})}_{${j + 1}}`, ws = l === L ? '' : `^{(${l})}`;
    const terms = [tn(b)];
    for (let i = 0; i < n; i++) terms.push(`${tp(W.d[j * n + i])} \\cdot ${tp(inp[i])}`);
    const lines = []; for (let q = 0; q < terms.length; q += 3) lines.push(terms.slice(q, q + 3).join(' + '));
    const gen = `${zs} = w${ws}_{0,${j + 1}} + \\sum_{i=1}^{${n}} w${ws}_{i,${j + 1}}\\, ${l === 1 ? 'x_i' : `h^{(${l - 1})}_i`}`;
    const numTex = `\\begin{aligned} ${zs} &= ${lines.join(' \\\\ &\\quad + ')} \\\\ &= ${tn(tr.z[l - 1][j], 3)} \\end{aligned}`;
    return { gen, numTex };
  }
  pNeuron(l, j) {
    const tr = this.tr, key = this.net.cfg.acts[l - 1], A = DL.ACT[key], z = tr.z[l - 1][j], h = tr.h[l][j];
    const { gen, numTex } = this.sumTex(l, j), fd = A.d(z), gh = tr.gh[l][j], gz = tr.gz[l - 1][j];
    const n = this.net.W[l - 1].shape[1], inName = l === 1 ? 'die Eingaben \\(x_i\\)' : `die Werte \\(h^{(${l - 1})}_i\\) der Schicht ${l - 1}`;
    const nx = this.net.sizes[l + 1], last = l + 1 === this.net.L;
    const dropped = tr.drops[l - 1] && tr.drops[l - 1][j] === 0;
    return `${backBtn}<h2>Neuron ${j + 1} in Schicht ${l}<span class="chip">versteckt</span></h2>
      <p class="unter">Gewählter Punkt \\(\\mathbf{x} = (${tn(tr.x[0])},\\ ${tn(tr.x[1])})\\), Klasse ${this.sample.y + 1}.</p>
      ${dropped ? '<div class="karte hinweis">In diesem Trainingsschritt ist das Neuron durch Dropout aus: seine Ausgabe wird auf 0 gesetzt.</div>' : ''}
      <ol class="schritte">
        <li><span class="nr">1</span><div><b>Gewichtete Summe</b> (Fully Connected). Das Neuron nimmt ${inName}, multipliziert jede mit ihrem Gewicht \\(w_{i,${j + 1}}\\) und addiert den Bias \\(w_{0,${j + 1}}\\) (Bias-Trick: Gewicht des Neurons „1“).
          <div class="tex">$$${gen}$$</div><p class="klein">Eingesetzt (Gewicht · Eingabe, die erste Zahl ist der Bias):</p><div class="tex">$$${numTex}$$</div></div></li>
        <li><span class="nr">2</span><div><b>Aktivierung</b> mit ${ACTINFO[key].name}:
          <div class="tex">$$h^{(${l})}_{${j + 1}} = f\\big(z^{(${l})}_{${j + 1}}\\big) = f(${tn(z)}) = ${tn(h, 3)}$$</div>
          <figure><canvas class="plot" id="pl-act"></canvas><figcaption>Blau: \\(f(z)\\), lila gestrichelt: \\(f'(z)\\). Die Punkte zeigen dieses Neuron: \\(f(${tn(z)}) = ${tn(h)}\\) und \\(f'(${tn(z)}) = ${tn(fd)}\\).</figcaption></figure></div></li>
        <li><span class="nr">3</span><div><b>Fehlersignal</b> (Backpropagation, kommt von hinten). Erst sammelt das Neuron, was die ${nx} Neuronen ${last ? 'der Ausgabe' : `von Schicht ${l + 1}`} zurückgeben, dann multipliziert es mit \\(f'(z)\\):
          <div class="tex">$$\\frac{\\partial L}{\\partial h^{(${l})}_{${j + 1}}} = \\sum_{k=1}^{${nx}} w_{${j + 1},k}\\, \\frac{\\partial L}{\\partial z_k} = ${tsci(gh)}$$</div>
          <div class="tex">$$\\frac{\\partial L}{\\partial z^{(${l})}_{${j + 1}}} = f'(z) \\cdot \\frac{\\partial L}{\\partial h^{(${l})}_{${j + 1}}} = ${tp(fd)} \\cdot ${tp(gh, 3)} = ${tsci(gz)}$$</div>
          <p class="klein">Bedeutung: wird \\(z^{(${l})}_{${j + 1}}\\) um eine kleine Zahl \\(\\varepsilon\\) größer, ändert sich \\(L\\) um ungefähr \\(${tsci(gz)} \\cdot \\varepsilon\\). ${Math.abs(fd) < 1e-9 ? '<b>Hier ist \\(f\'(z) = 0\\): kein Fehlersignal kommt durch.</b>' : ''}</p></div></li>
      </ol>
      <h3>Was dieses Neuron „sieht“</h3>
      <figure><canvas class="plot" id="pl-map" style="max-width:260px"></canvas><figcaption>Wert \\(h^{(${l})}_{${j + 1}}\\) für jeden Punkt \\((x_1, x_2)\\) der Eingabefläche (\\(x_1\\) nach rechts, \\(x_2\\) nach oben). Blau = positiv, orange = negativ, grau = 0. In Schicht 1 ist die Grenze immer eine Gerade, dahinter werden Geraden kombiniert.</figcaption></figure>
      <h3>Eingehende Gewichte</h3>${this.weightTable(l, j)}`;
  }
  weightTable(l, j) {
    const W = this.net.W[l - 1], n = W.shape[1], tr = this.tr, gz = tr.gz[l - 1][j], L = this.net.L;
    let rows = `<tr ${selAttr({ kind: 'edge', l: l - 1, i: -1, j })}><td>\\(w_{0,${j + 1}}\\) (Bias)</td><td class="r">${num(this.net.b[l - 1].d[j], 3)}</td><td class="r">${sci(gz)}</td></tr>`;
    for (let i = 0; i < n; i++) rows += `<tr ${selAttr({ kind: 'edge', l: l - 1, i, j })}><td>\\(w_{${i + 1},${j + 1}}\\) von \\(${this.inSym(l - 1, i + 1)}\\)</td><td class="r">${num(W.d[j * n + i], 3)}</td><td class="r">${sci(gz * tr.inp[l - 1][i])}</td></tr>`;
    return `<div class="tabelle-wrap"><table class="tabelle"><tr><th>Gewicht</th><th class="r">Wert</th><th class="r">\\(\\partial L / \\partial w\\)</th></tr>${rows}</table></div>
      <p class="klein">Zeile ${j + 1} von \\(\\mathbf{W}_{${l}}\\) enthält genau diese Gewichte (L2: „row j of W = all weights into neuron j“). Antippen zeigt ein Gewicht im Bild.</p>`;
  }
  pOutputNeuron(k) {
    const tr = this.tr, L = this.net.L, { gen, numTex } = this.sumTex(L, k), y = this.sample.y === k ? 1 : 0;
    return `${backBtn}<h2>Ausgabeneuron ${k + 1}<span class="chip">Klasse ${k + 1}</span></h2>
      <p class="unter">Ein Ausgabeneuron gibt es pro Klasse. Es rechnet eine Summe wie jedes Neuron, aber ohne eigene Aktivierungsfunktion: danach kommt Softmax über alle Ausgabeneuronen.</p>
      <ol class="schritte">
        <li><span class="nr">1</span><div><b>Logit</b> \\(z_{${k + 1}}\\):<div class="tex">$$${gen}$$</div><div class="tex">$$${numTex}$$</div></div></li>
        <li><span class="nr">2</span><div><b>Softmax</b> macht aus allen Logits Wahrscheinlichkeiten, die zusammen \\(1\\) ergeben:
          <div class="tex">$$\\hat{y}_{${k + 1}} = \\frac{e^{z_{${k + 1}}}}{\\sum_{j=1}^{${this.net.sizes[L]}} e^{z_j}} = ${tn(tr.probs[k], 3)}$$</div></div></li>
        <li><span class="nr">3</span><div><b>Fehlersignal</b>: Softmax und Cross-Entropy zusammen ergeben einen sehr einfachen Gradienten. Hier ist \\(y_{${k + 1}} = ${y}\\), weil die richtige Klasse ${y ? 'diese' : 'eine andere'} ist:
          <div class="tex">$$\\frac{\\partial L}{\\partial z_{${k + 1}}} = \\hat{y}_{${k + 1}} - y_{${k + 1}} = ${tn(tr.probs[k], 3)} - ${y} = ${tn(tr.gz[L - 1][k], 3)}$$</div>
          <p class="klein">Negativ heißt: \\(z_{${k + 1}}\\) größer machen senkt den Loss. Positiv: kleiner machen senkt den Loss.</p></div></li>
      </ol>
      <h3>Wo das Netz Klasse ${k + 1} sagt</h3>
      <figure><canvas class="plot" id="pl-map" style="max-width:260px"></canvas><figcaption>\\(\\hat{y}_{${k + 1}}\\) über der Eingabefläche: kräftig = Wahrscheinlichkeit nahe 1.</figcaption></figure>
      <h3>Eingehende Gewichte</h3>${this.weightTable(L, k)}`;
  }
  pInput(j) {
    const tr = this.tr;
    return `${backBtn}<h2>Eingabe \\(\\mathbf{x}\\)<span class="chip">Schicht 0</span></h2>
      <p>Die Eingabe ist ein Punkt der Fläche links: \\(x_1\\) ist die waagerechte, \\(x_2\\) die senkrechte Koordinate. Die Eingabeschicht rechnet nichts, sie gibt die Zahlen nur weiter.</p>
      <div class="tex">$$\\mathbf{x} = \\begin{pmatrix} x_1 \\\\ x_2 \\end{pmatrix} = \\begin{pmatrix} ${tn(tr.x[0])} \\\\ ${tn(tr.x[1])} \\end{pmatrix}$$</div>
      <p>Darunter sitzt das Bias-Neuron mit dem festen Wert \\(1\\). So wird der Bias zu einem ganz normalen Gewicht \\(w_{0,j}\\) (Bias-Trick aus L2): \\(\\mathbf{x}' = (1, x_1, x_2)^\\top\\).</p>
      <p class="klein">Einen anderen Punkt wählst du, indem du auf die Fläche „Eingaberaum“ tippst.</p>${j !== null ? `<p>Gewählt: \\(x_{${j + 1}} = ${tn(tr.x[j], 3)}\\).</p>` : ''}`;
  }
  pBias(l) {
    const sizes = this.net.sizes;
    return `${backBtn}<h2>Bias-Neuron vor Schicht ${l + 1}</h2>
      <p>Dieses Neuron hat immer den Wert \\(1\\). Sein Gewicht in Neuron \\(j\\) ist der Bias \\(w_{0,j}\\). Dadurch gilt für jedes Neuron einfach \\(z_j = \\mathbf{w}'^\\top_j \\mathbf{x}'\\) mit \\(\\mathbf{x}' = (1, x_1, \\dots, x_n)^\\top\\) (Bias-Trick, L2).</p>
      <p>Der Bias verschiebt die Grenze \\(z_j = 0\\): ohne ihn müsste jede Gerade durch \\((0|0)\\) gehen.</p>
      <h3>Biases von Schicht ${l + 1}</h3><div class="tabelle-wrap"><table class="tabelle"><tr><th>Neuron</th><th class="r">\\(w_{0,j}\\)</th></tr>${Array.from({ length: sizes[l + 1] }, (_, j) => `<tr ${selAttr({ kind: 'edge', l, i: -1, j })}><td>${j + 1}</td><td class="r">${num(this.net.b[l].d[j], 3)}</td></tr>`).join('')}</table></div>`;
  }
  pEdge(s) {
    const tr = this.tr, L = this.net.L, w = this.weight(s), toL = s.l + 1, isOut = toL === L;
    const zSym = isOut ? `z_{${s.j + 1}}` : `z^{(${toL})}_{${s.j + 1}}`;
    const inS = s.i < 0 ? '1' : this.inSym(s.l, s.i + 1), inV = s.i < 0 ? 1 : tr.inp[s.l][s.i], gz = tr.gz[s.l][s.j];
    const wS = s.i < 0 ? `w_{0,${s.j + 1}}` : `w_{${s.i + 1},${s.j + 1}}`;
    return `${backBtn}<h2>Gewicht \\(${wS}\\)<span class="chip">in \\(\\mathbf{W}_{${toL}}\\)</span></h2>
      <p class="unter">${s.i < 0 ? 'Bias von' : `Verbindung von \\(${inS}\\) zu`} Neuron ${s.j + 1} in ${isOut ? 'der Ausgabe' : 'Schicht ' + toL}. Steht in Zeile ${s.j + 1}${s.i < 0 ? ' (Bias-Spalte)' : `, Spalte ${s.i + 1}`} von \\(\\mathbf{W}_{${toL}}\\).</p>
      <div class="tex">$$${wS} = ${tn(w, 3)}$$</div>
      <p>Dicke des Stabs = Betrag \\(|w|\\), Farbe = Vorzeichen (blau positiv, orange negativ).</p>
      <h3>Gradient für den gewählten Punkt</h3>
      <p>Das Gewicht wird mit dem Wert multipliziert, der durch den Stab kommt. Darum ist sein Gradient (Kettenregel) „Fehlersignal am Ziel mal Wert am Anfang“:</p>
      <div class="tex">$$\\frac{\\partial L}{\\partial ${wS}} = \\frac{\\partial L}{\\partial ${zSym}} \\cdot ${inS} = ${tp(gz, 3)} \\cdot ${tp(inV)} = ${tsci(gz * inV)}$$</div>
      <h3>Update</h3>
      <p>Beim Training gilt \\(w \\leftarrow w - \\eta \\cdot \\frac{\\partial J}{\\partial w}\\), wobei \\(J\\) der mittlere Loss eines Mini-Batches ist. Letzte Änderung: \\(\\Delta w = ${tsci(this.delta(s))}\\).</p>`;
  }
  pLayer(l) {
    const sizes = this.net.sizes, key = this.net.cfg.acts[l - 1], nIn = sizes[l - 1], nOut = sizes[l], inV = l === 1 ? '\\mathbf{x}' : `\\mathbf{h}^{(${l - 1})}`;
    return `${backBtn}<h2>Schicht ${l}<span class="chip">Fully Connected + ${ACTINFO[key].name}</span></h2>
      <p class="unter">${nOut} Neuronen, jedes ist mit allen ${nIn} Werten davor verbunden (plus Bias).</p>
      <ol class="schritte">
        <li><span class="nr">1</span><div><b>Matrix mal Vektor.</b> Alle Summen der Schicht auf einmal:
          <div class="tex">$$\\mathbf{z}^{(${l})} = \\mathbf{W}_{${l}}\\, ${inV} + \\mathbf{w}_0, \\qquad \\mathbf{W}_{${l}} \\in \\mathbb{R}^{${nOut} \\times ${nIn}}$$</div>
          <p class="klein">Zeile \\(j\\) von \\(\\mathbf{W}_{${l}}\\) = alle Gewichte in Neuron \\(j\\). Parameter: \\(${nOut} \\cdot (${nIn} + 1) = ${nOut * (nIn + 1)}\\) (das \\(+1\\) ist der Bias).</p></div></li>
        <li><span class="nr">2</span><div><b>Aktivierung</b> elementweise: \\(\\mathbf{h}^{(${l})} = f(\\mathbf{z}^{(${l})})\\).
          ${auswahl('act-' + l, 'Aktivierung', ACTKEYS.map((k) => [k, ACTINFO[k].name]), key)}
          ${actCard(key, undefined, 'pl-act')}</div></li>
      </ol>
      ${+this.cfg.dropout > 0 ? `<div class="karte">Nach dieser Schicht: Dropout mit \\(p = ${tn(+this.cfg.dropout, 1)}\\) (nur beim Training).</div>` : ''}
      <p class="klein">Die Aktivierung lässt sich ändern, ohne neu zu initialisieren: das Netz rechnet sofort mit der neuen Funktion weiter.</p>`;
  }
  pSoftmax() {
    const tr = this.tr, L = this.net.L, K = this.net.sizes[L], z = tr.z[L - 1];
    const rows = Array.from({ length: K }, (_, k) => `<tr class="${this.sample.y === k ? 'aktiv' : ''}"><td>Klasse ${k + 1}${this.sample.y === k ? ' (richtig)' : ''}</td><td class="r">${num(z[k])}</td><td class="r">${num(Math.exp(z[k]), 2)}</td><td class="r">${num(tr.probs[k], 3)}</td></tr>`).join('');
    return `${backBtn}<h2>Ausgabe mit Softmax</h2>
      <p class="unter">Die letzte Fully-Connected-Schicht gibt \\(K = ${K}\\) Logits \\(z_k\\). Softmax macht daraus eine Wahrscheinlichkeitsverteilung \\(\\hat{\\mathbf{y}}\\).</p>
      <div class="tex">$$\\hat{y}_k = \\frac{e^{z_k}}{\\sum_{j=1}^{K} e^{z_j}}$$</div>
      <ol class="schritte"><li><span class="nr">1</span><div>\\(e^{z_k}\\) ist immer positiv, auch für negative \\(z_k\\).</div></li><li><span class="nr">2</span><div>Durch die Summe teilen: alle \\(\\hat{y}_k\\) liegen zwischen 0 und 1 und ergeben zusammen \\(1\\).</div></li></ol>
      <h3>Für den gewählten Punkt</h3>
      <div class="tabelle-wrap"><table class="tabelle"><tr><th>Label</th><th class="r">\\(z_k\\)</th><th class="r">\\(e^{z_k}\\)</th><th class="r">\\(\\hat{y}_k\\)</th></tr>${rows}</table></div>
      <p class="klein">Die Ground Truth ist one-hot: \\(\\mathbf{y} = (${Array.from({ length: K }, (_, k) => (k === this.sample.y ? 1 : 0)).join(', ')})^\\top\\). Softmax ist die Aktivierung der letzten Schicht, Sigmoid/Tanh/ReLU sind Aktivierungen der versteckten Schichten.</p>`;
  }
  pLoss() {
    const tr = this.tr, k = this.sample.y, L = this.net.L, K = this.net.sizes[L], e = this.ev;
    return `${backBtn}<h2>Loss: Cross-Entropy</h2>
      <p class="unter">Der Loss misst, wie schlecht die Vorhersage \\(\\hat{\\mathbf{y}}\\) zur Ground Truth \\(\\mathbf{y}\\) passt.</p>
      <div class="tex">$$L(\\mathbf{y}, \\hat{\\mathbf{y}}) = -\\sum_{k=1}^{${K}} y_k \\log \\hat{y}_k = -\\log \\hat{y}_k \\Big|_{y_k = 1}$$</div>
      <p>Weil \\(\\mathbf{y}\\) one-hot ist, bleibt von der Summe nur der Summand der richtigen Klasse übrig.</p>
      <h3>Für den gewählten Punkt</h3>
      <div class="tex">$$L = -\\log \\hat{y}_{${k + 1}} = -\\log(${tn(tr.probs[k], 3)}) = ${tn(tr.loss, 3)}$$</div>
      <div class="karte hinweis"><b>Warum das Minus?</b> \\(\\hat{y}_{${k + 1}}\\) liegt zwischen 0 und 1, also ist \\(\\log \\hat{y}_{${k + 1}} \\le 0\\). Das Minus macht den Loss positiv. \\(\\hat{y}_{${k + 1}} = 1\\) ergibt \\(L = 0\\), \\(\\hat{y}_{${k + 1}} \\to 0\\) ergibt \\(L \\to \\infty\\). (Das Minus beim Gradientenabstieg ist ein anderes: dort geht es um die Richtung des Schritts.)</div>
      <h3>Start der Backpropagation</h3>
      <div class="tex">$$\\frac{\\partial L}{\\partial \\mathbf{z}} = \\hat{\\mathbf{y}} - \\mathbf{y} = \\begin{pmatrix} ${Array.from({ length: K }, (_, q) => tn(tr.gz[L - 1][q], 3)).join(' \\\\ ')} \\end{pmatrix}$$</div>
      <h3>Über alle Daten</h3><p>Trainiert wird der Mittelwert \\(J = \\frac{1}{N}\\sum_{i=1}^{N} L(\\mathbf{y}^{(i)}, \\hat{\\mathbf{y}}^{(i)})\\), hier mit \\(N = ${this.data.X.length}\\): \\(J = ${tn(e.loss, 3)}\\).</p>`;
  }
  pData() {
    const d = Models.DATASETS[this.cfg.data], tr = this.tr, K = this.data.K;
    return `${backBtn}<h2>Eingaberaum: ${d.name}</h2><p class="unter">${d.text}</p>
      <p>Jeder Punkt ist ein Trainingsbeispiel \\((\\mathbf{x}^{(i)}, y^{(i)})\\), die Farbe ist seine Klasse. Der Hintergrund zeigt, welche Klasse das Netz an jeder Stelle vorhersagt. Die Grenze zwischen den Farben ist die <b>Entscheidungsgrenze</b>.</p>
      ${auswahl('data', 'Datensatz', Object.entries(Models.DATASETS).map(([k, v]) => [k, v.name]), this.cfg.data)}
      <h3>Gewählter Punkt</h3>
      <div class="tex">$$\\mathbf{x} = (${tn(tr.x[0])},\\ ${tn(tr.x[1])}), \\quad \\text{Klasse } ${this.sample.y + 1}$$</div>
      ${balkenHTML(Array.from({ length: K }, (_, k) => 'Klasse ' + (k + 1)), tr.probs, T.cls.map((c) => c.getStyle()), this.sample.y)}
      <p class="klein">Tipp auf die Fläche wählt den nächsten Datenpunkt. Danach zeigen „Vorwärts“ und „Rückwärts“ genau diesen Punkt.</p>`;
  }
  overview() {
    const c = this.cfg, sizes = this.net.sizes, L = sizes.length - 1;
    let nParam = 0; for (let l = 0; l < L; l++) nParam += sizes[l + 1] * (sizes[l] + 1);
    const lrS = Math.log10(+c.lr).toFixed(1);
    return `<h2>Fully Connected Netz</h2>
      <p class="unter">Ein echtes kleines Netz, das hier im Browser trainiert: ${sizes.join(' → ')} Neuronen, Softmax, Cross-Entropy, Backpropagation. Tippe auf alles, was du siehst.</p>
      <div class="legende">
        <div><span class="punkt" style="background:var(--pos)"></span><span>Kugel = Neuron. Farbe = sein Wert \\(h\\) beim gewählten Punkt: blau positiv, orange negativ, grau nahe 0.</span></div>
        <div><span class="stab" style="background:linear-gradient(90deg,var(--pos) 50%,var(--neg) 50%)"></span><span>Stab = Gewicht \\(w_{i,j}\\). Dicke = Betrag, Farbe = Vorzeichen.</span></div>
        <div><span class="quadrat" style="background:var(--pos)"></span><span>Würfel „1“ = Bias-Neuron (Bias-Trick).</span></div>
        <div><span class="punkt" style="background:var(--grad);opacity:.6"></span><span>Lila Hülle = Fehlersignal \\(\\partial L/\\partial z\\) nach „Rückwärts“.</span></div>
      </div>
      <p class="klein">Ziehen dreht, zwei Finger oder Mausrad zoomen, Rechtsklick verschiebt. Doppelklick fliegt hin.</p>
      <h3>Zum Entdecken</h3>
      ${EXPERIMENTE.map((e, i) => `<details class="auftrag"><summary>${e.titel}</summary><p>${e.frage}</p><div class="knopfgruppe"><button type="button" class="knopf haupt" data-act="exp" data-i="${i}">Einstellen und trainieren</button></div><details class="innen"><summary>Was passiert?</summary><p>${e.antwort}</p></details></details>`).join('')}
      <div class="knopfgruppe"><button type="button" class="knopf zart" data-act="basis">Alles zurück auf Start</button></div>
      <h3>Daten</h3>
      <div class="segment klein" style="flex-wrap:wrap">${Object.entries(Models.DATASETS).map(([k, v]) => `<button type="button" data-act="data" data-k="${k}" aria-pressed="${c.data === k}">${v.name}</button>`).join('')}</div>
      <h3>Architektur</h3>
      <div class="schicht-zeile" style="grid-template-columns:1fr auto"><span><b>Eingabe</b> · 2 Werte \\(x_1, x_2\\)</span><span class="klein">fest</span></div>
      ${c.hidden.map((n, l) => `<div class="schicht-zeile"><b>${l + 1}</b><span class="zahl-knopf"><button type="button" data-act="minus" data-l="${l}" aria-label="ein Neuron weniger">−</button><span>${n}</span><button type="button" data-act="plus" data-l="${l}" aria-label="ein Neuron mehr">+</button></span><select data-set="act-${l + 1}" aria-label="Aktivierung Schicht ${l + 1}">${ACTKEYS.map((k) => `<option value="${k}" ${c.acts[l] === k ? 'selected' : ''}>${ACTINFO[k].name}</option>`).join('')}</select><button type="button" class="x-knopf" data-act="del" data-l="${l}" aria-label="Schicht entfernen">×</button></div>`).join('')}
      ${c.hidden.length < 4 ? '<div class="knopfgruppe"><button type="button" class="knopf zart" data-act="add">+ Versteckte Schicht</button></div>' : ''}
      <div class="schicht-zeile" style="grid-template-columns:1fr auto"><span><b>Ausgabe</b> · ${sizes[L]} Klassen, Softmax</span><span class="klein">fest</span></div>
      <p class="klein">Parameter insgesamt: \\(${nParam}\\) (jede Schicht: Neuronen · (Eingänge + 1)).</p>
      <h3>Training</h3>
      ${auswahl('opt', 'Optimierer', [['sgd', 'SGD'], ['momentum', 'SGD + Momentum'], ['adam', 'Adam']], c.opt)}
      <div class="zeile"><label for="s-lr">Lernrate \\(\\eta\\)</label><input id="s-lr" type="range" min="-3" max="0.5" step="0.1" value="${lrS}" data-set="lr"><output id="o-lr">${num(+c.lr, 3)}</output></div>
      ${auswahl('batch', 'Batchgröße \\(B\\)', [[1, '1 (SGD)'], [8, '8'], [16, '16'], [64, '64'], [200, '200 (ganzer Datensatz)']], c.batch)}
      ${auswahl('l2', 'L2 \\(\\lambda\\)', [[0, 'aus'], [0.0001, '0,0001'], [0.001, '0,001'], [0.01, '0,01']], c.l2)}
      ${auswahl('dropout', 'Dropout \\(p\\)', [[0, 'aus'], [0.1, '0,1'], [0.3, '0,3'], [0.5, '0,5']], c.dropout)}
      ${auswahl('init', 'Initialisierung', [['he', 'He: \\(\\sigma = \\sqrt{2/\\text{fan\\_in}}\\)'], ['xavier', 'Xavier/Glorot'], ['gleich', 'Alle gleich (0,3)']], c.init)}
      ${auswahl('tempo', 'Tempo', [[1, 'langsam (1 Schritt pro Bild)'], [2, 'normal'], [8, 'schnell'], [30, 'sehr schnell']], c.tempo)}
      <p class="klein">Ein Trainingsschritt = ein Mini-Batch: Vorwärts, Loss, Rückwärts, Update. Eine Epoche = einmal alle ${this.data.X.length} Punkte.</p>
      <h3>Ansicht</h3>
      ${schalter('bias', 'Bias-Neuronen zeigen', c.bias)}${schalter('zahlen', 'Zahlen an den Neuronen', c.zahlen)}${schalter('karten', 'Neuronen als Karten (was jedes Neuron sieht)', c.karten)}`;
  }
  afterPanel(el, s) {
    const tr = this.tr;
    if (s && s.kind === 'neuron' && s.l > 0) {
      const cv = $('#pl-act', el); if (cv) plotAct(cv, this.net.cfg.acts[s.l - 1], tr.z[s.l - 1][s.j]);
      const mv = $('#pl-map', el);
      if (mv) { const G = 44, vals = this.gridValues(G); plotMap(mv, vals[s.l][s.j], G, this.data, s.l === this.net.L ? 'prob' : 'val', s.j); }
    }
    if (s && s.kind === 'layer' && s.l > 0 && s.l < this.net.L) { const cv = $('#pl-act', el); if (cv) plotAct(cv, this.net.cfg.acts[s.l - 1]); }
  }
  live(el, s) {
    if (!s || s.kind === 'layer' || s.kind === 'data') return;
    const now = performance.now();
    if (this.playing && now - this.lastLive < 1200) { App.liveDirty = true; return; }
    this.lastLive = now; App.renderPanel(false);
  }
  // ---------------- Einstellungen
  restart(full) {
    this.playing = false; this.anim = null; Pulses.clear();
    if (full) { this.data = Models.makeData(this.cfg.data); }
    this.newNet(); this.sample = this.nearest(this.sample ? this.sample.x[0] : 0.45, this.sample ? this.sample.x[1] : -0.55);
    this.build(); this.trace(); App.select(null);
  }
  act(name, el) {
    const c = this.cfg, l = +el.dataset.l;
    if (name === 'exp') {
      const e = EXPERIMENTE[+el.dataset.i];
      Object.assign(this.cfg, JSON.parse(JSON.stringify(MLP_BASE)), JSON.parse(JSON.stringify(e.cfg)), { bias: c.bias, zahlen: c.zahlen, karten: c.karten });
      this.restart(true); this.playing = true; this.autoStop = 900; App.updateDock();
      App.caption(`<b>${e.titel}</b>: ${e.frage}`); setTimeout(() => { if (!this.anim) App.caption(null); }, 6000);
      return;
    }
    if (name === 'basis') { Object.assign(this.cfg, JSON.parse(JSON.stringify(MLP_BASE))); this.restart(true); App.updateDock(); return; }
    if (name === 'data') { c.data = el.dataset.k; this.restart(true); return; }
    if (name === 'plus') { c.hidden[l] = Math.min(16, c.hidden[l] + 1); this.restart(); return; }
    if (name === 'minus') { c.hidden[l] = Math.max(1, c.hidden[l] - 1); this.restart(); return; }
    if (name === 'del') { c.hidden.splice(l, 1); c.acts.splice(l, 1); this.restart(); return; }
    if (name === 'add') { c.hidden.push(4); c.acts.push(c.acts[c.acts.length - 1] || 'relu'); this.restart(); return; }
  }
  setOpt(key, val, el, type) {
    const c = this.cfg, net = this.net;
    if (key === 'lr') {
      c.lr = +Math.pow(10, +val).toPrecision(2); net.opt.set({ lr: c.lr });
      const o = $('#o-lr'); if (o) o.textContent = num(c.lr, 3); return;
    }
    if (type === 'input' && el.tagName === 'SELECT') return; // „change“ reicht
    if (key.startsWith('act-')) {
      const l = +key.slice(4); c.acts[l - 1] = val; net.cfg.acts[l - 1] = val;
      this.layerLbl[l].element.innerHTML = `Schicht ${l} <small>${ACTINFO[val].name}</small>`;
      this.trace(); this.ev = net.evaluate(this.data); this.dirty = true; App.renderPanel(); App.updateDock(); return;
    }
    if (key === 'data') { c.data = val; this.restart(true); return; }
    if (key === 'opt') { c.opt = val; net.opt.set({ type: val }); net.opt.reset(); return; }
    if (key === 'batch') { c.batch = +val; net.cfg.batch = +val; return; }
    if (key === 'l2') { c.l2 = +val; net.opt.set({ l2: +val }); return; }
    if (key === 'dropout') { c.dropout = +val; net.cfg.dropout = +val; return; }
    if (key === 'init') { c.init = val; this.newNet(); this.trace(); this.dirty = true; App.updateDock(); return; }
    if (key === 'tempo') { c.tempo = +val; return; }
    if (key === 'bias' || key === 'zahlen' || key === 'karten') { c[key] = !!val; this.applyView(); return; }
  }
}
App.modes.mlp = new MLPMode();
