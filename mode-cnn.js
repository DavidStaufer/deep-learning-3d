/* ============================================================
   Modus 2: Convolutional Neural Network (L4)
   ============================================================ */
const CNN_U = 0.13; // Abstand zweier Pixel in 3D
const CNN_LAYERS = [
  { key: 'in', name: 'Eingabe', short: 'Eingabe', C: 1, H: 16, W: 16, x: 0, grid: [1, 1], gap: 0 },
  { key: 'c1', name: 'Conv 1 + ReLU', short: 'Conv 1', C: 4, H: 16, W: 16, x: 3.9, grid: [2, 2], gap: 0.3 },
  { key: 'p1', name: 'Max Pooling 1', short: 'Pool 1', C: 4, H: 8, W: 8, x: 7.2, grid: [2, 2], gap: 0.22 },
  { key: 'c2', name: 'Conv 2 + ReLU', short: 'Conv 2', C: 8, H: 8, W: 8, x: 10.1, grid: [2, 4], gap: 0.22 },
  { key: 'p2', name: 'Max Pooling 2', short: 'Pool 2', C: 8, H: 4, W: 4, x: 12.6, grid: [2, 4], gap: 0.16 },
];
const CNN_X = { flat: 14.4, fc1: 16.1, fc2: 18.0, loss: 21.6 };
const CNN_LY = -2.1; // Loss unterhalb der Ausgabe

class CNNMode {
  constructor() {
    this.key = 'cnn'; this.group = new THREE.Group(); this.built = false;
    this.cnn = null; this.playing = false; this.anim = null; this.hist = []; this.acc = 0; this.lossEMA = null;
    this.vol = 0; this.volTarget = 0; this.relief = true; this.gradView = false; this.lr = 0.004;
    this.reveal = null; this.autoStop = 0; this.lastLines = 0; this.lastLive = 0; this.tool = 'mal'; this.seedN = 21;
  }
  init() {
    DL.seed(this.seedN); this.cnn = new Models.CNN();
    this.newExample(2);
    this.build(); this.trace();
    this.playing = true; this.autoStop = 350;
  }
  enter() { if (!this.cnn.steps && !this.playing) { this.playing = true; this.autoStop = 350; } }
  newExample(cls) { this.label = cls; this.img = Models.makeShape(cls); this.userLabel = false; }
  trace(train = false) {
    const r = this.cnn.forward(this.img, this.label, train);
    DL.backward(r.loss);
    const cp = (v) => Float64Array.from(v.d), cg = (v) => (v.g ? Float64Array.from(v.g) : new Float64Array(v.n));
    this.tr = {
      vals: [cp(r.x), cp(r.r1), cp(r.p1), cp(r.r2), cp(r.p2)], grads: [cg(r.x), cg(r.r1), cg(r.p1), cg(r.r2), cg(r.p2)],
      z1: cp(r.c1), z2: cp(r.c2), arg1: Int32Array.from(r.p1.arg), arg2: Int32Array.from(r.p2.arg),
      flat: cp(r.f), gflat: cg(r.f), z3: cp(r.z3), h3: cp(r.h3), g3: cg(r.z3), mask: r.d3.mask ? Float64Array.from(r.d3.mask) : null,
      z4: cp(r.z4), g4: cg(r.z4), probs: Float64Array.from(r.probs), loss: r.loss.d[0],
    };
    for (const p of this.cnn.params) p.g.fill(0);
  }
  // ---------------- Positionen
  mapCenter(Li, c, vol) {
    const D = CNN_LAYERS[Li], size = D.W * CNN_U, [cols, rows] = D.grid, col = c % cols, row = Math.floor(c / cols);
    const sx = D.x + (col - (cols - 1) / 2) * (size + D.gap), sy = ((rows - 1) / 2 - row) * (size + D.gap);
    const vx = D.x, vy = 0, vz = (c - (D.C - 1) / 2) * 0.3;
    return V3(sx + (vx - sx) * vol, sy + (vy - sy) * vol, vz * vol);
  }
  tilePos(Li, c, r, q, vol, out = V3()) {
    const D = CNN_LAYERS[Li], m = this.mapCenter(Li, c, vol);
    return out.set(m.x + (q - (D.W - 1) / 2) * CNN_U, m.y + ((D.H - 1) / 2 - r) * CNN_U, m.z);
  }
  flatPos(k, out = V3()) { const col = Math.floor(k / 32), row = k % 32; return out.set(CNN_X.flat + (col - 1.5) * 0.085, (15.5 - row) * 0.085, 0); }
  fcPos(l, j, out = V3()) {
    if (l === 1) { const col = Math.floor(j / 8), row = j % 8; return out.set(CNN_X.fc1 + (col - 0.5) * 0.25, (3.5 - row) * 0.36, (col - 0.5) * 0.4); }
    return out.set(CNN_X.fc2, (1.5 - j) * 0.62, 0);
  }
  view() { return viewFor(V3(-1.3, -2.8, -1), V3(CNN_X.loss + 0.6, 2.9, 1), App.phone() ? V3(1, 0.42, 0.62) : V3(0.38, 0.3, 1)); }
  build() {
    disposeGroup(this.group);
    const g = this.group, box = new THREE.BoxGeometry(1, 1, 1);
    this.meshes = CNN_LAYERS.map((D, Li) => {
      const m = instanced(box, new THREE.MeshStandardMaterial({ roughness: 0.55 }), D.C * D.H * D.W);
      m.userData.pick = (h) => { const id = h.instanceId, per = D.H * D.W, c = Math.floor(id / per), rq = id % per; return { kind: 'tile', L: Li, c, r: Math.floor(rq / D.W), q: rq % D.W }; };
      g.add(m); return m;
    });
    this.flatMesh = instanced(box, new THREE.MeshStandardMaterial({ roughness: 0.55 }), 128);
    this.flatMesh.userData.pick = (h) => ({ kind: 'flat', k: h.instanceId });
    this.fc1Mesh = instanced(new THREE.SphereGeometry(0.13, 20, 14), new THREE.MeshStandardMaterial({ roughness: 0.4 }), 16);
    this.fc1Mesh.userData.pick = (h) => ({ kind: 'fc', l: 1, j: h.instanceId });
    this.fc2Mesh = instanced(new THREE.SphereGeometry(0.18, 24, 16), new THREE.MeshStandardMaterial({ roughness: 0.4 }), 4);
    this.fc2Mesh.userData.pick = (h) => ({ kind: 'fc', l: 2, j: h.instanceId });
    this.barMesh = instanced(new THREE.BoxGeometry(1, 0.3, 0.3), new THREE.MeshStandardMaterial({ roughness: 0.4 }), 4);
    this.barMesh.userData.pick = () => ({ kind: 'layer', L: 'out' });
    this.lossMesh = new THREE.Mesh(new THREE.IcosahedronGeometry(0.34, 3), new THREE.MeshStandardMaterial({ roughness: 0.35 }));
    this.lossMesh.position.set(CNN_X.loss, CNN_LY, 0); this.lossMesh.userData.pick = () => ({ kind: 'loss' });
    g.add(this.flatMesh, this.fc1Mesh, this.fc2Mesh, this.barMesh, this.lossMesh);
    for (let k = 0; k < 128; k++) setInst(this.flatMesh, k, this.flatPos(k), 0.065);
    for (let j = 0; j < 16; j++) setInst(this.fc1Mesh, j, this.fcPos(1, j), 1);
    for (let j = 0; j < 4; j++) setInst(this.fc2Mesh, j, this.fcPos(2, j), 1);
    // Verbindungen der Fully-Connected-Schichten als feine Linien
    const seg = (n) => { const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(n * 6), 3)); geo.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(n * 6), 3)); return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true })); };
    this.l3 = seg(128 * 16); this.l4 = seg(16 * 4); this.l5 = seg(4);
    const a = V3(), b = V3();
    let pa = this.l3.geometry.attributes.position;
    for (let j = 0; j < 16; j++) for (let k = 0; k < 128; k++) { this.flatPos(k, a); this.fcPos(1, j, b); const i = j * 128 + k; pa.setXYZ(2 * i, a.x + 0.03, a.y, a.z); pa.setXYZ(2 * i + 1, b.x, b.y, b.z); }
    pa = this.l4.geometry.attributes.position;
    for (let k = 0; k < 4; k++) for (let j = 0; j < 16; j++) { this.fcPos(1, j, a); this.fcPos(2, k, b); const i = k * 16 + j; pa.setXYZ(2 * i, a.x, a.y, a.z); pa.setXYZ(2 * i + 1, b.x, b.y, b.z); }
    pa = this.l5.geometry.attributes.position;
    for (let k = 0; k < 4; k++) { this.fcPos(2, k, a); pa.setXYZ(2 * k, a.x + 2.25, a.y, a.z); pa.setXYZ(2 * k + 1, CNN_X.loss - 0.25, CNN_LY + 0.15, 0); }
    g.add(this.l3, this.l4, this.l5);
    // Rahmen für Auswahl und rezeptive Felder
    this.rf = new THREE.Group(); g.add(this.rf);
    this.scan = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color: T.c.akzent }));
    this.scan.visible = false; g.add(this.scan);
    this.ring = new THREE.Mesh(new THREE.TorusGeometry(0.24, 0.02, 8, 40), new THREE.MeshBasicMaterial({ color: T.c.akzent })); this.ring.visible = false; g.add(this.ring);
    g.add(shadowDisc(26, 8, -3.2));
    // Beschriftungen
    this.layerLbl = {};
    const addL = (key, html, pos) => { const o = label(html, 'lbl', () => App.select({ kind: 'layer', L: key })); o.position.copy(pos); g.add(o); this.layerLbl[key] = o; };
    CNN_LAYERS.forEach((D, Li) => {
      const [cols, rows] = D.grid, size = D.W * CNN_U, top = ((rows - 1) / 2) * (size + D.gap) + size / 2;
      addL(Li, `${D.short} <small>${D.W}×${D.H}×${D.C}</small>`, V3(D.x, D.key[0] === 'p' ? -top - 0.42 : top + 0.45, 0));
    });
    addL('flat', 'Flatten <small>128</small>', V3(CNN_X.flat, 1.75, 0));
    addL('fc1', 'FC + ReLU <small>16</small>', V3(CNN_X.fc1, -1.85, 0));
    addL('out', 'Softmax <small>4</small>', V3(CNN_X.fc2 + 0.9, 1.65, 0));
    this.lossLbl = label('Loss', 'lbl', () => App.select({ kind: 'loss' })); this.lossLbl.position.set(CNN_X.loss, CNN_LY - 0.75, 0); g.add(this.lossLbl);
    this.outLbl = Models.SHAPES.map((n, k) => { const o = label('', 'wert'); o.position.copy(this.fcPos(2, k)).add(V3(2.75, 0, 0)); g.add(o); return o; });
    this.pickables = [...this.meshes, this.flatMesh, this.fc1Mesh, this.fc2Mesh, this.barMesh, this.lossMesh];
    this.lastVol = -1;
  }
  // ---------------- Pro Bild
  frame(dt, now) {
    if (this.anim) this.runAnim(now);
    else if (this.playing) {
      const t0 = performance.now(); let n = 0;
      while (performance.now() - t0 < 11 && n < 6) {
        const r = this.cnn.trainStep(16); n++;
        this.lossEMA = this.lossEMA === null ? r.loss : 0.95 * this.lossEMA + 0.05 * r.loss;
        this.acc = this.cnn.steps < 2 ? r.acc : 0.95 * this.acc + 0.05 * r.acc;
        if (this.cnn.steps % 5 === 0) { this.hist.push(this.lossEMA); if (this.hist.length > 150) this.hist.shift(); }
      }
      this.trace(); App.liveDirty = true;
      if (this.autoStop && this.cnn.steps >= this.autoStop) { this.playing = false; this.autoStop = 0; App.updateDock(); }
    }
    const dv = this.volTarget - this.vol;
    if (Math.abs(dv) > 0.001) { this.vol += Math.sign(dv) * Math.min(Math.abs(dv), dt * 1.6); if (Math.abs(this.volTarget - this.vol) < 0.002) this.vol = this.volTarget; }
    this.updateVisuals(now);
  }
  layerShown(idx) { // Reihenfolge: 0..4 Karten, 5 flatten, 6 fc1, 7 fc2/out, 8 loss
    if (!this.reveal) return 1;
    const r = this.reveal;
    if (r.dir > 0) return idx < r.idx ? 1 : idx === r.idx ? r.frac : 0;
    return 1;
  }
  gradShown(idx) { const r = this.reveal; if (!r || r.dir > 0) return this.gradView; return idx >= r.idx; }
  updateVisuals(now) {
    const tr = this.tr, c = new THREE.Color(), vol = this.vol, sel = App.sel, pos = V3();
    const relief = this.relief && vol < 0.5;
    CNN_LAYERS.forEach((D, Li) => {
      const m = this.meshes[Li], vals = tr.vals[Li], grads = tr.grads[Li], per = D.H * D.W, frac = this.layerShown(Li), gShow = this.gradShown(Li);
      let mx = 1e-9, gmx = 1e-12; for (let i = 0; i < vals.length; i++) { mx = Math.max(mx, Math.abs(vals[i])); gmx = Math.max(gmx, Math.abs(grads[i])); }
      if (Li === 0) mx = 1;
      const selMap = sel && sel.kind === 'layer' && sel.L === Li;
      for (let i = 0; i < vals.length; i++) {
        const ch = Math.floor(i / per), rq = i % per, r = Math.floor(rq / D.W), q = rq % D.W;
        const shown = (rq + 0.5) / per <= frac;
        const v = vals[i] / mx;
        if (!shown) c.copy(T.c.null);
        else if (gShow) c.copy(T.c.null).lerp(T.c.grad, Math.min(1, Math.sqrt(Math.abs(grads[i]) / gmx)));
        else if (Li === 0) c.copy(T.c.null).lerp(T.c.tinte, Math.min(1, Math.max(0, vals[i])));
        else valueColor(v, c);
        this.tilePos(Li, ch, r, q, vol, pos);
        const depth = relief && shown && !gShow ? 0.02 + 0.32 * Math.max(0, v) : 0.03;
        pos.z += depth / 2;
        setInst(m, i, pos, CNN_U * 0.9, CNN_U * 0.9, depth);
        m.setColorAt(i, c);
      }
      m.instanceMatrix.needsUpdate = true; m.instanceColor.needsUpdate = true;
      if (this.layerLbl[Li]) this.layerLbl[Li].element.classList.toggle('aktiv', !!selMap);
    });
    // Flatten
    const fs = this.layerShown(5), fg = this.gradShown(5);
    let fm = 1e-9, fgm = 1e-12; for (let k = 0; k < 128; k++) { fm = Math.max(fm, tr.flat[k]); fgm = Math.max(fgm, Math.abs(tr.gflat[k])); }
    for (let k = 0; k < 128; k++) {
      if ((k + 0.5) / 128 > fs) c.copy(T.c.null); else if (fg) c.copy(T.c.null).lerp(T.c.grad, Math.sqrt(Math.abs(tr.gflat[k]) / fgm)); else valueColor(tr.flat[k] / fm, c);
      this.flatMesh.setColorAt(k, c);
    }
    this.flatMesh.instanceColor.needsUpdate = true;
    // FC 1 und Ausgabe
    const s1 = this.layerShown(6) >= 1, g1 = this.gradShown(6);
    let hm = 1e-9, gm = 1e-12; for (let j = 0; j < 16; j++) { hm = Math.max(hm, tr.h3[j]); gm = Math.max(gm, Math.abs(tr.g3[j])); }
    for (let j = 0; j < 16; j++) {
      const off = tr.mask && tr.mask[j] === 0;
      if (!s1 || off) c.copy(T.c.null); else if (g1) c.copy(T.c.null).lerp(T.c.grad, Math.sqrt(Math.abs(tr.g3[j]) / gm)); else valueColor(tr.h3[j] / hm, c);
      this.fc1Mesh.setColorAt(j, c); setInst(this.fc1Mesh, j, this.fcPos(1, j, pos), off ? 0.5 : 1);
    }
    this.fc1Mesh.instanceColor.needsUpdate = true; this.fc1Mesh.instanceMatrix.needsUpdate = true;
    const s2 = this.layerShown(7) >= 1;
    for (let k = 0; k < 4; k++) {
      c.copy(T.c.null); if (s2) c.lerp(T.cls[k], 0.2 + 0.8 * tr.probs[k]);
      this.fc2Mesh.setColorAt(k, c);
      const len = Math.max(0.02, (s2 ? tr.probs[k] : 0) * 1.8), p = this.fcPos(2, k, pos);
      _p.set(p.x + 0.3 + len / 2, p.y, p.z); setInst(this.barMesh, k, _p, len, 1, 1); this.barMesh.setColorAt(k, T.cls[k]);
    }
    this.fc2Mesh.instanceColor.needsUpdate = true; this.barMesh.instanceMatrix.needsUpdate = true; this.barMesh.instanceColor.needsUpdate = true;
    const lossOn = this.layerShown(8) >= 1;
    this.lossMesh.scale.setScalar(lossOn ? 0.7 + 0.5 * Math.min(1, tr.loss / 1.5) : 0.6);
    this.lossMesh.material.color.copy(T.c.null).lerp(T.c.neg, lossOn ? 0.25 + 0.75 * Math.min(1, tr.loss / 1.2) : 0);
    // Linien (nicht jedes Bild nötig)
    if (now - this.lastLines > 0.2 || this.linesDirty) { this.lastLines = now; this.linesDirty = false; this.colorLines(); }
    // Auswahl-Rahmen neu, wenn sich die Ansicht bewegt
    if (this.lastVol !== this.vol || this.rfDirty) { this.lastVol = this.vol; this.rfDirty = false; this.buildRF(); }
    if (sel && sel.kind === 'fc') { this.ring.visible = true; this.fcPos(sel.l, sel.j, this.ring.position); this.ring.scale.setScalar(sel.l === 1 ? 0.75 : 1); this.ring.quaternion.copy(camera.quaternion); } else this.ring.visible = false;
    // Texte
    Models.SHAPES.forEach((n, k) => { const t = `${n} ${s2 ? num(tr.probs[k]) : ''}${this.label === k ? ' ✓' : ''}`; if (this.outLbl[k].element.innerHTML !== t) this.outLbl[k].element.innerHTML = t; });
    const lt = lossOn ? `Loss <small>L = ${num(tr.loss, 3)}</small>` : 'Loss';
    if (this.lossLbl.element.innerHTML !== lt) this.lossLbl.element.innerHTML = lt;
  }
  colorLines() {
    const c = new THREE.Color(), sel = App.sel, W3 = this.cnn.W3.d, W4 = this.cnn.W4.d;
    const focus = sel && sel.kind === 'fc' ? sel : null, fk = sel && sel.kind === 'flat' ? sel.k : null;
    let col = this.l3.geometry.attributes.color;
    for (let j = 0; j < 16; j++) for (let k = 0; k < 128; k++) {
      const w = W3[j * 128 + k], i = j * 128 + k;
      valueColor(Math.sign(w) * Math.min(1, Math.abs(w) / 0.35), c);
      const on = (focus && focus.l === 1 && focus.j === j) || fk === k;
      c.lerp(T.c.bg, on ? 0.1 : (focus || fk !== null) ? 0.97 : 0.86);
      col.setXYZ(2 * i, c.r, c.g, c.b); col.setXYZ(2 * i + 1, c.r, c.g, c.b);
    }
    col.needsUpdate = true;
    col = this.l4.geometry.attributes.color;
    for (let k = 0; k < 4; k++) for (let j = 0; j < 16; j++) {
      const w = W4[k * 16 + j], i = k * 16 + j;
      valueColor(Math.sign(w) * Math.min(1, Math.abs(w) / 0.8), c);
      const on = focus && ((focus.l === 2 && focus.j === k) || (focus.l === 1 && focus.j === j));
      c.lerp(T.c.bg, on ? 0.05 : focus ? 0.92 : 0.55);
      col.setXYZ(2 * i, c.r, c.g, c.b); col.setXYZ(2 * i + 1, c.r, c.g, c.b);
    }
    col.needsUpdate = true;
    col = this.l5.geometry.attributes.color;
    for (let k = 0; k < 4; k++) { c.copy(T.cls[k]).lerp(T.c.bg, 0.4); col.setXYZ(2 * k, c.r, c.g, c.b); col.setXYZ(2 * k + 1, c.r, c.g, c.b); }
    col.needsUpdate = true;
  }
  // Rahmen um Fenster und Linien zum gewählten Wert
  frameBox(Li, c, r0, r1, q0, q1, color) {
    const D = CNN_LAYERS[Li]; r0 = Math.max(0, r0); q0 = Math.max(0, q0); r1 = Math.min(D.H - 1, r1); q1 = Math.min(D.W - 1, q1);
    const a = this.tilePos(Li, c, r0, q0, this.vol), b = this.tilePos(Li, c, r1, q1, this.vol);
    const w = Math.abs(b.x - a.x) + CNN_U, h = Math.abs(b.y - a.y) + CNN_U;
    const m = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, 0.06)), new THREE.LineBasicMaterial({ color }));
    m.position.set((a.x + b.x) / 2, (a.y + b.y) / 2, a.z + 0.05); this.rf.add(m);
    return [V3(m.position.x - w / 2, m.position.y - h / 2, m.position.z), V3(m.position.x + w / 2, m.position.y - h / 2, m.position.z), V3(m.position.x + w / 2, m.position.y + h / 2, m.position.z), V3(m.position.x - w / 2, m.position.y + h / 2, m.position.z)];
  }
  rays(corners, target, color) {
    const pts = []; for (const p of corners) pts.push(p.x, p.y, p.z, target.x, target.y, target.z);
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    this.rf.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.75 })));
  }
  buildRF() {
    disposeGroup(this.rf);
    const s = App.sel; if (!s) return;
    const acc = T.c.akzent;
    if (s.kind === 'tile') {
      const t = this.tilePos(s.L, s.c, s.r, s.q, this.vol); t.z += 0.06;
      this.frameBox(s.L, s.c, s.r, s.r, s.q, s.q, T.c.text);
      if (s.L === 1) this.rays(this.frameBox(0, 0, s.r - 1, s.r + 1, s.q - 1, s.q + 1, acc), t, acc);
      if (s.L === 3) {
        for (let ch = 0; ch < 4; ch++) this.rays(this.frameBox(2, ch, s.r - 1, s.r + 1, s.q - 1, s.q + 1, acc), t, acc);
        this.frameBox(0, 0, 2 * s.r - 3, 2 * s.r + 4, 2 * s.q - 3, 2 * s.q + 4, T.c.grad);
      }
      if (s.L === 2 || s.L === 4) this.rays(this.frameBox(s.L - 1, s.c, 2 * s.r, 2 * s.r + 1, 2 * s.q, 2 * s.q + 1, acc), t, acc);
    }
    if (s.kind === 'flat') {
      const c = Math.floor(s.k / 16), rq = s.k % 16, r = Math.floor(rq / 4), q = rq % 4;
      this.rays(this.frameBox(4, c, r, r, q, q, acc), this.flatPos(s.k), acc);
    }
  }
  onSelect() { this.rfDirty = true; this.linesDirty = true; }
  // ---------------- Animation
  startAnim(kind) {
    if (this.anim) return;
    this.playing = false;
    const ph = [], D = 0.75;
    if (kind !== 'backward') for (let i = 0; i <= 8; i++) ph.push({ type: 'fwd', idx: i, dur: i >= 1 && i <= 4 ? 1.1 : D });
    if (kind !== 'forward') for (let i = 8; i >= 0; i--) ph.push({ type: 'bwd', idx: i, dur: i === 8 ? 0.6 : 0.5 });
    if (kind === 'step') ph.push({ type: 'update', dur: 1.4 });
    let t = 0; for (const p of ph) { p.t = t; t += p.dur; }
    this.trace(kind === 'step');
    this.anim = { kind, ph, idx: -1, t0: performance.now() / 1000, end: t + 0.2 };
    this.reveal = kind === 'backward' ? { dir: -1, idx: 9 } : { dir: 1, idx: 0, frac: 0 };
    App.updateDock();
  }
  runAnim(now) {
    const a = this.anim, t = now - a.t0;
    while (a.idx + 1 < a.ph.length && t >= a.ph[a.idx + 1].t) { a.idx++; this.enterPhase(a.ph[a.idx]); }
    const p = a.ph[a.idx];
    if (p && p.type === 'fwd') {
      const u = clamp((t - p.t) / p.dur, 0, 1); this.reveal.frac = u;
      if (p.idx >= 1 && p.idx <= 4 && u < 1) { // Fenster wandert über die Eingabe (Karte 0)
        const D = CNN_LAYERS[p.idx], per = D.H * D.W, rq = Math.min(per - 1, Math.floor(u * per)), r = Math.floor(rq / D.W), q = rq % D.W;
        const src = p.idx - 1, k = p.idx === 1 || p.idx === 3 ? 3 : 2, r0 = k === 3 ? r - 1 : 2 * r, q0 = k === 3 ? q - 1 : 2 * q;
        const A = this.tilePos(src, 0, clamp(r0, 0, CNN_LAYERS[src].H - 1), clamp(q0, 0, CNN_LAYERS[src].W - 1), this.vol), B = this.tilePos(src, 0, clamp(r0 + k - 1, 0, CNN_LAYERS[src].H - 1), clamp(q0 + k - 1, 0, CNN_LAYERS[src].W - 1), this.vol);
        this.scan.visible = true; this.scan.position.set((A.x + B.x) / 2, (A.y + B.y) / 2, A.z + 0.08);
        this.scan.scale.set(Math.abs(B.x - A.x) + CNN_U, Math.abs(B.y - A.y) + CNN_U, 0.08);
      } else this.scan.visible = false;
    } else this.scan.visible = false;
    if (t >= a.end) { this.anim = null; this.reveal = null; this.scan.visible = false; App.caption(null); if (a.kind === 'step') this.trace(); App.updateDock(); App.liveDirty = true; }
  }
  enterPhase(p) {
    const tr = this.tr, c = new THREE.Color(), a = V3(), b = V3();
    if (p.type === 'fwd') {
      this.reveal = { dir: 1, idx: p.idx, frac: 0 };
      const cap = [
        '<b>Eingabe</b>: ein Graustufenbild mit 16 × 16 Pixeln, Werte zwischen 0 (hell) und 1 (dunkel).',
        '<b>Conv 1</b>: jeder der 4 Filter (3 × 3) gleitet über das Bild. An jeder Stelle: Fenster mal Filter, alles addieren, Bias dazu, dann ReLU. Derselbe Filter an jeder Stelle.',
        '<b>Max Pooling</b>: aus jedem 2 × 2-Fenster bleibt nur der größte Wert. Die Karten werden halb so breit und halb so hoch.',
        '<b>Conv 2</b>: 8 Filter der Größe 3 × 3 × 4. Jeder schaut gleichzeitig in alle 4 Karten davor und kombiniert ihre Merkmale.',
        '<b>Max Pooling</b>: wieder 2 × 2, aus 8 × 8 wird 4 × 4.',
        '<b>Flatten</b>: die 8 Karten mit je 4 × 4 Werten werden zu einem Vektor mit 128 Einträgen hintereinandergelegt.',
        '<b>Fully Connected + ReLU</b>: 16 Neuronen, jedes mit allen 128 Werten verbunden.',
        '<b>Ausgabe</b>: 4 Logits, Softmax macht daraus Wahrscheinlichkeiten für die 4 Formen.',
        `<b>Loss</b>: richtig ist „${Models.SHAPES[this.label]}“, ŷ = ${num(tr.probs[this.label])}, also L = −log ŷ = ${num(tr.loss, 3)}.`,
      ];
      App.caption(cap[p.idx]);
      if (p.idx === 6) { let n = 0; for (let j = 0; j < 16; j++) for (let k = 0; k < 128; k++) { const v = this.cnn.W3.d[j * 128 + k] * tr.flat[k]; if (Math.abs(v) > 0.08 && n < 260) { n++; Pulses.add(this.flatPos(k, a), this.fcPos(1, j, b), Math.random() * 0.1, p.dur * 0.85, valueColor(Math.sign(v), c), 0.03); } } }
      if (p.idx === 7) for (let k = 0; k < 4; k++) for (let j = 0; j < 16; j++) { const v = this.cnn.W4.d[k * 16 + j] * tr.h3[j] * (tr.mask ? tr.mask[j] : 1); if (Math.abs(v) > 0.05) Pulses.add(this.fcPos(1, j, a), this.fcPos(2, k, b), 0, p.dur * 0.85, valueColor(Math.sign(v), c), 0.035); }
      if (p.idx === 8) for (let k = 0; k < 4; k++) Pulses.add(this.fcPos(2, k, a).add(V3(2.25, 0, 0)), V3(CNN_X.loss, CNN_LY, 0), 0, p.dur * 0.85, T.cls[k], 0.06);
    } else if (p.type === 'bwd') {
      this.reveal = { dir: -1, idx: p.idx };
      const cap = {
        8: '<b>Rückwärts</b>: das Fehlersignal startet beim Loss mit ∂L/∂z<sub>k</sub> = ŷ<sub>k</sub> − y<sub>k</sub>. Lila = Betrag des Gradienten.',
        6: '<b>Rückwärts durch FC</b>: Kettenregel wie im Fully-Connected-Netz.',
        4: '<b>Rückwärts durch Max Pooling</b>: nur der Wert, der das Maximum war, bekommt einen Gradienten. Die anderen drei im Fenster bekommen 0.',
        3: '<b>Rückwärts durch Conv 2</b>: jeder Wert gibt sein Fehlersignal über die Filtergewichte an sein 3 × 3 × 4-Fenster zurück.',
        1: '<b>Rückwärts durch Conv 1</b>: hier entsteht der Gradient der Filter K₁. Ganz vorne: welche Pixel den Loss am stärksten beeinflussen.',
      };
      if (cap[p.idx]) App.caption(cap[p.idx]);
      if (p.idx === 7) for (let k = 0; k < 4; k++) Pulses.add(V3(CNN_X.loss, CNN_LY, 0), this.fcPos(2, k, a), 0, p.dur * 0.9, T.c.grad, 0.03 + 0.06 * Math.min(1, Math.abs(tr.g4[k])));
    } else if (p.type === 'update') {
      this.reveal = null;
      const r = this.cnn.trainStep(16);
      this.lossEMA = this.lossEMA === null ? r.loss : 0.9 * this.lossEMA + 0.1 * r.loss; this.hist.push(this.lossEMA);
      this.trace();
      App.caption(`<b>Update</b> (Adam): alle 2468 Parameter (Filter, Biases, FC-Gewichte) machen einen Schritt gegen ihren Gradienten. Mittelwert über ein Mini-Batch von 16 frisch erzeugten Bildern.`);
    }
  }
  toggleTrain() { if (this.anim) return; this.playing = !this.playing; this.autoStop = 0; }
  stepOnce() { this.startAnim('step'); }
  animForward() { this.startAnim('forward'); }
  animBackward() { this.startAnim('backward'); }
  reinit() { this.seedN++; DL.seed(this.seedN); this.cnn = new Models.CNN(); this.cnn.p = this.p || 0.3; this.cnn.opt.set({ lr: this.lr }); this.hist = []; this.lossEMA = null; this.acc = 0; this.anim = null; this.reveal = null; App.caption(null); this.trace(); App.renderPanel(); }
  stats() {
    return `<span class="lang">Schritt <b>${this.cnn.steps}</b> · ${this.cnn.steps * 16} Bilder gesehen · </span>Loss <b>${this.lossEMA === null ? '–' : num(this.lossEMA, 3)}</b> · <span class="lang">Genauigkeit </span><b>${this.cnn.steps ? pct(this.acc) : '–'}</b>`;
  }
  // ---------------- Inspektor
  where(s) {
    if (s.kind === 'tile') return this.tilePos(s.L, s.c, s.r, s.q, this.vol);
    if (s.kind === 'flat') return this.flatPos(s.k);
    if (s.kind === 'fc') return this.fcPos(s.l, s.j);
    if (s.kind === 'loss') return V3(CNN_X.loss, CNN_LY, 0);
    if (s.kind === 'layer') return typeof s.L === 'number' ? V3(CNN_LAYERS[s.L].x, 0, 0) : V3(CNN_X[s.L === 'out' ? 'fc2' : s.L], 0, 0);
    return null;
  }
  tip(s) {
    const tr = this.tr;
    if (s.kind === 'tile') { const D = CNN_LAYERS[s.L], i = (s.c * D.H + s.r) * D.W + s.q; return `<b>${D.short}${D.C > 1 ? ' · Karte ' + (s.c + 1) : ''}</b> · Zeile ${s.r + 1}, Spalte ${s.q + 1}<br>Wert ${num(tr.vals[s.L][i])}`; }
    if (s.kind === 'flat') return `<b>Flatten</b> Eintrag ${s.k + 1} = ${num(tr.flat[s.k])}`;
    if (s.kind === 'fc') return s.l === 1 ? `<b>FC-Neuron ${s.j + 1}</b> h = ${num(tr.h3[s.j])}` : `<b>${Models.SHAPES[s.j]}</b> ŷ = ${num(tr.probs[s.j])}`;
    if (s.kind === 'loss') return `<b>Loss</b> L = ${num(tr.loss, 3)}`;
    if (s.kind === 'layer') return '<b>Softmax</b>-Ausgabe';
    return '';
  }
  panel(s) {
    if (!s) return this.overview();
    if (s.kind === 'tile') return s.L === 0 ? this.pPixel(s) : s.L === 1 || s.L === 3 ? this.pConvTile(s) : this.pPoolTile(s);
    if (s.kind === 'layer') { if (s.L === 0) return this.pPixel(null); if (s.L === 1 || s.L === 3) return this.pConvLayer(s.L); if (s.L === 2 || s.L === 4) return this.pPoolLayer(s.L); if (s.L === 'flat') return this.pFlat(null); if (s.L === 'fc1') return this.pFC(null); return this.pOut(); }
    if (s.kind === 'flat') return this.pFlat(s.k);
    if (s.kind === 'fc') return s.l === 1 ? this.pFC(s.j) : this.pOut(s.j);
    if (s.kind === 'loss') return this.pLoss();
    return this.overview();
  }
  window(Li, c, r0, q0, k) { // k×k Werte ab (r0, q0), außerhalb = 0 (Padding)
    const D = CNN_LAYERS[Li], v = this.tr.vals[Li], out = [];
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { const r = r0 + a, q = q0 + b; out.push(r < 0 || q < 0 || r >= D.H || q >= D.W ? 0 : v[(c * D.H + r) * D.W + q]); }
    return out;
  }
  kernel(K, h, ch, S) { return Array.from(K.d.slice(((h * S + ch) * 9), ((h * S + ch) * 9) + 9)); }
  pConvTile(s) {
    const isOne = s.L === 1, D = CNN_LAYERS[s.L], S = isOne ? 1 : 4, K = isOne ? this.cnn.K1 : this.cnn.K2, bias = (isOne ? this.cnn.b1 : this.cnn.b2).d[s.c];
    const z = (isOne ? this.tr.z1 : this.tr.z2)[(s.c * D.H + s.r) * D.W + s.q], out = Math.max(0, z);
    const srcName = isOne ? 'dem Eingabebild' : 'den 4 Karten von Pool 1';
    let blocks = '', parts = [];
    for (let ch = 0; ch < S; ch++) {
      const f = this.window(s.L - 1, ch, s.r - 1, s.q - 1, 3), g = this.kernel(K, s.c, ch, S), prod = f.map((v, i) => v * g[i]), sum = prod.reduce((a, b) => a + b, 0);
      parts.push(sum);
      blocks += `${S > 1 ? `<p class="gitter-titel">Kanal ${ch + 1} von Pool 1</p>` : ''}<div class="gitter-reihe"><div><div class="gitter-titel">Fenster \\(f\\)</div>${gridHTML(f, 3, isOne ? { gray: true } : {})}</div><span class="op">⊙</span><div><div class="gitter-titel">Filter \\(g\\)</div>${gridHTML(g, 3)}</div><span class="op">=</span><div><div class="gitter-titel">Summe</div><b>${num(sum, 3)}</b></div></div>`;
    }
    const rf = isOne ? '3 × 3' : '8 × 8';
    return `${backBtn}<h2>${D.short} · Karte ${s.c + 1}<span class="chip">Zeile ${s.r + 1}, Spalte ${s.q + 1}</span></h2>
      <p class="unter">Dieser eine Wert entsteht aus einem kleinen Fenster in ${srcName} und Filter ${s.c + 1}.</p>
      <ol class="schritte">
        <li><span class="nr">1</span><div><b>Fenster mal Filter</b>: das 3 × 3-Fenster um diese Stelle${S > 1 ? ' (in jedem der 4 Kanäle)' : ''} wird Eintrag für Eintrag mit dem Filter multipliziert, dann wird alles addiert. Außerhalb des Bildes zählt 0 (Zero Padding).${blocks}</div></li>
        <li><span class="nr">2</span><div><b>Bias dazu</b>:
          <div class="tex">$$z = ${parts.map((p) => tp(p, 3)).join(' + ')} + ${tp(bias, 3)} = ${tn(z, 3)}$$</div></div></li>
        <li><span class="nr">3</span><div><b>ReLU</b>: \\(\\max(0, ${tn(z, 3)}) = ${tn(out, 3)}\\). Das ist die Höhe dieser Säule im Bild.</div></li>
      </ol>
      <p class="klein">Als Formel (Kreuzkorrelation, so rechnen PyTorch und die Übung; die echte Faltung aus der Vorlesung spiegelt \\(g\\) zusätzlich):</p>
      <div class="tex">$$z(u, v) = b + ${S > 1 ? '\\sum_{s=1}^{S} ' : ''}\\sum_{m=-1}^{1} \\sum_{n=-1}^{1} f${S > 1 ? '_s' : ''}(u + m, v + n)\\, g${S > 1 ? '_s' : ''}(m, n)$$</div>
      <div class="karte"><b>Rezeptives Feld:</b> ${isOne ? 'dieser Wert hängt nur von 3 × 3 Pixeln ab (blauer Rahmen).' : 'dieser Wert sieht 3 × 3 Werte in jeder Pool-1-Karte. Weil davor gepoolt wurde und Conv 1 auch 3 × 3 sieht, hängt er im Eingabebild von einem ' + rf + '-Bereich ab (lila Rahmen). Je tiefer die Schicht, desto größer das Feld.'}</div>
      <p class="klein">Lokale Verbindungen und geteilte Gewichte (L4): derselbe Filter wird an allen ${D.H} · ${D.W} = ${D.H * D.W} Stellen benutzt.</p>`;
  }
  pPoolTile(s) {
    const D = CNN_LAYERS[s.L], src = s.L - 1, f = this.window(src, s.c, 2 * s.r, 2 * s.q, 2), arg = (s.L === 2 ? this.tr.arg1 : this.tr.arg2)[(s.c * D.H + s.r) * D.W + s.q];
    const S = CNN_LAYERS[src], ar = Math.floor((arg % (S.H * S.W)) / S.W) - 2 * s.r, aq = (arg % S.W) - 2 * s.q, mi = ar * 2 + aq;
    return `${backBtn}<h2>${D.short} · Karte ${s.c + 1}<span class="chip">Zeile ${s.r + 1}, Spalte ${s.q + 1}</span></h2>
      <p class="unter">Max Pooling 2 × 2 mit Stride 2: aus dem Fenster in ${S.short}, Karte ${s.c + 1}, bleibt nur der größte Wert.</p>
      <div class="gitter-reihe"><div><div class="gitter-titel">Fenster</div>${gridHTML(f, 2, { mark: mi })}</div><span class="op">→ max →</span><b>${num(Math.max(...f), 3)}</b></div>
      <ol class="schritte"><li><span class="nr">1</span><div><b>Vorwärts</b>: der umrandete Wert ist das Maximum und wird weitergegeben.</div></li>
      <li><span class="nr">2</span><div><b>Rückwärts</b>: nur dieser eine Wert bekommt das Fehlersignal, die anderen drei bekommen \\(0\\) (L4: „only one value contributes to error“).</div></li></ol>
      <p class="klein">Pooling hat keine Parameter. Es macht das Netz unempfindlicher gegen kleine Verschiebungen und spart Rechenzeit.</p>`;
  }
  pPixel(s) {
    return `${backBtn}<h2>Eingabebild<span class="chip">16 × 16 × 1</span></h2>
      <p class="unter">\\(X = Y = 16\\) Pixel, \\(S = 1\\) Kanal (Graustufen). Jeder Pixel ist eine Zahl zwischen 0 und 1.</p>
      ${s ? `<p>Pixel Zeile ${s.r + 1}, Spalte ${s.q + 1}: \\(f(${s.r + 1}, ${s.q + 1}) = ${tn(this.tr.vals[0][s.r * 16 + s.q], 2)}\\).</p>` : ''}
      ${this.drawUI()}`;
  }
  drawUI() {
    return `<h3>Eingabe zeichnen</h3>
      <canvas class="malflaeche" id="malen" width="16" height="16"></canvas>
      <div class="knopfgruppe"><div class="segment klein"><button type="button" data-act="tool" data-t="mal" aria-pressed="${this.tool === 'mal'}">Stift</button><button type="button" data-act="tool" data-t="weg" aria-pressed="${this.tool === 'weg'}">Radierer</button></div><button type="button" class="knopf zart" data-act="leeren">Leeren</button></div>
      <p class="klein">Zufälliges Beispiel:</p>
      <div class="knopfgruppe">${Models.SHAPES.map((n, k) => `<button type="button" class="knopf zart" data-act="bsp" data-k="${k}">${n}</button>`).join('')}</div>
      <p class="klein">Richtige Klasse (für Loss und „Rückwärts“):</p>
      <div class="segment klein" style="flex-wrap:wrap">${Models.SHAPES.map((n, k) => `<button type="button" data-act="label" data-k="${k}" aria-pressed="${this.label === k}">${n}</button>`).join('')}</div>
      <h3>Vorhersage</h3><div id="dyn-pred">${balkenHTML(Models.SHAPES, this.tr.probs, T.cls.map((c) => c.getStyle()), this.label)}</div>`;
  }
  pConvLayer(Li) {
    const one = Li === 1, D = CNN_LAYERS[Li], S = one ? 1 : 4, H = D.C, K = one ? this.cnn.K1 : this.cnn.K2, n = 3 * 3 * S * H + H;
    let filt = '';
    for (let h = 0; h < H; h++) { filt += `<div><div class="gitter-titel">Filter ${h + 1}</div><div style="display:flex;gap:4px">`; for (let ch = 0; ch < S; ch++) filt += gridHTML(this.kernel(K, h, ch, S), 3, { d: 1 }); filt += '</div></div>'; }
    return `${backBtn}<h2>${D.name}</h2>
      <p class="unter">\\(H = ${H}\\) Filter der Größe \\(M \\times N \\times S = 3 \\times 3 \\times ${S}\\), Zero Padding „same“, Stride 1. Ausgabe: \\(${D.H} \\times ${D.W} \\times ${H}\\).</p>
      <ol class="schritte">
        <li><span class="nr">1</span><div><b>Lokale Verbindungen</b>: jeder Ausgabewert sieht nur ein 3 × 3-Fenster${S > 1 ? ', aber über alle ' + S + ' Kanäle (voll verbunden in der Tiefe)' : ''}.</div></li>
        <li><span class="nr">2</span><div><b>Geteilte Gewichte</b>: ein Filter wird an jeder Stelle benutzt. Darum findet er dasselbe Muster überall im Bild (Translations-Äquivarianz).</div></li>
        <li><span class="nr">3</span><div><b>Eine Karte pro Filter</b>: ${H} Filter ergeben ${H} Merkmalskarten (Feature Maps).</div></li>
      </ol>
      <h3>Gelernte Filter</h3><div style="display:flex;flex-wrap:wrap;gap:10px">${filt}</div>
      <p class="klein">Nach dem Training sehen die Filter von Conv 1 oft aus wie Kanten-Detektoren (vgl. Sobel-Filter in L4): positive Zahlen auf einer Seite, negative auf der anderen.</p>
      <h3>Parameter</h3>
      <div class="tex">$$H \\cdot (M \\cdot N \\cdot S + 1) = ${H} \\cdot (3 \\cdot 3 \\cdot ${S} + 1) = ${n}$$</div>
      ${one ? '<div class="karte hinweis">Zum Vergleich: ein Fully-Connected-Layer von 256 Pixeln auf dieselben \\(16 \\cdot 16 \\cdot 4 = 1024\\) Ausgabewerte bräuchte \\(1024 \\cdot 257 = 263.168\\) Parameter.</div>' : ''}
      <h3>Padding und Stride</h3>
      <p>Ohne Padding wird die Karte um \\(2 \\cdot \\lfloor n/2 \\rfloor = 2\\) Pixel kleiner (\\(n = 3\\): Filtergröße), also \\(${D.H} \\to ${D.H - 2}\\). Mit „same“ Padding (Rand mit Nullen) bleibt sie \\(${D.H} \\times ${D.W}\\). Mit Stride \\(s = 2\\) würde nur jede zweite Stelle berechnet: \\(${D.H / 2} \\times ${D.W / 2}\\).</p>`;
  }
  pPoolLayer(Li) {
    const D = CNN_LAYERS[Li], S = CNN_LAYERS[Li - 1];
    return `${backBtn}<h2>${D.name}</h2>
      <p class="unter">2 × 2-Fenster, Stride 2: \\(${S.H} \\times ${S.W} \\times ${S.C} \\to ${D.H} \\times ${D.W} \\times ${D.C}\\). Jede Karte wird für sich gepoolt.</p>
      <ol class="schritte"><li><span class="nr">1</span><div><b>Vorwärts</b>: aus jedem Fenster geht nur das Maximum weiter.</div></li>
      <li><span class="nr">2</span><div><b>Rückwärts</b>: der Gradient geht nur an die Stelle, die das Maximum war.</div></li>
      <li><span class="nr">3</span><div><b>Wozu?</b> Weniger Werte (Rechenzeit, weniger Parameter danach), etwas Unempfindlichkeit gegen kleine Verschiebungen, größeres rezeptives Feld für die nächste Schicht.</div></li></ol>
      <p class="klein">Keine Parameter. Tipp auf eine Säule zeigt ihr Fenster.</p>`;
  }
  pFlat(k) {
    let extra = '';
    if (k !== null) { const c = Math.floor(k / 16), rq = k % 16; extra = `<p>Eintrag ${k + 1} kommt aus Pool 2, Karte ${c + 1}, Zeile ${Math.floor(rq / 4) + 1}, Spalte ${rq % 4 + 1}: \\(${tn(this.tr.flat[k], 3)}\\).</p>`; }
    return `${backBtn}<h2>Flatten<span class="chip">4 × 4 × 8 → 128</span></h2>
      <p class="unter">Flatten rechnet nichts. Es legt die 8 Karten mit je 4 × 4 Werten hintereinander, damit eine Fully-Connected-Schicht folgen kann.</p>${extra}
      <p>Ab hier geht die räumliche Anordnung verloren: die FC-Schicht weiß nicht mehr, welche Werte nebeneinanderlagen. Darum macht man vorher Faltungen, die lokale Muster finden.</p>`;
  }
  pFC(j) {
    const tr = this.tr, p = this.cnn.p;
    let top = '';
    if (j !== null) {
      const W = this.cnn.W3.d, contrib = Array.from({ length: 128 }, (_, k) => [k, W[j * 128 + k] * tr.flat[k]]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 5);
      const off = tr.mask && tr.mask[j] === 0;
      top = `<div class="tex">$$z_{${j + 1}} = w_{0,${j + 1}} + \\sum_{i=1}^{128} w_{i,${j + 1}}\\, h_i = ${tn(tr.z3[j], 3)}, \\qquad h_{${j + 1}} = \\text{ReLU}(z_{${j + 1}}) = ${tn(tr.h3[j], 3)}$$</div>
        ${off ? '<div class="karte hinweis">Dieses Neuron ist in diesem Trainingsschritt durch Dropout aus.</div>' : ''}
        <h3>Die 5 größten Beiträge \\(w_{i,${j + 1}} h_i\\)</h3><div class="tabelle-wrap"><table class="tabelle"><tr><th>aus Flatten</th><th class="r">\\(w\\)</th><th class="r">\\(h_i\\)</th><th class="r">Produkt</th></tr>${contrib.map(([k, v]) => `<tr ${selAttr({ kind: 'flat', k })}><td>Eintrag ${k + 1}</td><td class="r">${num(W[j * 128 + k], 3)}</td><td class="r">${num(tr.flat[k], 3)}</td><td class="r">${num(v, 3)}</td></tr>`).join('')}</table></div>`;
    }
    return `${backBtn}<h2>${j === null ? 'Fully Connected + ReLU' : 'FC-Neuron ' + (j + 1)}<span class="chip">16 Neuronen</span></h2>
      <p class="unter">Wie im Fully-Connected-Netz: jedes Neuron sieht alle 128 Werte. \\(\\mathbf{W}_3 \\in \\mathbb{R}^{16 \\times 128}\\), also \\(16 \\cdot 129 = 2064\\) Parameter, fast so viele wie der ganze Rest.</p>${top}
      <h3>Dropout</h3>
      <p>Beim Training wird jede Ausgabe dieser Schicht mit Wahrscheinlichkeit \\(p\\) auf \\(0\\) gesetzt, die anderen durch \\(1 - p\\) geteilt (L3). „Ein Schritt“ zeigt das: ausgeschaltete Neuronen werden klein und grau. Beim Testen ist Dropout aus.</p>
      ${auswahl('p', 'Dropout \\(p\\)', [[0, 'aus'], [0.3, '0,3'], [0.5, '0,5']], p)}`;
  }
  pOut(k) {
    const tr = this.tr, y = this.label;
    return `${backBtn}<h2>Ausgabe mit Softmax${k !== undefined ? `<span class="chip">${Models.SHAPES[k]}</span>` : ''}</h2>
      <p class="unter">4 Logits \\(z_k\\) aus \\(\\mathbf{W}_4 \\in \\mathbb{R}^{4 \\times 16}\\) (\\(4 \\cdot 17 = 68\\) Parameter), dann Softmax:</p>
      <div class="tex">$$\\hat{y}_k = \\frac{e^{z_k}}{\\sum_{j=1}^{4} e^{z_j}}$$</div>
      <div class="tabelle-wrap"><table class="tabelle"><tr><th>Label</th><th class="r">\\(z_k\\)</th><th class="r">\\(e^{z_k}\\)</th><th class="r">\\(\\hat{y}_k\\)</th></tr>${Models.SHAPES.map((n, q) => `<tr class="${q === y ? 'aktiv' : ''}"><td>${n}${q === y ? ' (richtig)' : ''}</td><td class="r">${num(tr.z4[q])}</td><td class="r">${num(Math.exp(tr.z4[q]))}</td><td class="r">${num(tr.probs[q], 3)}</td></tr>`).join('')}</table></div>
      ${k !== undefined ? `<div class="tex">$$\\frac{\\partial L}{\\partial z_{${k + 1}}} = \\hat{y}_{${k + 1}} - y_{${k + 1}} = ${tn(tr.probs[k], 3)} - ${k === y ? 1 : 0} = ${tn(tr.g4[k], 3)}$$</div>` : ''}`;
  }
  pLoss() {
    const tr = this.tr, y = this.label;
    return `${backBtn}<h2>Loss: Cross-Entropy</h2>
      <div class="tex">$$L = -\\log \\hat{y}_k \\Big|_{y_k = 1} = -\\log(${tn(tr.probs[y], 3)}) = ${tn(tr.loss, 3)}$$</div>
      <p>Richtige Klasse: „${Models.SHAPES[y]}“. Je näher \\(\\hat{y}\\) dieser Klasse an \\(1\\) liegt, desto näher ist \\(L\\) an \\(0\\).</p>
      <p class="klein">Beim Training ist der Loss der Mittelwert über ein Mini-Batch von 16 Bildern. Die Kurve unten zeigt einen gleitenden Mittelwert.</p>`;
  }
  overview() {
    const tr = this.tr;
    const rows = [['Eingabe', '16 × 16 × 1', 0], ['Conv 1: 4 Filter 3 × 3 × 1 + ReLU', '16 × 16 × 4', 40], ['Max Pooling 2 × 2', '8 × 8 × 4', 0], ['Conv 2: 8 Filter 3 × 3 × 4 + ReLU', '8 × 8 × 8', 296], ['Max Pooling 2 × 2', '4 × 4 × 8', 0], ['Flatten', '128', 0], ['FC + ReLU + Dropout', '16', 2064], ['FC + Softmax', '4', 68]];
    const keys = [0, 1, 2, 3, 4, 'flat', 'fc1', 'out'];
    return `<h2>Faltendes Netz (CNN)</h2>
      <p class="unter">Erkennt vier Formen in 16 × 16-Bildern: ${Models.SHAPES.join(', ')}. Die Trainingsbilder werden zufällig erzeugt, das Netz lernt hier im Browser.</p>
      <div class="legende">
        <div><span class="quadrat" style="background:var(--tinte)"></span><span>Eingabe: dunkel = Pixelwert nahe 1.</span></div>
        <div><span class="quadrat" style="background:var(--pos)"></span><span>Säule = ein Wert einer Merkmalskarte nach ReLU. Höhe und Farbe = wie stark der Filter hier anspricht.</span></div>
        <div><span class="quadrat" style="background:var(--grad)"></span><span>Lila (nach „Rückwärts“) = Betrag des Gradienten \\(\\partial L / \\partial a\\).</span></div>
      </div>
      <p class="klein">Tipp auf eine Säule: du siehst ihr Fenster, den Filter und die Rechnung. Tipp auf eine Beschriftung: Erklärung der Schicht.</p>
      ${this.drawUI()}
      <h3>Aufbau</h3>
      <div class="tabelle-wrap"><table class="tabelle"><tr><th>Schicht</th><th class="r">Ausgabe</th><th class="r">Parameter</th></tr>${rows.map(([a, b, c], i) => `<tr ${selAttr({ kind: 'layer', L: keys[i] })}><td>${a}</td><td class="r">${b}</td><td class="r">${c.toLocaleString('de-DE')}</td></tr>`).join('')}<tr><td><b>Summe</b></td><td></td><td class="r"><b>2.468</b></td></tr></table></div>
      <h3>Ansicht</h3>
      <div class="segment klein"><button type="button" data-act="vol" data-v="0" aria-pressed="${this.volTarget === 0}">Karten nebeneinander</button><button type="button" data-act="vol" data-v="1" aria-pressed="${this.volTarget === 1}">Als Tensor (Volumen)</button></div>
      <p class="klein">„Als Tensor“ stapelt die Karten einer Schicht in die Tiefe: so sieht \\(X \\times Y \\times S\\) aus. Ein Filter von Conv 2 ist dann ein kleiner Quader \\(3 \\times 3 \\times 4\\), der durch alle Karten von Pool 1 reicht.</p>
      ${schalter('relief', 'Höhe = Aktivierung', this.relief)}${schalter('grad', 'Gradienten statt Aktivierungen', this.gradView)}
      <h3>Training</h3>
      ${auswahl('lr', 'Lernrate \\(\\eta\\) (Adam)', [[0.001, '0,001'], [0.004, '0,004'], [0.01, '0,01'], [0.05, '0,05']], this.lr)}
      ${auswahl('p', 'Dropout \\(p\\)', [[0, 'aus'], [0.3, '0,3'], [0.5, '0,5']], this.cnn.p)}`;
  }
  afterPanel(el) {
    const cv = $('#malen', el); if (!cv) return;
    this.drawCanvas(cv);
    const paint = (e) => {
      const r = cv.getBoundingClientRect(), q = Math.floor((e.clientX - r.left) / r.width * 16), row = Math.floor((e.clientY - r.top) / r.height * 16);
      if (q < 0 || q > 15 || row < 0 || row > 15) return;
      const set = (a, b, v) => { if (a < 0 || b < 0 || a > 15 || b > 15) return; const i = a * 16 + b; this.img[i] = this.tool === 'mal' ? Math.max(this.img[i], v) : 0; };
      set(row, q, 1); for (const [da, db] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) set(row + da, q + db, this.tool === 'mal' ? 0.35 : 0);
      this.afterDraw(cv);
    };
    let on = false;
    cv.addEventListener('pointerdown', (e) => { on = true; cv.setPointerCapture(e.pointerId); paint(e); });
    cv.addEventListener('pointermove', (e) => { if (on) paint(e); });
    const end = () => { if (on) { on = false; if (!this.userLabel) { this.label = this.argmax(); } App.renderPanel(); } };
    cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
  }
  argmax() { let b = 0; for (let k = 1; k < 4; k++) if (this.tr.probs[k] > this.tr.probs[b]) b = k; return b; }
  afterDraw(cv) {
    this.trace(); this.drawCanvas(cv);
    const d = $('#dyn-pred'); if (d) d.innerHTML = balkenHTML(Models.SHAPES, this.tr.probs, T.cls.map((c) => c.getStyle()), this.label);
  }
  drawCanvas(cv) {
    const g = cv.getContext('2d'), img = g.createImageData(16, 16), c = new THREE.Color();
    for (let i = 0; i < 256; i++) { c.copy(T.c.flaeche).lerp(T.c.tinte, Math.max(0, Math.min(1, this.img[i]))); img.data[4 * i] = c.r * 255; img.data[4 * i + 1] = c.g * 255; img.data[4 * i + 2] = c.b * 255; img.data[4 * i + 3] = 255; }
    g.putImageData(img, 0, 0);
  }
  live(el, s) {
    const d = $('#dyn-pred', el); if (d) d.innerHTML = balkenHTML(Models.SHAPES, this.tr.probs, T.cls.map((c) => c.getStyle()), this.label);
    if (!s || s.kind === 'layer' && (s.L === 'fc1' || s.L === 0)) return;
    const now = performance.now(); if (this.playing && now - this.lastLive < 1500) { App.liveDirty = true; return; }
    this.lastLive = now; App.renderPanel(false);
  }
  act(name, el) {
    if (name === 'tool') { this.tool = el.dataset.t; App.renderPanel(); return; }
    if (name === 'leeren') { this.img = new Float64Array(256); this.userLabel = false; this.trace(); App.renderPanel(); return; }
    if (name === 'bsp') { this.newExample(+el.dataset.k); this.trace(); App.renderPanel(); return; }
    if (name === 'label') { this.label = +el.dataset.k; this.userLabel = true; this.trace(); App.renderPanel(); return; }
    if (name === 'vol') { this.volTarget = +el.dataset.v; App.renderPanel(); return; }
  }
  setOpt(key, val, el, type) {
    if (type === 'input' && el.tagName === 'SELECT') return;
    if (key === 'relief') this.relief = !!val;
    if (key === 'grad') this.gradView = !!val;
    if (key === 'lr') { this.lr = +val; this.cnn.opt.set({ lr: this.lr }); }
    if (key === 'p') { this.p = +val; this.cnn.p = +val; }
  }
  themeChanged() { this.scan.material.color.copy(T.c.akzent); this.ring.material.color.copy(T.c.akzent); this.rfDirty = true; this.linesDirty = true; }
}
App.modes.cnn = new CNNMode();
