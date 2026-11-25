/* ============================================================
   Kern der Oberfläche: Szene, Kamera, Auswahl, Inspektor, Animationen
   ============================================================ */
const $ = (s, r = document) => r.querySelector(s);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const V3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// Zahlen auf Deutsch: 0,42 und echtes Minuszeichen
function num(v, d = 2) {
  if (!isFinite(v)) return '–';
  const s = Math.abs(v).toFixed(d).replace('.', ',');
  return (v < 0 && /[1-9]/.test(s) ? '−' : '') + s;
}
function tn(v, d = 2) { // für LaTeX
  if (!isFinite(v)) return '\\infty';
  const s = Math.abs(v).toFixed(d).replace('.', '{,}');
  return (v < 0 && /[1-9]/.test(s) ? '-' : '') + s;
}
const tp = (v, d = 2) => { const s = tn(v, d); return s[0] === '-' ? `(${s})` : s; };
const pct = (v) => Math.round(v * 100) + ' %';
function sci(v) { // kleine Gradienten lesbar
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 0.01) return num(v, 3);
  const e = Math.floor(Math.log10(a)), m = v / Math.pow(10, e);
  return `${num(m, 1)}·10<sup>${e}</sup>`;
}
function tsci(v) {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 0.01) return tn(v, 3);
  const e = Math.floor(Math.log10(a)), m = v / Math.pow(10, e);
  return `${tn(m, 1)} \\cdot 10^{${e}}`;
}

// ---------- Farben aus den CSS-Tokens (hell/dunkel)
const T = { c: {} };
function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ['pos', 'neg', 'null', 'grad', 'tinte', 'c0', 'c1', 'c2', 'c3', 'text', 'text-2', 'text-3', 'flaeche', 'flaeche-2', 'flaeche-3', 'buehne-1', 'buehne-2', 'akzent', 'gut']) {
    const v = cs.getPropertyValue('--' + k).trim() || '#888888';
    T[k] = v; T.c[k] = new THREE.Color(v);
  }
  T.cls = [T.c.c0, T.c.c1, T.c.c2, T.c.c3];
  T.c.bg = T.c['buehne-1'].clone().lerp(T.c['buehne-2'], 0.45);
  T.dark = T.c['buehne-2'].r + T.c['buehne-2'].g + T.c['buehne-2'].b < 1.2;
  T.version = (T.version || 0) + 1;
}
const _c = new THREE.Color();
function valueColor(v, out = _c) {
  if (v >= 0) out.copy(T.c.null).lerp(T.c.pos, Math.min(1, v));
  else out.copy(T.c.null).lerp(T.c.neg, Math.min(1, -v));
  return out;
}
const cssRGB = (c) => `rgb(${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)})`;
function cssValue(v) { return cssRGB(valueColor(v, new THREE.Color())); }

// ---------- MathJax (SVG, keine Schriftdateien nötig)
var mjReady = window.mjReady || false; let mjChain = Promise.resolve();
function typeset(el) {
  if (!mjReady || !el) return;
  mjChain = mjChain.then(() => window.MathJax.typesetPromise([el])).catch((e) => console.warn(e));
}
function untypeset(el) { if (mjReady && window.MathJax.typesetClear) try { window.MathJax.typesetClear([el]); } catch (e) { /* egal */ } }
function setTex(el, html) { if (!el) return; untypeset(el); el.innerHTML = html; typeset(el); }

// ---------- Szene
const App = {
  modes: {}, mode: null, sel: null, live: 0, liveDirty: false,
};
const stage = $('#buehne');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
stage.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 500);
camera.position.set(4, 5, 18);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.09; controls.screenSpacePanning = true;
controls.minDistance = 1.5; controls.maxDistance = 90; controls.rotateSpeed = 0.8;
const labelRenderer = new THREE.CSS2DRenderer({ element: $('#etiketten') });
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a90a0, 0.95));
const sun = new THREE.DirectionalLight(0xffffff, 0.75); sun.position.set(6, 14, 10); scene.add(sun);
const sun2 = new THREE.DirectionalLight(0xffffff, 0.25); sun2.position.set(-8, 4, -6); scene.add(sun2);

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  labelRenderer.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix();
  if (App.mode && !App.userMoved && !fly) App.home(true);
}
new ResizeObserver(resize).observe(stage);
controls.addEventListener('start', () => { App.userMoved = true; });

// Kamera so setzen, dass ein Quader ganz zu sehen ist (Ecken in Kamerakoordinaten projiziert)
function viewFor(min, max, dir = V3(0.16, 0.3, 1)) {
  const c = min.clone().add(max).multiplyScalar(0.5), f = dir.clone().normalize();
  const right = V3(0, 1, 0).cross(f).normalize(), up = f.clone().cross(right).normalize();
  const tv = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2), th = tv * camera.aspect;
  let d = 4;
  for (let i = 0; i < 8; i++) {
    const o = V3(i & 1 ? max.x : min.x, i & 2 ? max.y : min.y, i & 4 ? max.z : min.z).sub(c);
    d = Math.max(d, o.dot(f) + Math.max(Math.abs(o.dot(right)) / th, Math.abs(o.dot(up)) / tv));
  }
  return { pos: c.clone().add(f.multiplyScalar(d * 1.06)), target: c };
}
let fly = null;
function flyTo(pos, target, dur = 0.9) {
  fly = { p0: camera.position.clone(), t0: controls.target.clone(), p1: pos.clone(), t1: target.clone(), s: performance.now(), dur: dur * 1000 };
}
function flyUpdate(now) {
  if (!fly) return;
  const u = clamp((now - fly.s) / fly.dur, 0, 1), e = ease(u);
  camera.position.lerpVectors(fly.p0, fly.p1, e); controls.target.lerpVectors(fly.t0, fly.t1, e);
  if (u >= 1) fly = null;
}
function focusOn(point, dist = 6) {
  const dir = camera.position.clone().sub(controls.target).normalize();
  flyTo(point.clone().add(dir.multiplyScalar(dist)), point);
}
App.phone = () => window.matchMedia('(max-width: 860px)').matches;
App.home = (instant) => {
  const v = App.mode.view();
  if (instant) { camera.position.copy(v.pos); controls.target.copy(v.target); fly = null; }
  else flyTo(v.pos, v.target);
};

// ---------- Hilfen für 3D-Objekte
function label(html, cls = 'lbl', onClick) {
  const el = document.createElement('div');
  el.className = cls; el.innerHTML = html;
  if (onClick) el.addEventListener('click', (e) => { e.stopPropagation(); onClick(e); });
  return new THREE.CSS2DObject(el);
}
function disposeGroup(g) {
  g.traverse((o) => {
    if (o.isCSS2DObject && o.element.parentNode) o.element.parentNode.removeChild(o.element);
    if (o.geometry) o.geometry.dispose();
    if (o.material) { const ms = Array.isArray(o.material) ? o.material : [o.material]; for (const m of ms) { if (m.map) m.map.dispose(); m.dispose(); } }
  });
  g.clear();
}
function hideLabels(g, hide) { g.traverse((o) => { if (o.isCSS2DObject) o.element.style.display = hide ? 'none' : ''; }); }
const _m4 = new THREE.Matrix4(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _p = new THREE.Vector3();
const UP = V3(0, 1, 0);
function instanced(geo, mat, n) {
  const m = new THREE.InstancedMesh(geo, mat, Math.max(1, n));
  m.count = n; m.frustumCulled = false;
  const c = new THREE.Color(0.5, 0.5, 0.5);
  for (let i = 0; i < Math.max(1, n); i++) { m.setColorAt(i, c); m.setMatrixAt(i, _m4.identity()); }
  return m;
}
function setInst(mesh, i, pos, sx, sy, sz, quat) {
  _s.set(sx, sy === undefined ? sx : sy, sz === undefined ? sx : sz);
  _m4.compose(pos, quat || _q.identity(), _s);
  mesh.setMatrixAt(i, _m4);
}
// Stab zwischen zwei Punkten (für Gewichte)
function stick(a, b) {
  const dir = b.clone().sub(a), len = dir.length();
  return { mid: a.clone().add(b).multiplyScalar(0.5), len, quat: new THREE.Quaternion().setFromUnitVectors(UP, dir.normalize()) };
}
function shadowDisc(w, d, y) {
  const cv = document.createElement('canvas'); cv.width = cv.height = 128;
  const g = cv.getContext('2d'), gr = g.createRadialGradient(64, 64, 4, 64, 64, 64);
  gr.addColorStop(0, 'rgba(0,0,0,0.20)'); gr.addColorStop(0.6, 'rgba(0,0,0,0.07)'); gr.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(cv);
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
  m.rotation.x = -Math.PI / 2; m.position.y = y; m.renderOrder = -1;
  return m;
}

// ---------- Leuchtpunkte, die über Verbindungen wandern
const Pulses = {
  max: 2400, items: [],
  init() {
    this.mesh = instanced(new THREE.SphereGeometry(1, 10, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }), this.max);
    this.mesh.count = 0; scene.add(this.mesh);
  },
  add(a, b, delay, dur, color, size = 0.07) {
    if (this.items.length >= this.max) return;
    this.items.push({ a: a.clone(), b: b.clone(), t0: performance.now() / 1000 + delay, dur, color: color.clone(), size });
  },
  clear() { this.items = []; this.mesh.count = 0; },
  update(now) {
    let k = 0; const keep = [];
    for (const it of this.items) {
      const u = (now - it.t0) / it.dur;
      if (u > 1) continue;
      keep.push(it);
      if (u < 0) continue;
      const e = ease(u);
      _p.lerpVectors(it.a, it.b, e);
      setInst(this.mesh, k, _p, it.size * (0.6 + 0.6 * Math.sin(Math.PI * u)));
      this.mesh.setColorAt(k, it.color); k++;
    }
    this.items = keep; this.mesh.count = k;
    this.mesh.instanceMatrix.needsUpdate = true; if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  },
};

// ---------- Auswahl mit Maus/Finger
const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
function pickAt(cx, cy) {
  const r = renderer.domElement.getBoundingClientRect();
  ptr.set(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ptr, camera);
  const hits = ray.intersectObjects(App.mode.pickables || [], false);
  for (const h of hits) { const f = h.object.userData.pick; if (f) { const s = f(h); if (s) return s; } }
  return null;
}
const canvas = renderer.domElement, tip = $('#tipp');
let down = null, hoverT = 0;
canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; tip.hidden = true; });
canvas.addEventListener('pointerup', (e) => {
  if (!down) return;
  if (Math.hypot(e.clientX - down.x, e.clientY - down.y) < 7) App.select(pickAt(e.clientX, e.clientY));
  down = null;
});
canvas.addEventListener('dblclick', (e) => { const s = pickAt(e.clientX, e.clientY); if (s && App.mode.where) { const p = App.mode.where(s); if (p) focusOn(p, 5); } });
canvas.addEventListener('pointermove', (e) => {
  if (e.pointerType !== 'mouse' || e.buttons) { tip.hidden = true; return; }
  const now = performance.now(); if (now - hoverT < 40) return; hoverT = now;
  const s = pickAt(e.clientX, e.clientY);
  const html = s && App.mode.tip ? App.mode.tip(s) : '';
  canvas.style.cursor = s ? 'pointer' : 'grab';
  if (!html) { tip.hidden = true; return; }
  const r = stage.getBoundingClientRect();
  tip.innerHTML = html; tip.hidden = false;
  const x = Math.min(e.clientX - r.left + 14, r.width - tip.offsetWidth - 8), y = Math.min(e.clientY - r.top + 14, r.height - tip.offsetHeight - 8);
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
});
canvas.addEventListener('pointerleave', () => { tip.hidden = true; });

App.select = (s) => {
  App.sel = s;
  App.mode.onSelect && App.mode.onSelect(s);
  App.renderPanel(true);
  if (s && window.matchMedia('(max-width: 860px)').matches) $('#panel').classList.remove('zu');
};

// ---------- Inspektor
const inhalt = $('#inhalt');
App.renderPanel = (toTop) => {
  untypeset(inhalt);
  inhalt.innerHTML = App.mode.panel(App.sel);
  App.mode.afterPanel && App.mode.afterPanel(inhalt, App.sel);
  typeset(inhalt);
  if (toTop) $('#panel').scrollTop = 0;
};
inhalt.addEventListener('click', (e) => {
  const a = e.target.closest('[data-act]');
  if (a) { e.preventDefault(); App.mode.act(a.dataset.act, a, e); return; }
  const s = e.target.closest('[data-sel]');
  if (s) App.select(JSON.parse(s.dataset.sel));
});
const onSet = (e) => { const t = e.target; if (t.dataset && t.dataset.set) App.mode.setOpt(t.dataset.set, t.type === 'checkbox' ? t.checked : t.value, t, e.type); };
inhalt.addEventListener('input', onSet);
inhalt.addEventListener('change', onSet);
$('#griff').addEventListener('click', () => $('#panel').classList.toggle('zu'));
const backBtn = '<button type="button" class="zurueck" data-sel="null">‹ Überblick</button>';
const selAttr = (o) => `data-sel='${JSON.stringify(o)}'`;

// ---------- Ablauf-Hinweis oben
const ablauf = $('#ablauf');
App.caption = (html) => { if (!html) { ablauf.hidden = true; return; } ablauf.innerHTML = html; ablauf.hidden = false; };

// ---------- Steuerleiste
const ICON_PLAY = '<svg viewBox="0 0 16 16"><path d="M4 2.5v11l9.5-5.5z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 16 16"><path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>';
App.updateDock = () => {
  const m = App.mode, st = m.stats();
  $('#b-train').innerHTML = m.playing ? `${ICON_PAUSE}<span>Pause</span>` : `${ICON_PLAY}<span>Trainieren</span>`;
  $('#stand-text').innerHTML = st;
  for (const id of ['b-schritt', 'b-vor', 'b-rueck']) $('#' + id).disabled = !!m.anim;
  drawSpark(m.hist || []);
};
function drawSpark(h) {
  const cv = $('#spark'), g = cv.getContext('2d'), W = cv.width, H = cv.height;
  g.clearRect(0, 0, W, H);
  if (h.length < 2) return;
  const mx = Math.max(...h, 1e-6);
  g.strokeStyle = T.pos; g.lineWidth = 3; g.lineJoin = 'round'; g.beginPath();
  h.forEach((v, i) => { const x = 4 + (i / (h.length - 1)) * (W - 8), y = H - 4 - (v / mx) * (H - 8); i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.stroke();
}
$('#b-train').addEventListener('click', () => { App.mode.toggleTrain(); App.updateDock(); });
$('#b-schritt').addEventListener('click', () => App.mode.stepOnce());
$('#b-vor').addEventListener('click', () => App.mode.animForward());
$('#b-rueck').addEventListener('click', () => App.mode.animBackward());
$('#b-neu').addEventListener('click', () => { App.mode.reinit(); App.updateDock(); });
$('#b-heim').addEventListener('click', () => { App.userMoved = false; App.home(); });

// ---------- Moduswechsel
App.setMode = (key) => {
  if (App.mode && App.mode.key === key) return;
  if (App.mode) { App.mode.playing = false; App.mode.anim = null; scene.remove(App.mode.group); hideLabels(App.mode.group, true); }
  Pulses.clear(); App.caption(null); tip.hidden = true;
  const m = App.modes[key];
  App.mode = m;
  if (!m.built) { m.init(); m.built = true; }
  scene.add(m.group); hideLabels(m.group, false);
  m.enter && m.enter();
  App.sel = null; App.userMoved = false;
  for (const b of document.querySelectorAll('#modi button')) b.setAttribute('aria-pressed', String(b.dataset.modus === key));
  App.home(true);
  App.renderPanel(true); App.updateDock();
  try { if (location.hash !== '#' + key) history.replaceState(null, '', '#' + key); } catch (e) { /* egal */ }
};
for (const b of document.querySelectorAll('#modi button')) b.addEventListener('click', () => App.setMode(b.dataset.modus));

// ---------- Kleine Zeichenhilfen für Grafiken im Inspektor
function prepCanvas(cv, w, h) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = w * dpr; cv.height = h * dpr; cv.style.aspectRatio = `${w} / ${h}`;
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  return g;
}
const ACTRANGE = { heaviside: [-1.4, 1.4], sign: [-1.4, 1.4], sigmoid: [-0.4, 1.3], tanh: [-1.4, 1.4], linear: [-3.2, 3.2], relu: [-1.2, 3.4], leaky: [-1.2, 3.4], swish: [-1.2, 3.4], gelu: [-1.2, 3.4] };
// Aktivierungsfunktion f (durchgezogen) und Ableitung f' (gestrichelt), Punkt bei z
function plotAct(cv, key, z) {
  const W = 320, H = 190, g = prepCanvas(cv, W, H), A = DL.ACT[key];
  const [y0, y1] = ACTRANGE[key] || [-1.5, 3], x0 = -4, x1 = 4, L = 30, R = 10, Tp = 12, B = 24;
  const X = (x) => L + (x - x0) / (x1 - x0) * (W - L - R), Y = (y) => Tp + (y1 - y) / (y1 - y0) * (H - Tp - B);
  g.font = '11px ' + getComputedStyle(document.body).fontFamily; g.fillStyle = T['text-3']; g.strokeStyle = T['flaeche-3']; g.lineWidth = 1;
  for (let v = -4; v <= 4; v += 2) { g.beginPath(); g.moveTo(X(v), Tp); g.lineTo(X(v), H - B); g.stroke(); g.fillText(num(v, 0), X(v) - 4, H - 8); }
  for (const v of [Math.ceil(y0), 0, 1, Math.floor(y1)].filter((a, i, s) => s.indexOf(a) === i && a >= y0 && a <= y1)) { g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(W - R, Y(v)); g.stroke(); g.fillText(num(v, 0), 6, Y(v) + 4); }
  g.strokeStyle = T['text-3']; g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(W - R, Y(0)); g.moveTo(X(0), Tp); g.lineTo(X(0), H - B); g.stroke();
  const curve = (fn, col, dash) => {
    g.strokeStyle = col; g.lineWidth = 2.4; g.setLineDash(dash || []); g.beginPath();
    let first = true;
    for (let i = 0; i <= 240; i++) {
      const x = x0 + (i / 240) * (x1 - x0), y = clamp(fn(x), y0 - 1, y1 + 1);
      if (first) { g.moveTo(X(x), Y(y)); first = false; } else g.lineTo(X(x), Y(y));
    }
    g.stroke(); g.setLineDash([]);
  };
  curve(A.d, T.grad, [5, 4]); curve(A.f, T.pos);
  g.fillStyle = T.pos; g.fillText('f(z)', L + 6, Tp + 12); g.fillStyle = T.grad; g.fillText("f′(z)", L + 6, Tp + 26);
  if (z !== undefined && z !== null) {
    const zc = clamp(z, x0, x1), fz = A.f(z), dz = A.d(z);
    g.strokeStyle = T['text-2']; g.setLineDash([3, 3]); g.lineWidth = 1; g.beginPath(); g.moveTo(X(zc), H - B); g.lineTo(X(zc), Y(clamp(fz, y0, y1))); g.stroke(); g.setLineDash([]);
    for (const [v, col] of [[dz, T.grad], [fz, T.pos]]) { g.fillStyle = col; g.beginPath(); g.arc(X(zc), Y(clamp(v, y0, y1)), 5, 0, 7); g.fill(); g.strokeStyle = T.flaeche; g.lineWidth = 2; g.stroke(); }
    g.fillStyle = T.text; g.fillText('z = ' + num(z), clamp(X(zc) + 6, L, W - 70), H - B - 6);
  }
}
// Karte: Wert über der Eingabefläche (x1 nach rechts, x2 nach oben)
function plotMap(cv, vals, G, data, mode, clsIdx) {
  const W = 220, g = prepCanvas(cv, W, W), cell = W / G;
  let mx = 1e-9; for (const v of vals) mx = Math.max(mx, Math.abs(v));
  const c = new THREE.Color();
  for (let r = 0; r < G; r++) for (let q = 0; q < G; q++) {
    const v = vals[r * G + q];
    if (mode === 'prob') c.copy(T.c.flaeche).lerp(T.cls[clsIdx], v * 0.9); else valueColor(v / mx, c);
    g.fillStyle = cssRGB(c); g.fillRect(q * cell, (G - 1 - r) * cell, cell + 0.6, cell + 0.6);
  }
  if (data) for (let i = 0; i < data.X.length; i++) {
    const [a, b] = data.X[i]; g.fillStyle = T.cls[data.Y[i]].getStyle(); g.globalAlpha = 0.55;
    g.beginPath(); g.arc((a + 1) / 2 * W, (1 - (b + 1) / 2) * W, 2.2, 0, 7); g.fill();
  }
  g.globalAlpha = 1;
  return mx;
}
function gridHTML(vals, n, opts = {}) {
  let mx = opts.max || 1e-9; if (!opts.max) for (const v of vals) mx = Math.max(mx, Math.abs(v));
  let s = `<div class="gitter" style="grid-template-columns:repeat(${n},34px)">`;
  vals.forEach((v, i) => {
    const hi = opts.mark === i ? 'outline:2px solid var(--text);outline-offset:-2px;' : '';
    const bg = opts.gray ? `color-mix(in srgb, var(--tinte) ${Math.round(clamp(v, 0, 1) * 70)}%, var(--flaeche-2))` : cssValue(v / mx);
    s += `<span style="background:${bg};${hi}">${num(v, opts.d === undefined ? 2 : opts.d)}</span>`;
  });
  return s + '</div>';
}
function balkenHTML(names, probs, colors, mark) {
  return '<div class="balken">' + names.map((n, k) => `<div><span>${mark === k ? '<b>' + esc(n) + ' ✓</b>' : esc(n)}</span><i style="width:${Math.max(1, probs[k] * 100)}%;background:${colors[k]}"></i><span>${num(probs[k], 2)}</span></div>`).join('') + '</div>';
}
const schalter = (key, text, on) => `<label class="schalter"><span>${text}</span><input type="checkbox" data-set="${key}" ${on ? 'checked' : ''}></label>`;
const auswahl = (key, text, opts, val) => `<div class="zeile"><label for="s-${key}">${text}</label><select id="s-${key}" data-set="${key}">${opts.map(([v, t]) => `<option value="${v}" ${String(v) === String(val) ? 'selected' : ''}>${t}</option>`).join('')}</select></div>`;

// Aktivierungsfunktionen: Steckbriefe nach L2
const ACTINFO = {
  heaviside: { name: 'Heaviside', tex: 'f(z) = \\begin{cases} 1 & z \\ge 0 \\\\ 0 & z < 0 \\end{cases}', plus: ['„Alles oder nichts“ wie beim Perzeptron'], minus: ['Die Ableitung ist fast überall \\(0\\). Backpropagation bekommt dadurch kein Fehlersignal, nur die letzte Schicht lernt.'] },
  sign: { name: 'Sign', tex: 'f(z) = \\operatorname{sign}(z) \\in \\{-1, 0, 1\\}', plus: ['Normierte Ausgabe'], minus: ['Die Ableitung ist fast überall \\(0\\): kein Fehlersignal für Backpropagation.'] },
  linear: { name: 'Linear', tex: 'f(z) = z', plus: ['Ableitung überall \\(1\\)'], minus: ['Keine Nichtlinearität: mehrere Schichten fallen zu einer einzigen linearen Abbildung zusammen (Kollaps aus L2).'] },
  sigmoid: { name: 'Sigmoid', tex: '\\sigma(z) = \\frac{1}{1 + e^{-z}} \\in (0, 1)', plus: ['Ausgabe zwischen \\(0\\) und \\(1\\), wie eine Wahrscheinlichkeit', 'differenzierbar'], minus: ['Sättigt für \\(z \\ll 0\\) und \\(z \\gg 0\\): die Ableitung ist dort fast \\(0\\) (verschwindende Gradienten)', 'Nicht null-zentriert', 'Größte Ableitung nur \\(0{,}25\\) (bei \\(z = 0\\))'] },
  tanh: { name: 'Tanh', tex: '\\tanh(z) = 2\\sigma(2z) - 1 \\in (-1, 1)', plus: ['Null-zentriert'], minus: ['Sättigt ebenfalls für große \\(|z|\\)'] },
  relu: { name: 'ReLU', tex: 'f(z) = \\max(0, z)', plus: ['Ableitung \\(1\\) für \\(z > 0\\): der Gradient wird nicht kleiner', 'schnell zu rechnen'], minus: ['„Dying ReLU“: für \\(z < 0\\) ist die Ableitung \\(0\\); bleibt \\(z\\) immer negativ, lernt das Neuron nichts mehr'] },
  leaky: { name: 'Leaky ReLU', tex: 'f(z) = \\begin{cases} z & z > 0 \\\\ \\alpha z & z \\le 0 \\end{cases}, \\quad \\alpha = 0{,}01', plus: ['Behebt Dying ReLU: Ableitung \\(\\alpha\\) statt \\(0\\)'], minus: [] },
  swish: { name: 'Swish (SiLU)', tex: 'f(z) = z \\cdot \\sigma(z)', plus: ['Gradient fließt auch für \\(z < 0\\)', 'glatter als Leaky ReLU'], minus: ['etwas teurer zu rechnen'] },
  gelu: { name: 'GELU', tex: 'f(z) = z \\cdot \\Phi(z)', note: '\\(\\Phi\\) ist die Verteilungsfunktion der Standardnormalverteilung.', plus: ['Glatter Gradient auch für \\(z < 0\\)', 'üblich in Transformern'], minus: [] },
};
const ACTKEYS = Object.keys(ACTINFO);
function actCard(key, z, canvasId) {
  const a = ACTINFO[key];
  return `<div class="tex">$$${a.tex}$$</div>${a.note ? `<p class="klein">${a.note}</p>` : ''}
    <figure><canvas class="plot" id="${canvasId}"></canvas><figcaption>Blau: \\(f(z)\\). Lila gestrichelt: Ableitung \\(f'(z)\\), also wie stark ein Fehlersignal beim Rückwärtsgehen durchgelassen wird.${z !== undefined ? ' Punkte: Werte beim gewählten Beispiel.' : ''}</figcaption></figure>
    <div class="legende">${a.plus.map((t) => `<div><span class="punkt" style="background:var(--gut)"></span><span>${t}</span></div>`).join('')}${a.minus.map((t) => `<div><span class="punkt" style="background:var(--neg)"></span><span>${t}</span></div>`).join('')}</div>`;
}

// ---------- Hauptschleife
let lastT = performance.now();
function loop(now) {
  const dt = Math.min(0.05, (now - lastT) / 1000); lastT = now;
  if (App.mode) {
    App.mode.frame(dt, now / 1000);
    if (App.liveDirty && now - App.live > 350) { App.live = now; App.liveDirty = false; App.mode.live && App.mode.live(inhalt, App.sel); App.updateDock(); }
  }
  Pulses.update(now / 1000);
  flyUpdate(now);
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  requestAnimationFrame(loop);
}
App.start = () => {
  readTheme(); Pulses.init(); resize();
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  const onTheme = () => { readTheme(); for (const m of Object.values(App.modes)) m.built && m.themeChanged && m.themeChanged(); App.renderPanel(); App.updateDock(); };
  mq.addEventListener ? mq.addEventListener('change', onTheme) : mq.addListener(onTheme);
  new MutationObserver(onTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  const h = (location.hash || '').slice(1);
  App.setMode(App.modes[h] ? h : 'mlp');
  $('#lade').hidden = true;
  requestAnimationFrame(loop);
};
