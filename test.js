global.DL = require('./engine.js'); const M = require('./models.js');
const { backward, noGrad } = DL;
function gradCheck(name, params, lossFn) {
  for (const p of params) p.g.fill(0);
  const L = lossFn(); backward(L);
  let maxRel = 0;
  for (const p of params) {
    const ana = Float64Array.from(p.g);
    for (let t = 0; t < Math.min(p.n, 25); t++) {
      const k = Math.floor(DL.rnd() * p.n), e = 1e-5, o = p.d[k];
      p.d[k] = o + e; const lp = noGrad(lossFn).d[0]; p.d[k] = o - e; const lm = noGrad(lossFn).d[0]; p.d[k] = o;
      const num = (lp - lm) / (2 * e), a = ana[k];
      const rel = Math.abs(num - a) / Math.max(1e-6, Math.abs(num) + Math.abs(a));
      if (rel > maxRel) maxRel = rel;
    }
  }
  console.log(name.padEnd(22), 'max rel err', maxRel.toExponential(2), maxRel < 1e-4 ? 'OK' : '!!!');
}
DL.seed(3);
for (const a of ['sigmoid','tanh','relu','leaky','swish','gelu','linear']) {
  const net = new M.MLP({ hidden: [5, 4], act: a, K: 3, dropout: 0 });
  gradCheck('MLP '+a, net.params, () => net.forward([0.3, -0.7], 1).loss);
}
const cnn = new M.CNN(); const img = M.makeShape(2);
gradCheck('CNN', cnn.params, () => cnn.forward(img, 2, false).loss);
for (const t of ['elman','lstm','gru']) { const r = new M.RNN(t); gradCheck('RNN '+t, r.params, () => r.run(3, 6).loss); }
// GELU derivative check
for (const z of [-2,-0.5,0,0.7,2]) { const e=1e-5, A=DL.ACT.gelu; const num=(A.f(z+e)-A.f(z-e))/(2*e); if (Math.abs(num-A.d(z))>1e-4) console.log('gelu d bad', z); }
// Training sanity
for (const ds of ['xor','kreise','spirale','wolken']) {
  DL.seed(1); const data = M.makeData(ds); const net = new M.MLP({ hidden: ds==='spirale'?[8,8]:[6,6], K: data.K, act: 'relu' });
  const t0 = Date.now(); for (let s = 0; s < 600; s++) net.trainStep(data);
  const ev = net.evaluate(data); console.log('MLP', ds.padEnd(8), 'loss', ev.loss.toFixed(3), 'acc', ev.acc.toFixed(3), (Date.now()-t0)+'ms');
}
{ DL.seed(1); const data = M.makeData('xor'); const net = new M.MLP({ hidden: [], K: 2 }); for (let s=0;s<600;s++) net.trainStep(data); console.log('XOR ohne versteckte Schicht acc', net.evaluate(data).acc.toFixed(3)); }
{ DL.seed(1); const data = M.makeData('xor'); const net = new M.MLP({ hidden: [6,6], act:'heaviside', K: 2 }); for (let s=0;s<600;s++) net.trainStep(data); console.log('XOR Heaviside acc', net.evaluate(data).acc.toFixed(3)); }
{ DL.seed(1); const data = M.makeData('xor'); const net = new M.MLP({ hidden: [6,6], act:'linear', K: 2 }); for (let s=0;s<600;s++) net.trainStep(data); console.log('XOR linear acc', net.evaluate(data).acc.toFixed(3)); }
{ DL.seed(2); const t0=Date.now(); const c = new M.CNN(); let last; for (let s=0;s<400;s++){ last=c.trainStep(16); if(s%100===99) console.log('CNN step', s+1, 'loss', last.loss.toFixed(3), 'acc', last.acc.toFixed(2), (Date.now()-t0)+'ms'); }
  let ok=0; for (let i=0;i<400;i++){ const cls=i%4; const p=noGrad(()=>c.forward(M.makeShape(cls))).probs; let b=0; for(let k=1;k<4;k++) if(p[k]>p[b]) b=k; if(b===cls) ok++; } console.log('CNN test acc', ok/400); }
for (const t of ['elman','lstm','gru']) { DL.seed(4); const r = new M.RNN(t); const t0=Date.now(); for (let s=0;s<400;s++) r.trainStep(); const ev=r.evaluate(); console.log('RNN', t.padEnd(6), 'acc', ev.acc.toFixed(3), 'loss', ev.loss.toFixed(3), (Date.now()-t0)+'ms', JSON.stringify(r.generate(28))); }
