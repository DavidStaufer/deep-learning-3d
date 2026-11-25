global.DL = require('./engine.js');
const { param, tensor, conv2d, maxpool, act, flatten, linear, softmaxCE, backward, noGrad, randn } = DL;
DL.seed(5);
const K = param([3,2,3,3], ()=>randn()*0.5), b = param([3], ()=>randn()*0.1), W = param([4, 3*3*3], ()=>randn()*0.3), b2 = param([4], ()=>0);
const xin = Float64Array.from({length: 2*6*6}, ()=>randn());
const xs = param([2,6,6], (k)=>xin[k]);
const params=[K,b,W,b2,xs];
const f = () => { const c = conv2d(xs, K, b, 1, 1); const t = act(c, 'tanh'); const p = maxpool(t, 2); const fl = flatten(p); return softmaxCE(linear(W, b2, fl), 1); };
for (const p of params) p.g.fill(0); backward(f());
let mx=0; for (const p of params) for (let k=0;k<p.n;k++){ const e=1e-6,o=p.d[k]; p.d[k]=o+e; const lp=noGrad(f).d[0]; p.d[k]=o-e; const lm=noGrad(f).d[0]; p.d[k]=o; const num=(lp-lm)/(2*e); const rel=Math.abs(num-p.g[k])/Math.max(1e-7,Math.abs(num)+Math.abs(p.g[k])); if(rel>mx){mx=rel;} }
console.log('conv+pool smooth check max rel', mx.toExponential(2));
// stride 2 check
const f2 = () => { const c = conv2d(xs, K, b, 1, 2); const t = act(c, 'tanh'); const fl = flatten(t); return softmaxCE(linear(W, b2, fl), 2); };
for (const p of params) p.g.fill(0); backward(f2()); mx=0;
for (const p of params) for (let k=0;k<p.n;k++){ const e=1e-6,o=p.d[k]; p.d[k]=o+e; const lp=noGrad(f2).d[0]; p.d[k]=o-e; const lm=noGrad(f2).d[0]; p.d[k]=o; const num=(lp-lm)/(2*e); const rel=Math.abs(num-p.g[k])/Math.max(1e-7,Math.abs(num)+Math.abs(p.g[k])); if(rel>mx){mx=rel;} }
console.log('stride2 check max rel', mx.toExponential(2));
