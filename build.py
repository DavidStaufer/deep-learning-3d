import pathlib, sys, datetime
D = pathlib.Path(__file__).parent
r = lambda f: (D / f).read_text(encoding='utf-8')
TITEL = 'Deep Learning 3D'
BESCHR = 'Drei echte Netze in 3D zum Anfassen: Fully Connected, CNN und RNN/LSTM/GRU, mit Training, Backpropagation und jeder Rechnung zum Antippen.'
head = f'''<title>{TITEL}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="description" content="{BESCHR}">
<meta name="lernseite:id" content="deep-learning-3d">
<meta name="lernseite:modul" content="Deep Learning">
<meta name="lernseite:art" content="konzept">
<meta name="lernseite:datum" content="2026-11-19">
<style>
{r('app.css')}
</style>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['$$', '$$']] }},
  svg: {{ fontCache: 'global' }},
  options: {{ enableMenu: false }},
  startup: {{ pageReady: function () {{ return MathJax.startup.defaultPageReady().then(function () {{ mjReady = true; typeset(document.getElementById('inhalt')); }}); }} }}
}};
var mjReady = false;
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>'''
body = r('body.html') + '''
<script src="https://cdn.jsdelivr.net/npm/three@0.147.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.147.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.147.0/examples/js/renderers/CSS2DRenderer.js"></script>
<script>
''' + '\n'.join(r(f) for f in ['engine.js', 'models.js', 'core.js', 'mode-mlp.js', 'mode-cnn.js', 'mode-rnn.js']) + '''
App.start();
</script>'''
art = head + '\n' + body + '\n'
lokal = '<!doctype html>\n<html lang="de">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + head + '\n</head>\n<body>\n' + body + '\n</body>\n</html>\n'
out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else D / 'out'
out.mkdir(parents=True, exist_ok=True)
(out / 'deep-learning-3d.html').write_text(art, encoding='utf-8')
(out / 'lokal.html').write_text(lokal, encoding='utf-8')
print('ok', len(art) // 1024, 'KB')
