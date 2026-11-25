# Deep Learning 3D

Drei echte neuronale Netze in 3D zum Anfassen: **Fully Connected**, **CNN** und **RNN / LSTM / GRU**. Man kann sie trainieren, Vorwärts- und Rückwärtsdurchlauf (Backpropagation) Schritt für Schritt ansehen und jede Rechnung antippen.

Die Netze rechnen wirklich: Darunter liegt eine kleine Autograd-Engine in reinem JavaScript (`engine.js`), ohne ML-Bibliothek. Die 3D-Darstellung nutzt [three.js](https://threejs.org), die Formeln [MathJax](https://www.mathjax.org).

## Was drin ist

| Modus | Netz | Aufgabe |
|---|---|---|
| Fully Connected | MLP mit wählbaren Schichten, Aktivierungen (Sigmoid, tanh, ReLU, Leaky ReLU, Swish, GELU …), Initialisierung, Dropout und Optimierer (SGD, Momentum, Adam) | 2D-Klassifikation: XOR, Kreise, Spirale, Wolken |
| CNN | 2 × (Faltung → ReLU → Max-Pooling) → Dense | Formen auf kleinen Bildern erkennen |
| RNN · LSTM | Elman-RNN, LSTM oder GRU | Text Zeichen für Zeichen vorhersagen |

## Dateien

| Datei | Inhalt |
|---|---|
| `engine.js` | Tensoren, Autograd, Schichten (Linear, Conv2D, Max-Pooling, Aktivierungen, Softmax + Cross-Entropy), Optimierer |
| `models.js` | Die drei Modelle (MLP, CNN, RNN/LSTM/GRU) und ihre Trainingsdaten |
| `core.js` | App-Gerüst, 3D-Szene, Kamera, Panel |
| `mode-mlp.js`, `mode-cnn.js`, `mode-rnn.js` | Darstellung und Erklärungen pro Modus |
| `body.html`, `app.css` | Seitenaufbau und Stil (hell und dunkel) |
| `build.py` | Baut alles zu einer einzigen HTML-Datei zusammen |
| `test.js`, `test2.js` | Gradienten-Checks und Trainings-Tests |

## Bauen und öffnen

```bash
python3 build.py
open out/lokal.html
```

`build.py` schreibt zwei Dateien nach `out/`: `lokal.html` ist eine vollständige Seite für den Browser, `deep-learning-3d.html` ist nur der Seiteninhalt (ohne `<html>`-Gerüst) zum Einbetten.

## Tests

```bash
node test.js
node test2.js
```

`test.js` vergleicht die analytischen Gradienten aller Modelle mit numerischen und prüft, dass das Training konvergiert. Beim CNN meldet der numerische Check dort `!!!`, weil ReLU und Max-Pooling Knicke haben, an denen der Differenzenquotient danebenliegt. `test2.js` prüft Faltung und Pooling deshalb zusätzlich mit tanh, also ohne Knicke, und liegt bei einem relativen Fehler um $10^{-6}$.
