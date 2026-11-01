# Kegelform – ein Format für LP, SOCP und SDP – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kegelform-demo.streamlit.app/)** -->

Stück 2 von 16 der **Konzepte-Reihe „Konvexe und konische Optimierung“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Löser für Kegelprogramme, dessen Kegel von Stück zu Stück wächst (Orthant, Lorentz-Kegel, PSD-Kegel). Vorgänger ist [konvexitaet-zertifikat-demo](https://github.com/sebastian-hanisch/konvexitaet-zertifikat-demo) (die kleinste Kugel mit Zertifikat); Nachfolger sind Kegel-Dualität (Stück 4) und die Barriere im Kegel (Stück 5). Die Linien-Spezifikation und die Einbindung in die Website folgen, wenn die Reihe vollständig ist.

## Kernfrage

**Wird aus einer Aufgabe ein Kegelprogramm, und stimmt es?** Ein Kegelprogramm hat drei Zutaten: `min cᵀx` unter der Bedingung, dass `s = b − A x` in einem Kegel liegt, der ein Produkt aus **Orthant** (LP), **Lorentz-Kegel** (SOCP) und **PSD-Kegel** (SDP) ist. Dieselbe Aufgabe lässt sich in mehreren Formen schreiben; die Form entscheidet über Größe und Aufwand, und **jeder Kegel kann etwas, das der vorige nicht kann**.

Das Beispiel ist wie in Stück 1 die kleinste umschließende Form einer Punktwolke, jetzt in sechs Formen: Würfel (l∞), Raute (l1), Kugel als Lorentz-Kegel, dieselbe Kugel als SDP (Schur-Komplement), Ellipse achsenparallel (gedrehte Lorentz-Kegel) und Ellipse allgemein (PSD-Kegel). Bei den Ellipsen liegt der Mittelpunkt im Ursprung, das Ziel ist die kleinste Summe der quadrierten Halbachsen (`trace(P⁻¹)`).

## Modell

- **Kegelform:** `min cᵀx`, `s = b − A x ∈ K`. Der PSD-Kegel wird als Vektor `svec(S)` geführt (Außerdiagonale mit √2, damit `⟨S, T⟩ = svec(S)·svec(T)`).
- **Löser** (`kegel_kern.py`, wortgleich in allen Stücken der Reihe, nur numpy): primale Barrierenmethode. Für wachsendes `t` wird `t cᵀx + φ(b − A x)` mit Newton minimiert; `φ` ist die Summe der Barrieren der Kegel (−Σ log s, −log(s₀² − ‖s̄‖²), −log det S). Zu jedem Punkt gilt die **bewiesene Lücke** `ν/t` zum Optimum (ν: Orthant m, Lorentz-Kegel 2, PSD-Kegel k). Gleiche Kegel werden gebündelt gerechnet.
- **Rückumformung:** Das Ergebnis wird in die Ausgangsaufgabe zurückübersetzt und dort geprüft (Normen ≤ R, `xᵀPx ≤ 1`, P positiv definit) und im Kegelprogramm von einem unabhängigen Prüfer (`s` im Kegel: Eigenwerte, `s₀ − ‖s̄‖`, Komponenten), nicht von der Barriere.
- **Strenger Start:** Jede Form liefert einen streng zulässigen Startpunkt durch Konstruktion mit (Würfel, Raute, Kugel: Mittelwert und großer Radius; Ellipse: kleine Matrix). Ein allgemeines Verfahren ohne Startpunkt kommt in Stück 6.

## Methodik

Vier Experimente, ein Baukasten und ein Datenblick, jedes Experiment mit live berechnetem Verdikt: (1) eine Wolke, sechs Formen (Tabelle mit Zielwert, Größe, Kegel, Newton-Schritten und bewiesener Lücke; dazu die Lücke über den Newton-Schritten der gewählten Form); (2) wie viele LP-Zeilen ein Lorentz-Kegel ersetzt; (3) dieselbe Kugel als Lorentz-Kegel und als PSD-Kegel; (4) was die Drehung der allgemeinen Ellipse bringt. 🧰 Baukasten: welche Bedingung in welchen Kegel gehört. 🔍 Datenblick: `A`, `b`, `c` und Kegelliste für die ersten drei Punkte.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`; Experimente laufen über feste Wolken `SWEEP_SEEDS`, nicht über einen einzelnen Seed. Die Standardwolke (20 Punkte, Streckung 4, 45 Grad, Seed 35) ist ausdrücklich eine einzelne.

| Frage | Befund |
|---|---|
| Handinstanz (1,1), (−1,−1), (0,5; −0,5), (−0,5; 0,5) | Würfel 1, Raute 2, Kugel √2 (beide Formen), Ellipse achsenparallel 4, Ellipse allgemein 2,5: alle sechs Werte stimmen mit der Handrechnung überein; Verhältnis 0,625 |
| Standardwolke, sechs Formen | Würfel 5,784 (3 Variablen, 80 Zeilen); Raute 10,090 (43, 100); Kugel 7,168 (3, 60 als 20 Lorentz-Kegel der Größe 3) und 7,168 (3, 120 als 20 PSD-Kegel der Größe 3); Ellipse achsenparallel 134,27 (4, 26); Ellipse allgemein 78,52 (6, 30); Verhältnis allgemein/achsenparallel 0,585 |
| Rückumformung und Lücke | Alle 90 Läufe (6 Formen auf 15 festen Wolken in der Ebene) erreichen die Lücke 10⁻⁹, bestehen die Prüfung im Kegelprogramm und in der Ausgangsaufgabe |
| Newton-Schritte (diese Implementierung) | 29 bis 84 je Lauf, Median je Form 31 (Würfel, Kugel, Kugel als SDP), 38 (Raute), 47 (Ellipse achsenparallel), 57 (Ellipse allgemein). Keine Eigenschaft der Kegel: gedämpftes Newton-Verfahren, `t` je Runde mal 8 |
| Kugel als Lorentz-Kegel und als PSD-Kegel (n = 6 bis 30) | Gleicher Radius (Abweichung unter 10⁻⁹); der PSD-Kegel hat bei jeder Punktzahl **genau doppelt so viele Zeilen** (6 n statt 3 n); Newton-Schritte 30 bis 33 in beiden Formen, höchstens 3 auseinander |
| Wie viele LP-Zeilen ersetzt ein Lorentz-Kegel? (Radius höchstens 1 % daneben, 20 Punkte, Median über 3 feste Wolken, ohne Hilfsvariablen) | **16 Richtungen je Punkt** in der Ebene (regelmäßiges Vieleck), **256** in d = 3, **2048** in d = 4 (Zufallsrichtungen, einmal gezogen); bei 2048 Richtungen noch 0,07 % (d = 3) und 0,7 % (d = 4) Fehler |
| Was bringt die Drehung? (Wolke 4 : 1 gestreckt, 20 Punkte, 5 feste Wolken je Winkel) | Verhältnis der Summe der quadrierten Halbachsen allgemein/achsenparallel im Median 0,88 / 0,83 / 0,62 / 0,53 / 0,56 / 0,66 / 0,88 bei 0 / 15 / 30 / 45 / 60 / 75 / 90 Grad, in keiner Wolke über 1 (kleinster Wert 0,52) |
| Gegenprobe | cvxpy mit Clarabel (anderer Löser, andere Modellierung) stimmt auf 12 Wolken in 2 und 3 Dimensionen für alle sechs Formen auf 10⁻⁶ (Kugeln, Würfel, Raute) bzw. 10⁻⁵ (Ellipsen) überein |

## Befunde und Korrekturen gegenüber dem Plan

- **Kugel als SDP kostet nicht mehr Newton-Schritte.** Phase 0 sah 43 bis 148 Schritte gegen 43 bis 54; das war ein Artefakt des Prototyps (Zentrieren bis 10⁻⁹ am Rand der Rundung). Mit Zentrieren bis 10⁻⁶ und einem Start-`t` aus dem Gradienten sind es in beiden Formen etwa 30. Der Unterschied liegt in der Größe: doppelt so viele Zeilen und eine größere Hesse-Matrix je Schritt.
- **Raute in d = 3 endet an der Rechengenauigkeit.** Bei `t` um 2·10¹¹ wird die Hesse-Matrix numerisch singulär (Kondition über 10²²). Der Kern fängt das ab: er gibt die zuletzt zentrierte Runde zurück, setzt `converged = False` und weist die dort bewiesene Lücke aus (auf dem Entwicklungsrechner in 9 von 240 Läufen mit d = 2 bis 4 zwischen 2·10⁻⁹ und 1,5·10⁻⁷; auf GitHub-Runnern je nach CPU schon in 3: wo die Rechengenauigkeit endet, hängt von der BLAS-Auswahl ab). Ein Test hält den Fall fest (`test_diamond_in_three_dimensions_…`).
- **LP-Nachbildung braucht `R ≥ 0`.** Mit zu wenigen Richtungen ist `max_j u_j·v` keine Norm, und das LP wird unbeschränkt; die Zeile `R ≥ 0` hält es beschränkt (Fehler dann 100 % statt 10¹⁶).
- **Laufzeit der PSD-Form:** Der Prototyp brauchte 1 bis 4,5 s für 20 PSD-Kegel; gebündelt gerechnet (Stapel statt Schleife) sind es 0,06 bis 0,2 s.
- **Gemeinsamer Kern:** Die Eintragung in `SHARED_GROUPS` (`tools/portfolio_audit.py`) entfällt, solange nur ein Repo `kegel_kern.py` trägt; sie folgt mit Stück 5.
- **Port:** 9300 statt 9291 (von einer anderen Demo belegt).

## Ehrliche Grenzen

- **Die LP-Nachbildung hat keine Hilfsvariablen.** Mit Hilfsvariablen lässt sich der Lorentz-Kegel durch ein LP von vertretbarer Größe beliebig genau annähern (Ben-Tal und Nemirovski, 2001); Experiment 2 zeigt nur die einfache Variante, und ab d = 3 mit Zufallsrichtungen (eine bessere Verteilung braucht weniger, wächst aber ähnlich).
- **Schrittzahlen** gelten für diese Implementierung (Barrierenmethode, μ = 8, Start durch Konstruktion). Aussagen über Iterationsschranken der Kegel folgen in Stück 5.
- **Ellipsen nur um den Ursprung und mit Spurkriterium.** Ein freier Mittelpunkt oder das Volumenziel (`log det`) ist kein reines SDP in dieser Form.
- **Keine Dualität:** Die Lückenschranke kommt aus der Barriere, nicht aus einem dualen Kegelprogramm (Stück 4).
- **Zeiten** in Experiment 3 sind auf dem jeweiligen Server gemessen und werden nur angezeigt, nicht getestet.
- **`ok = True` und `converged = True` im Kern heißen nicht „zentriert“.** `center` gilt auch nach 100 erschöpften Schritten, bei negativem Dekrement-Quadrat und im Rundungsstopp als ok; die Schranke `ν/t` gilt streng nur für einen zentrierten Punkt, und bei großem t ist das Dekrement durch Rundung nicht verlässlich bestimmbar. Das Ergebnis des Kerns trägt seit der Härtung das Feld `far_from_path` für die gefährlichste Klasse (eine Zentrierung endete nach 100 Schritten weit vom Pfad, siehe `tests/test_kern.py`). In den 90 Läufen der sechs Formen auf den 15 Wolken meldet das Feld nie. Eine strengere Härtung wurde gemessen und verworfen, weil sie veröffentlichte Läufe der Stücke 2 und 4 abbrechen ließe, die richtige Werte liefern. Keine Zahl dieser README ändert sich.
- **d = 2 im Hauptdiagramm;** in höheren Dimensionen laufen die Formen (getestet in d = 1 und 3), das Hauptdiagramm und die Experimente 3 und 4 sind aber auf die Ebene begrenzt.

## Verifikation

- **Kern** (`tests/test_kern.py`): Gradient und Hesse-Matrix aller drei Barrieren gegen finite Differenzen; gebündelte Barriere gleich Summe der Einzelblöcke; `svec` isometrisch; Mini-Programme mit bekanntem Optimum (min x mit x ≥ 1; min t mit (t; 3, 4) im Kegel; min x mit [[x, 1],[1, x]] ⪰ 0), Lückenschranke gilt; singuläre Hesse-Matrix (simuliert) beendet den Lauf sauber.
- **Formen** (`tests/test_algorithm.py`): sechs Werte der Handinstanz; Größen von Hand; jede Zeile des Würfels, die Schur-Struktur der PSD-Kugel, der gedrehte Lorentz-Kegel `p·t ≥ 1` und der Block `[[P, I],[I, T]]` von Hand; Rückumformung nennt jede verletzte Regel.
- **Orakel** (`tests/test_oracle_kegelform.py`): Kugel gegen Kreise durch zwei und drei Punkte (30 Wolken); Würfel gegen die halbe größte Spannweite; cvxpy mit Clarabel für alle sechs Formen (12 Wolken); Raute und LP-Nachbildung gegen scipy `linprog` (HiGHS). Eigenschaften: Kette Würfel ≤ Kugel ≤ Raute, allgemein ≤ achsenparallel, Skalierung (Radius × a, Spur × a²), Reihenfolge der Punkte, Drehinvarianz von Kugel und allgemeiner Ellipse (nicht der achsenparallelen), d = 1.
- **Randfälle** (`tests/test_edge_cases.py`): ein Punkt, zwei Punkte, gleiche Punkte, kollineare Punkte, d = 1 und 3, Punkt im Ursprung, ungültige Eingaben mit Meldung, Start außerhalb des Kegels.
- **Verdikte** (`tests/test_verdicts.py`): reine Funktionen; jede Zahl im Text an handgesetzten Zeilen. **Claims** (`tests/test_claims.py`): jede Zahl dieser README.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum), Experimente 2 und 4, PDF, jede Form, alle Regler zugleich, Presets, Permalink, Layout, Footer, Laufzeit-Importe.
- **Umfang:** 129 Tests (2 davon, die cvxpy-Orakel, überspringen sich ohne cvxpy), rund 1,5 Minuten (2026-11-01, Windows).
- **Nicht gelaufen:** kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs). cvxpy und scipy sind Gegenproben im dev-Satz; ohne sie überspringen sich die jeweiligen Tests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kegel_kern.py` | Gemeinsamer Kern der Reihe: Kegelform, Barrieren der drei Kegel, Barrierenmethode, Prüfer (wortgleich in allen Stücken) |
| `kgf_algorithm.py` | Die sechs Formen als Kegelprogramm, Rückumformung, LP-Nachbildung der Kugel |
| `kgf_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kgf_evaluation.py` | Analyse, Experimente, Verdikte, Formatierung |
| `kgf_layout.py` | Fester Kopf (Stylesheet) |
| `kgf_pdf_export.py` | PDF-Export der aktuellen Wolke (fpdf2) |
| `kgf_presets.py` | Permalink (Begrenzen, gültige Auswahlwerte), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kgf_scenario.py` | Punktwolken (gestreckt, gedreht), Normalwolken in d Dimensionen, Handinstanz |
| `kgf_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und cvxpy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Kern, Formen, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, Abbildungen, AppTest-Rauchtest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Duales Kegelprogramm, Slater, Unzulässigkeitszertifikate:** Stück 4.
- **Zentraler Pfad, Barriere-Parameter und Iterationsschranken im Detail:** Stück 5; **Primal-Dual-Schritt und allgemeiner Start:** Stück 6; **Exponential- und Potenzkegel:** Stück 7.
- **Konvexitätsprüfung nach Regeln (DCP):** optionales Stück 3.
- **Hilfsvariablen für die LP-Nachbildung des Lorentz-Kegels** (Ben-Tal und Nemirovski): nicht gebaut.
- **Ellipse mit freiem Mittelpunkt oder Volumenziel:** nicht gebaut.
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
