"""Kegelform – ein Format für LP, SOCP und SDP - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 2 der Reihe „Konvexe und konische Optimierung“: Dieselbe Aufgabe (kleinste umschließende Form einer Punktwolke) in sechs Kegelformen. Jede ist ein Kegelprogramm
min c^T x mit s = b - A x im Kegel; ein gemeinsamer Barrierlöser (kegel_kern.py, wortgleich in allen Stücken der Reihe) löst alle sechs und beweist die Lücke.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import kgf_algorithm as A
import kgf_constants as C
from kgf_evaluation import (
    Settings,
    analyse,
    cones_text,
    data_view,
    de,
    form_rows,
    lp_experiment,
    psd_experiment,
    rotation_experiment,
    sci,
    verdict_forms,
    verdict_form,
    verdict_hand,
    verdict_lp,
    verdict_psd,
    verdict_rotation,
)
from kgf_layout import dunkles_design, kopf_css
from kgf_pdf_export import build_pdf
from kgf_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kgf_scenario import HAND_POINTS
from kgf_visualization import build_forms, build_gap, build_lp, build_psd, build_rotation, build_shape

st.set_page_config(page_title="Kegelform – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _psd():
    return psd_experiment()


@st.cache_data(show_spinner=False)
def _hand():
    return {f: A.solve_form(f, HAND_POINTS).result.value for f in A.FORMS}


@st.cache_data(show_spinner=False)
def _lp():
    return lp_experiment()


@st.cache_data(show_spinner=False)
def _rotation(n):
    return rotation_experiment(n=n)


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kgf_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Kegelform – ein Format für LP, SOCP und SDP")
st.markdown(
    """
**Ein Kegelprogramm hat nur drei Zutaten:** `min cᵀx` unter der Bedingung, dass `s = b − A x` in einem **Kegel** liegt. Der Kegel ist ein Produkt aus **Orthant** (LP), **Lorentz-Kegel** (SOCP) und **PSD-Kegel** (SDP).
Die Demo schreibt **dieselbe Aufgabe in sechs Formen** (Würfel, Raute, Kugel, Kugel als SDP, zwei Ellipsen) und löst sie mit einem Löser: Die Form entscheidet über Größe und Aufwand, und **im Allgemeinen kann jeder Kegel etwas, das der vorige nicht kann** (für PSD gegenüber Lorentz zeigt die Demo das in der Ebene nicht: dort genügen Lorentz-Kegel der Größe 3).
Stück 2 der Reihe „Konvexe und konische Optimierung“, nach der kleinsten Kugel mit Zertifikat.

Weiter unten: **vier Experimente** (**🔬**), ein **🧰 Baukasten**, ein **🔍 Datenblick**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Kegelform", expanded=True):
    st.markdown(
        """
1. **Die Aufgabe:** Eine Punktwolke; gesucht ist die kleinste Form, die alle Punkte enthält (Würfel, Raute, Kugel oder Ellipse um den Ursprung).
2. **Die Umformung:** Jede Bedingung wird zu einer Zeile `s = b − A x` in einem Kegel. Eine Norm `‖v‖ ≤ R` wird zu `(R; v)` im Lorentz-Kegel, eine Matrixbedingung zu einem PSD-Kegel. Im Baukasten unten steht, wie.
3. **Der Löser:** Eine Barrierenmethode hält `s` im Inneren der Kegel und zieht den Strafterm schrittweise ab. Zu jedem zentrierten Punkt gibt sie eine **bewiesene Lücke** `ν/t` zum Optimum (bis auf Rundung bei sehr großem `t`), wie das Zertifikat aus Stück 1.
4. **Die Probe:** Das Ergebnis wird zurück in die Ausgangsaufgabe übersetzt und dort geprüft (Normen ≤ R, `xᵀPx ≤ 1`), nicht nur im Kegelprogramm.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Form**")
    form = st.selectbox("Gesuchte Form", A.FORMS, key="form_select", format_func=A.FORM_NAMES.get,
                        help="Welche der sechs Kegelformen im Hauptdiagramm gezeigt wird; die Tabellen vergleichen immer alle sechs.")
    st.markdown("**Punktwolke**")
    n = st.slider("Punkte n", *bounds("n_slider"), key="n_slider", help="Wie viele Punkte die Wolke hat.")
    stretch = st.slider("Streckung", *bounds("stretch_slider"), step=0.1, key="stretch_slider", help="Wie viel breiter die Wolke entlang ihrer Längsachse ist als quer dazu.")
    angle = st.slider("Drehung (Grad)", *bounds("angle_slider"), key="angle_slider", help="Winkel der Längsachse gegen die x-Achse.")
    seed = st.number_input("Zufalls-Seed der Wolke", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Wolke generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"form_select": form, "n_slider": int(n), "stretch_slider": float(stretch), "angle_slider": int(angle), "seed_input": int(seed)})

settings = Settings(form, int(n), float(stretch), int(angle), int(seed))
a = _analysis(settings)
rows = form_rows(a)
row = next(r for r in rows if r["form"] == form)
sel = a.selected

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if not row["ok"]:
    st.error("Die Rückumformung besteht nicht: " + "; ".join(sel.cone_problems + sel.original_problems))
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(row["objective"], de(row["value"], 4), help="Zielwert des Kegelprogramms für die gewählte Form.")
m2.metric("Variablen", row["variables"], help="Zahl der Unbekannten x des Kegelprogramms.")
m3.metric("Zeilen von A", row["rows"], help="Zahl der Zeilen von A: die Komponenten von s.")
m4.metric("Newton-Schritte", row["newton"], help="In dieser Implementierung (Barrierenmethode, Zielgenauigkeit 1e-9).")
m5.metric("Bewiesene Lücke", sci(row["gap"]), help="Das Optimum liegt höchstens so weit unter dem gefundenen Wert (Barrierenparameter geteilt durch t).")
st.plotly_chart(build_shape(a.cloud.points, form, sel.shape), width="stretch", key="main_chart")
st.caption(f"Kegel: {row['cones']}. Die Wolke ist um ihren Mittelwert verschoben; Würfel, Raute und Kugel haben einen frei gewählten Mittelpunkt, die Ellipsen liegen fest im Ursprung (dem Mittelwert).")

if st.button("📄 PDF der aktuellen Wolke erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="kegelform.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wird aus der Aufgabe ein Kegelprogramm, und stimmt es?")
verdict(verdict_form(row, a.ellipse_ratio))
verdict(verdict_hand(_hand()))
st.caption("Die zweite Zeile rechnet die Handinstanz (vier Punkte, im 📐-Expander vorgerechnet) mit allen sechs Formen nach.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Eine Wolke, sechs Formen")
st.caption("Dieselbe Wolke, sechs Kegelprogramme. Die Tabelle zeigt Zielwert, Größe, Kegel und Aufwand; rechts läuft die bewiesene Lücke der gewählten Form mit den Newton-Schritten.")
t1, t2 = st.columns([3, 2])
t1.plotly_chart(build_forms(rows), width="stretch", key="forms_chart")
t2.plotly_chart(build_gap(sel.result), width="stretch", key="gap_chart")
st.dataframe(pd.DataFrame([{"Form": r["name"], "Zielgröße": r["objective"], "Wert": round(r["value"], 4), "Variablen": r["variables"], "Zeilen": r["rows"],
                            "Kegel": r["cones"], "Newton-Schritte": r["newton"], "Lücke ≤": sci(r["gap"])} for r in rows]), hide_index=True, width="stretch")
verdict(verdict_forms(rows))

st.markdown("---")

st.subheader("🔬 Experiment 2: Wie viele LP-Zeilen ersetzt ein Lorentz-Kegel?")
st.caption(f"Die Kugelbedingung `‖x − c‖ ≤ R` wird durch `u·(x − c) ≤ R` für k Richtungen u ersetzt (ein LP, je Punkt k Zeilen, ohne Hilfsvariablen). {C.LP_POINTS} Punkte, Median über {len(C.LP_SEEDS)} feste Wolken, "
           "d = 2 mit regelmäßigem Vieleck, ab d = 3 mit Zufallsrichtungen. Rechnet einige Sekunden.")
if st.button("🔬 Experiment 2 rechnen", key="lp_run"):
    st.session_state["lp_res"] = _lp()
if st.session_state.get("lp_res"):
    st.plotly_chart(build_lp(st.session_state["lp_res"]), width="stretch", key="lp_chart")
    verdict(verdict_lp(st.session_state["lp_res"]))
else:
    st.info("Noch nicht gerechnet.")

st.markdown("---")

st.subheader("🔬 Experiment 3: Dieselbe Kugel als Lorentz-Kegel und als PSD-Kegel")
st.caption(f"Median über {len(C.SWEEP_SEEDS)} feste Wolken je Punktzahl. Die Zeit je Lösung ist auf diesem Server gemessen und gilt nur hier.")
psd_rows = _psd()
st.plotly_chart(build_psd(psd_rows), width="stretch", key="psd_chart")
st.dataframe(pd.DataFrame([{"Punkte": r["n"], "Zeilen (Lorentz)": r["rows_soc"], "Zeilen (PSD)": r["rows_psd"], "Newton-Schritte (Lorentz)": r["newton_soc"],
                            "Newton-Schritte (PSD)": r["newton_psd"], "Zeit Lorentz (ms)": round(1000 * r["seconds_soc"], 1), "Zeit PSD (ms)": round(1000 * r["seconds_psd"], 1)}
                           for r in psd_rows]), hide_index=True, width="stretch")
verdict(verdict_psd(psd_rows))

st.markdown("---")

st.subheader("🔬 Experiment 4: Was die Drehung bringt")
st.caption(f"Wolken, um den Faktor {de(C.ROT_STRETCH, 0)} in die Länge gezogen und um verschiedene Winkel gedreht, {settings.n} Punkte, {len(C.SWEEP_SEEDS)} feste Wolken je Winkel. "
           "Verglichen wird die Summe der quadrierten Halbachsen der allgemeinen (PSD) mit der achsenparallelen Ellipse (Lorentz). Rechnet einige Sekunden.")
if st.button("🔬 Experiment 4 rechnen", key="rot_run"):
    st.session_state["rot_res"] = (settings.n, _rotation(settings.n))
res_rot = st.session_state.get("rot_res")
if res_rot and res_rot[0] == settings.n:
    st.plotly_chart(build_rotation(res_rot[1]), width="stretch", key="rot_chart")
    verdict(verdict_rotation(res_rot[1], C.ROT_STRETCH))
else:
    st.info("Noch nicht gerechnet für diese Punktzahl.")

st.markdown("---")

with st.expander("🧰 Baukasten: welche Bedingung in welchen Kegel gehört"):
    st.markdown(
        r"""
| Bedingung | Kegel | Umformung |
|---|---|---|
| `a·x ≤ b`, `‖x‖∞ ≤ R` | Orthant | je Koordinate zwei Zeilen `R − xⱼ ≥ 0`, `R + xⱼ ≥ 0` |
| `‖x‖₁ ≤ R` | Orthant | Hilfsvariablen `tⱼ ≥ xⱼ` und `tⱼ ≥ −xⱼ` (zwei Zeilen) sowie `Σ tⱼ ≤ R` |
| `‖x‖₂ ≤ R` | Lorentz-Kegel | `(R; x)` im Kegel: `R ≥ ‖x‖₂` |
| `p·t ≥ 1` (Bruch `1/p ≤ t`) | Lorentz-Kegel (3) | `(p + t; 2, p − t)` im Kegel: `(p+t)² ≥ 4 + (p−t)²` |
| `‖v‖₂ ≤ R` als Matrix | PSD-Kegel | `[[R·I, v],[vᵀ, R]] ⪰ 0` (Schur-Komplement) |
| `T ⪰ P⁻¹` (für `trace(T)` minimieren) | PSD-Kegel | `[[P, I],[I, T]] ⪰ 0` |
| `λ_max(M) ≤ t` | PSD-Kegel | `t·I − M ⪰ 0` |
        """
    )

with st.expander("🔍 Datenblick: das Kegelprogramm für die ersten drei Punkte"):
    small = a.cloud.points[:C.DATA_VIEW_POINTS]
    dv = data_view(form, small)
    sizes = {"orthant": sum(m for t, m in dv["cones"] if t == "l"), "soc": [m for t, m in dv["cones"] if t == "q"], "psd": [m for t, m in dv["cones"] if t == "s"]}
    st.markdown(f"**{A.FORM_NAMES[form]}** mit {C.DATA_VIEW_POINTS} Punkten: {dv['A'].shape[1]} Variablen, {dv['A'].shape[0]} Zeilen, Kegel: {cones_text(sizes)}. "
                "Die Zeilen von `s = b − A x` stehen in der Reihenfolge der Kegel.")
    st.markdown("**Zielvektor c** und **Start x₀** (streng zulässig durch Konstruktion):")
    st.dataframe(pd.DataFrame({"c": dv["c"], "x₀": dv["x0"]}).T.round(4), width="stretch")
    st.markdown("**Matrix A** (Zeilen = Komponenten von s) und **Vektor b**:")
    table = pd.DataFrame(dv["A"], columns=[f"x{j + 1}" for j in range(dv["A"].shape[1])]).round(3)
    table.insert(0, "b", dv["b"].round(3))
    st.dataframe(table, width="stretch", height=min(40 + 35 * len(table), 420))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Start ist streng zulässig** | Jede Form liefert ihren Startpunkt durch Konstruktion mit (Würfel: Mittelwert und großer Radius, Ellipse: kleine Matrix). Für allgemeine Programme gibt es das nicht; ohne Startpunkt scheitert die Barrierenmethode. | Stück 6: Primal-Dual-Schritt und Einbettung |
| **Die Schrittzahlen gelten für diese Implementierung** | Gedämpftes Newton-Verfahren mit festem Faktor 8 für t; die Zahlen sind keine Eigenschaft der Kegel. | Stück 5: Barriere und zentraler Pfad |
| **Die LP-Nachbildung hat keine Hilfsvariablen** | Mit Hilfsvariablen lässt sich der Lorentz-Kegel durch ein LP von vertretbarer Größe beliebig genau annähern (Ben-Tal und Nemirovski, 2001); Experiment 2 zeigt nur die einfache Variante. | – |
| **Ellipsen um den Ursprung, Spurkriterium** | Eine Ellipse mit freiem Mittelpunkt oder mit Volumenziel (`log det`) ist kein reines SDP in dieser Form. | spätere Stücke |
| **Es gibt keine Dualität in dieser Demo** | Die Lückenschranke kommt aus der Barriere, nicht aus einem dualen Kegelprogramm. | Stück 4: Kegel-Dualität |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Kegelform.** $\min\; c^\top x$ unter $s = b - Ax \in K$, mit $K = K_1 \times \dots \times K_m$ und jedes $K_i$ einer von drei Kegeln:
Orthant $\{s \ge 0\}$, Lorentz-Kegel $\{(s_0, \bar s) : s_0 \ge \lVert \bar s \rVert_2\}$ und PSD-Kegel $\{S \succeq 0\}$ (als Vektor $\operatorname{svec}(S)$ mit Faktor $\sqrt{2}$ auf den Außerdiagonalen, sodass $\langle S, T\rangle = \operatorname{svec}(S)^\top \operatorname{svec}(T)$).

**Barrieren.** $-\sum_j \log s_j$ (Orthant, $\nu = m$), $-\log(s_0^2 - \lVert \bar s \rVert^2)$ (Lorentz, $\nu = 2$), $-\log\det S$ (PSD der Größe $k$, $\nu = k$).
Die Summe $\phi$ ist selbstkonkordant mit Parameter $\nu = \sum \nu_i$.

**Barrierenmethode.** Für $t > 0$ minimiere $f_t(x) = t\, c^\top x + \phi(b - Ax)$ mit Newton-Schritten; das Minimum $x^\star(t)$ liegt auf dem zentralen Pfad. Dort gilt $c^\top x^\star(t) - p^\star \le \nu / t$:
das ist die **bewiesene Lücke**. Die Demo multipliziert $t$ je Runde mit 8, bis $\nu / t \le 10^{-9}$.

**Die sechs Formen** für Punkte $x_i \in \mathbb{R}^d$, Variablen $x = (c, R)$ bzw. Matrizen:
- **Würfel** $\min R$, $R \mp (x_{ij} - c_j) \ge 0$: Orthant mit $2nd$ Zeilen.
- **Raute** $\min R$, $t_{ij} \mp (x_{ij} - c_j) \ge 0$, $R - \sum_j t_{ij} \ge 0$: Orthant mit $2nd + n$ Zeilen und $nd$ Hilfsvariablen.
- **Kugel** $\min R$, $(R; x_i - c) \in \mathcal{Q}^{d+1}$ für jedes $i$.
- **Kugel als SDP** $\begin{psmallmatrix} R\,I & x_i - c \\ (x_i - c)^\top & R \end{psmallmatrix} \succeq 0$: nach dem Schur-Komplement gleichwertig zu $R \ge \lVert x_i - c\rVert$, aber ein PSD-Kegel der Größe $d+1$ je Punkt.
- **Ellipse, achsenparallel** $\{x : \sum_j p_j x_j^2 \le 1\}$, $\min \sum_j t_j$ mit $p_j t_j \ge 1$ als gedrehter Lorentz-Kegel $(p_j + t_j;\, 2,\, p_j - t_j)$. Die Summe $\sum_j 1/p_j$ ist die der quadrierten Halbachsen.
- **Ellipse, allgemein** $\{x : x^\top P x \le 1\}$, $\min \operatorname{trace}(T)$ unter $x_i^\top P x_i \le 1$ und $\begin{psmallmatrix} P & I \\ I & T \end{psmallmatrix} \succeq 0$; dann ist $T \succeq P^{-1}$ und $\operatorname{trace}(T) \ge \operatorname{trace}(P^{-1})$, im Optimum gleich.

**Handinstanz** $(1,1), (-1,-1), (0{,}5; -0{,}5), (-0{,}5; 0{,}5)$. Würfel $R = 1$, Raute $R = 2$ (die ersten beiden Punkte haben $\ell_1$-Abstand 4), Kugel $R = \sqrt{2}$. Achsenparallele Ellipse: der erste Punkt verlangt $p_1 + p_2 \le 1$, also $\sum 1/p_j \ge 4$ mit Gleichheit bei $p_1 = p_2 = 1/2$.
Allgemeine Ellipse: in den Diagonalachsen $u = (1,1)/\sqrt{2}$, $v = (1,-1)/\sqrt{2}$ liegen die Punkte bei $\pm\sqrt{2}\,u$ und $\pm\tfrac{1}{\sqrt{2}}\,v$. Aus $u^\top P u \le \tfrac12$ folgt $u^\top P^{-1} u \ge 1/(u^\top P u) \ge 2$ (Jensen), aus $v^\top P v \le 2$ analog $v^\top P^{-1} v \ge \tfrac12$; also $\operatorname{trace}(P^{-1}) \ge 2{,}5$, erreicht mit Halbachsen $\sqrt{2}$ und $1/\sqrt{2}$.

**Literatur.** S. Boyd und L. Vandenberghe: *Convex Optimization*, Cambridge University Press 2004, Kapitel 11 (Barrierenmethode, zentraler Pfad, verallgemeinerte Ungleichungen). A. Ben-Tal und A. Nemirovski: *On polyhedral approximations of the second-order cone*, Mathematics of Operations Research 26(2), 2001, S. 193–205 (Annäherung des Lorentz-Kegels durch ein LP mit Hilfsvariablen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html)."
)
