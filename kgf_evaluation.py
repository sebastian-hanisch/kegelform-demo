"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)."""

import math
import time
from dataclasses import dataclass
from statistics import median

import numpy as np

import kegel_kern as K
import kgf_algorithm as A
import kgf_constants as C
from kgf_scenario import Cloud, make_cloud, make_gaussian


@dataclass(frozen=True)
class Settings:
    form: str = C.DEFAULT_FORM
    n: int = C.DEFAULT_N
    stretch: float = C.DEFAULT_STRETCH
    angle: int = C.DEFAULT_ANGLE
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True, eq=False)
class Analysis:
    settings: Settings
    cloud: Cloud
    results: dict            # Form -> FormResult, alle sechs Formen für dieselbe Wolke

    @property
    def selected(self) -> A.FormResult:
        return self.results[self.settings.form]

    @property
    def ellipse_ratio(self) -> float:
        """Summe der quadrierten Halbachsen: allgemeine geteilt durch achsenparallele Ellipse (höchstens 1)."""
        return self.results["ellipse_psd"].result.value / self.results["ellipse_axis"].result.value


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(frac: float) -> str:
    """Anteil als Prozentzahl: ab 1 % mit einer Nachkommastelle, darunter mit zwei."""
    p = 100.0 * frac
    return de(p, 1 if p >= 1.0 else 2) + " %"


_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,0·10⁻¹⁰."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def cones_text(sizes: dict) -> str:
    """Kegelliste in Worten, z. B. „20 × Lorentz-Kegel (3)“ oder „Orthant (80) + PSD-Kegel (4)“."""
    parts = []
    if sizes["orthant"]:
        parts.append(f"Orthant ({sizes['orthant']})")
    for label, key in (("Lorentz-Kegel", "soc"), ("PSD-Kegel", "psd")):
        counts: dict = {}
        for m in sizes[key]:
            counts[m] = counts.get(m, 0) + 1
        for m, cnt in counts.items():
            parts.append(f"{cnt} × {label} ({m})" if cnt > 1 else f"{label} ({m})")
    return " + ".join(parts)


# ----------------------------------------------------------------------------- Analyse der eingestellten Wolke

def analyse(settings: Settings) -> Analysis:
    cloud = make_cloud(settings.n, settings.stretch, settings.angle, settings.seed)
    return Analysis(settings, cloud, {f: A.solve_form(f, cloud.points) for f in A.FORMS})


def form_rows(a: Analysis) -> list[dict]:
    """Eine Zeile je Form: Zielwert, Größe des Kegelprogramms, Newton-Schritte, bewiesene Lücke, Prüfer."""
    rows = []
    for f in A.FORMS:
        r = a.results[f]
        s = r.problem.sizes()
        rows.append({"form": f, "name": A.FORM_NAMES[f], "objective": A.objective_name(f), "value": r.result.value, "variables": s["variables"], "rows": s["rows"],
                     "cones": cones_text(s), "newton": r.result.newton_steps, "gap": r.result.gap_bound, "ok": not r.cone_problems and not r.original_problems})
    return rows


# ----------------------------------------------------------------------------- Experiment 2: LP-Nachbildung der Kugel

def lp_experiment(dims=C.LP_DIMS, ks=C.LP_KS, n: int = C.LP_POINTS, seeds=C.LP_SEEDS) -> list[dict]:
    """Je Dimension: Median des relativen Fehlers (R_Kugel - R_LP) / R_Kugel über die festen Wolken, für wachsende Zahl k von Richtungen; dazu das erste k unter LP_TARGET."""
    rows = []
    for d in dims:
        clouds = [make_gaussian(n, d, seed) for seed in seeds]
        exact = [A.solve_form("ball_soc", pts).result.value for pts in clouds]
        errors = []
        for k in ks:
            dirs = A.directions(d, k)
            errors.append(float(median((r - A.polyhedral_radius(pts, dirs)) / r for pts, r in zip(clouds, exact))))
        first = next((k for k, e in zip(ks, errors) if e <= C.LP_TARGET), None)
        rows.append({"d": d, "ks": list(ks), "errors": errors, "first": first})
    return rows


# ----------------------------------------------------------------------------- Experiment 3: dieselbe Kugel als Lorentz-Kegel und als PSD-Kegel

def psd_experiment(sizes=C.PSD_SIZES, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Je Punktzahl n: Zeilen, Newton-Schritte (Median) und größte Abweichung der beiden Radien; die Zeit je Form wird gemessen und nur angezeigt."""
    rows = []
    for n in sizes:
        steps = {"ball_soc": [], "ball_psd": []}
        dev, seconds = 0.0, {"ball_soc": [], "ball_psd": []}
        size_rows = {}
        for seed in seeds:
            pts = make_cloud(n, 3.0, 30, seed).points
            vals = {}
            for f in ("ball_soc", "ball_psd"):
                t0 = time.perf_counter()
                r = A.solve_form(f, pts)
                seconds[f].append(time.perf_counter() - t0)
                steps[f].append(r.result.newton_steps)
                vals[f] = r.result.value
                size_rows[f] = r.problem.sizes()["rows"]
            dev = max(dev, abs(vals["ball_soc"] - vals["ball_psd"]) / vals["ball_soc"])
        rows.append({"n": n, "rows_soc": size_rows["ball_soc"], "rows_psd": size_rows["ball_psd"], "newton_soc": float(median(steps["ball_soc"])),
                     "newton_psd": float(median(steps["ball_psd"])), "max_dev": dev, "seconds_soc": float(median(seconds["ball_soc"])), "seconds_psd": float(median(seconds["ball_psd"]))})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: Drehung

def rotation_experiment(angles=C.ROT_ANGLES, stretch: float = C.ROT_STRETCH, n: int = C.DEFAULT_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Je Drehwinkel: Verhältnis der Summe der quadrierten Halbachsen (allgemeine / achsenparallele Ellipse) über die festen Wolken (Median, Minimum, Maximum)."""
    rows = []
    for angle in angles:
        ratios = []
        for seed in seeds:
            pts = make_cloud(n, stretch, angle, seed).points
            ratios.append(A.solve_form("ellipse_psd", pts).result.value / A.solve_form("ellipse_axis", pts).result.value)
        rows.append({"angle": angle, "median": float(median(ratios)), "min": min(ratios), "max": max(ratios)})
    return rows


# ----------------------------------------------------------------------------- Datenblick

def data_view(form: str, points: np.ndarray) -> dict:
    """Die Daten (A, b, Kegelliste) eines kleinen Kegelprogramms für den Datenblick; ohne Lösen."""
    p = A.build_form(form, points)
    return {"A": p.A, "b": p.b, "c": p.c, "cones": p.cones, "x0": p.x0}


# ----------------------------------------------------------------------------- Verdikte (rein)

def verdict_form(row: dict, ratio: float) -> tuple[str, str]:
    """Kernfrage der Primäransicht: ein Kegelprogramm aus dieser Form, und ob es besteht."""
    base = (f"Die {row['name']} wird zu **einem Kegelprogramm mit {row['variables']} Variablen und {row['rows']} Zeilen** ({row['cones']}). "
            f"Der Löser findet {row['objective']} = **{de(row['value'], 4)}** und beweist dabei eine Lücke von höchstens {sci(row['gap'])}.")
    if row["form"] in A.ELLIPSES:
        base += f" Für diese Wolke braucht die allgemeine Ellipse {pct(ratio)} der Summe der achsenparallelen."
    if not row["ok"]:
        return "error", base + " Die Rückumformung besteht nicht: Das darf nicht passieren."
    return "success", base + " Die Rückumformung besteht (s liegt im Kegel, die Ausgangsaufgabe ist erfüllt)."


def verdict_forms(rows: list[dict]) -> tuple[str, str]:
    """Experiment 1: dieselbe Wolke, sechs Formen."""
    by = {r["form"]: r for r in rows}
    return "info", (f"Dieselbe Wolke gibt sechs Kegelprogramme von {min(r['rows'] for r in rows)} bis {max(r['rows'] for r in rows)} Zeilen. "
                    f"Kugel im Lorentz-Kegel und Kugel im PSD-Kegel haben **denselben Radius** ({de(by['ball_soc']['value'], 4)}), der PSD-Kegel braucht {by['ball_psd']['rows']} statt {by['ball_soc']['rows']} Zeilen. "
                    f"Der Würfel ({de(by['cube']['value'], 3)}) ist enger als die Kugel, die Raute ({de(by['diamond']['value'], 3)}) weiter, wie es `‖v‖∞ ≤ ‖v‖₂ ≤ ‖v‖₁` verlangt.")


def verdict_lp(rows: list[dict]) -> tuple[str, str]:
    """Experiment 2: wie viele Halbräume die Kugel in LP-Form braucht."""
    parts = []
    for r in rows:
        parts.append(f"d = {r['d']}: " + (f"{r['first']} Richtungen" if r["first"] else f"mehr als {r['ks'][-1]} Richtungen (Fehler {pct(r['errors'][-1])} bei {r['ks'][-1]})"))
    return "warning", ("Damit die LP-Nachbildung der Kugel höchstens " + pct(C.LP_TARGET) + " beim Radius danebenliegt, braucht jeder Punkt " + "; ".join(parts) +
                       ". Ein einziger Lorentz-Kegel je Punkt ersetzt das exakt. Ab d = 3 sind die Richtungen zufällig gezogen (eine bessere Verteilung braucht weniger, wächst aber ähnlich).")


def verdict_psd(rows: list[dict]) -> tuple[str, str]:
    """Experiment 3: Lorentz-Kegel gegen PSD-Kegel für dieselbe Kugel."""
    worst = max(r["max_dev"] for r in rows)
    first, last = rows[0], rows[-1]
    return "info", (f"Beide Formen geben denselben Radius (größte Abweichung {worst:.1e}). Der PSD-Kegel braucht bei n = {last['n']} Punkten {last['rows_psd']} statt {last['rows_soc']} Zeilen "
                    f"und je Schritt eine größere Hesse-Matrix; die Zahl der Newton-Schritte ist ähnlich ({de(last['newton_soc'], 0)} gegen {de(last['newton_psd'], 0)}). "
                    f"Der SDP-Kegel kann mehr (Stück 6 der Reihe), kostet aber auch dort mehr, wo er nichts Zusätzliches ausdrückt.")


def verdict_rotation(rows: list[dict], stretch: float) -> tuple[str, str]:
    """Experiment 4: was die Drehung der allgemeinen Ellipse bringt."""
    best = min(rows, key=lambda r: r["median"])
    flat = max(rows, key=lambda r: r["median"])
    return "info", (f"Bei einer um den Faktor {de(stretch, 1)} gestreckten Wolke braucht die allgemeine Ellipse im Median **{pct(best['median'])}** der Summe der achsenparallelen, "
                    f"wenn die Wolke um {best['angle']}° gedreht ist, aber {pct(flat['median'])} bei {flat['angle']}°. Sie ist nie schlechter, weil die achsenparallele Ellipse nur ein Sonderfall ist.")


def verdict_hand(values: dict) -> tuple[str, str]:
    """Handinstanz: alle sechs Werte gegen die Handrechnung."""
    expected = {"cube": 1.0, "diamond": 2.0, "ball_soc": math.sqrt(2), "ball_psd": math.sqrt(2), "ellipse_axis": 4.0, "ellipse_psd": 2.5}
    off = {f: abs(values[f] - expected[f]) for f in expected}
    if max(off.values()) < 1e-6:
        return "success", "Alle sechs Werte stimmen mit der Handrechnung überein (1 · 2 · √2 · √2 · 4 · 2,5)."
    return "error", "Ein Wert weicht von der Handrechnung ab: " + ", ".join(f"{A.FORM_NAMES[f]} ({off[f]:.1e})" for f in off if off[f] >= 1e-6)
