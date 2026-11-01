"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import kgf_algorithm as A

DEFAULT_FORM = "ball_soc"
N_MIN, N_MAX, DEFAULT_N = 8, 30, 20           # PSD-Formen: O(n) Kegel der Größe 3 bis 4; Aufwand je Newton-Schritt wächst mit n
STRETCH_MIN, STRETCH_MAX, DEFAULT_STRETCH = 1.2, 6.0, 4.0   # Verhältnis der Streuung entlang zur quer zur Längsachse; ab 1,2 wirkt der Winkel immer
ANGLE_MIN, ANGLE_MAX, DEFAULT_ANGLE = 0, 90, 45             # Drehung der Längsachse gegen die x-Achse in Grad
SEED_MAX, DEFAULT_SEED = 999999, 35
SWEEP_SEEDS = tuple(range(100000, 100005))   # feste Wolken für alle Sweeps: nie ein einzelner Seed als Befund

LP_KS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)   # Zahl der Richtungen; n k Zeilen, O(n k d^2) je Newton-Schritt
LP_DIMS = (2, 3, 4)
LP_POINTS = 20
LP_SEEDS = SWEEP_SEEDS[:3]
LP_TARGET = 0.01                              # relativer Fehler des Radius, ab dem die Nachbildung „gut genug“ heißt
ROT_ANGLES = (0, 15, 30, 45, 60, 75, 90)
ROT_STRETCH = 4.0
PSD_SIZES = (6, 10, 15, 20, 25, 30)
DATA_VIEW_POINTS = 3                          # Punkte im Datenblick

COLORS = {"cube": "#4c78a8", "diamond": "#f58518", "ball_soc": "#2F6B65", "ball_psd": "#7a9e9a", "ellipse_axis": "#c0504d", "ellipse_psd": "#8a2b27"}
COLOR_POINT = "rgba(150,150,150,0.85)"
COLOR_REF = "#444444"

PRESETS = {
    "Standardfall: Kugel (Lorentz-Kegel)": {"form": DEFAULT_FORM, "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
    "Die Raute (LP)": {"form": "diamond", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
    "Dieselbe Kugel als SDP": {"form": "ball_psd", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
    "Achsenparallel verfehlt die Drehung": {"form": "ellipse_axis", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
    "Allgemeine Ellipse findet sie": {"form": "ellipse_psd", "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall: Kugel (Lorentz-Kegel)": "20 Punkte, in die Länge gezogen und um 45 Grad gedreht; die kleinste Kugel als ein Lorentz-Kegel je Punkt.",
    "Die Raute (LP)": "Dieselbe Wolke mit der l1-Kugel: ein reines LP, aber mit Hilfsvariablen für die Beträge.",
    "Dieselbe Kugel als SDP": "Gleicher Radius wie die Kugel im Lorentz-Kegel, aber jede Bedingung als 3 x 3-Matrix im PSD-Kegel: mehr Zeilen, gleiche Lösung.",
    "Achsenparallel verfehlt die Drehung": "Die Ellipse muss mit den Koordinatenachsen ausgerichtet bleiben und wird für die um 45 Grad gedrehte Wolke deutlich größer.",
    "Allgemeine Ellipse findet sie": "Die allgemeine Ellipse (ein PSD-Kegel) dreht sich mit der Wolke und braucht nur etwa die halbe Summe der quadrierten Halbachsen.",
}
assert set(PRESETS) == set(PRESET_HELP) and all(p["form"] in A.FORMS for p in PRESETS.values())
