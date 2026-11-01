"""Kern der Demo: sechs Formen derselben Aufgabe (kleinste umschließende Form einer Punktwolke) als Kegelprogramm, ihre Rückumformung und die LP-Nachbildung der Kugel.

Jede Form ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b); `solve_form` ruft nur der Reihe nach auf. Der gemeinsame Kegel-Kern
steht in `kegel_kern.py` (wortgleich in allen Repos der Reihe).

Formen (Schlüssel):
  cube          Würfel (l-unendlich-Kugel): min R, |x_i - c|_j <= R                    Orthant
  diamond       Raute (l1-Kugel): min R, Summe_j t_ij <= R, t_ij >= |x_ij - c_j|       Orthant mit Hilfsvariablen
  ball_soc      Kugel: min R, ||x_i - c|| <= R                                         Lorentz-Kegel je Punkt
  ball_psd      dieselbe Kugel über das Schur-Komplement [[R I, v],[v^T, R]] >= 0      PSD-Kegel je Punkt
  ellipse_axis  Ellipse um den Ursprung, Achsen parallel zu den Koordinatenachsen      Orthant + gedrehte Lorentz-Kegel
  ellipse_psd   allgemeine Ellipse um den Ursprung                                     Orthant + ein PSD-Kegel
Ellipse: x^T P x <= 1, Ziel kleinste Summe der quadrierten Halbachsen = trace(P^-1); die Daten werden vorher um den Mittelwert verschoben."""

import math
from dataclasses import dataclass

import numpy as np

import kegel_kern as K

FORMS = ("cube", "diamond", "ball_soc", "ball_psd", "ellipse_axis", "ellipse_psd")
FORM_NAMES = {"cube": "Würfel (l-unendlich)", "diamond": "Raute (l1)", "ball_soc": "Kugel (Lorentz-Kegel)", "ball_psd": "Kugel (PSD-Kegel)",
              "ellipse_axis": "Ellipse, achsenparallel", "ellipse_psd": "Ellipse, allgemein"}
CONE_NAMES = {"cube": "Orthant", "diamond": "Orthant", "ball_soc": "Lorentz-Kegel", "ball_psd": "PSD-Kegel",
              "ellipse_axis": "Orthant + Lorentz-Kegel", "ellipse_psd": "Orthant + PSD-Kegel"}
ELLIPSES = ("ellipse_axis", "ellipse_psd")


@dataclass(frozen=True, eq=False)
class Shape:
    kind: str                       # 'box', 'diamond', 'ball', 'ellipse'
    center: np.ndarray
    radius: float = 0.0             # box, diamond, ball
    matrix: np.ndarray | None = None  # Ellipse: P mit x^T P x <= 1


def objective_name(form: str) -> str:
    return "Summe der quadrierten Halbachsen" if form in ELLIPSES else "Radius"


# ----------------------------------------------------------------------------- Aufbau der Kegelprogramme

def _start_radius(points: np.ndarray, c0: np.ndarray, norm: float) -> float:
    return 1.5 * float(np.linalg.norm(points - c0, ord=norm, axis=1).max()) + 1.0


def build_cube(points: np.ndarray) -> K.ConeProblem:
    """s = R -+ (x_ij - c_j) >= 0 für jede Koordinate und beide Vorzeichen: 2 n d Zeilen im Orthanten."""
    n, d = points.shape
    rows, b = [], []
    for sign in (1.0, -1.0):
        block = np.zeros((n * d, d + 1))
        block[:, d] = -1.0
        block[np.arange(n * d), np.tile(np.arange(d), n)] = -sign
        rows.append(block)
        b.append((-sign * points).reshape(-1))
    c = np.zeros(d + 1)
    c[d] = 1.0
    c0 = points.mean(axis=0)
    x0 = np.concatenate([c0, [_start_radius(points, c0, np.inf)]])
    return K.ConeProblem(c, np.vstack(rows), np.concatenate(b), (("l", 2 * n * d),), x0)


def build_diamond(points: np.ndarray) -> K.ConeProblem:
    """Hilfsvariablen t_ij >= |x_ij - c_j| (2 n d Zeilen) und R - Summe_j t_ij >= 0 (n Zeilen); Variablen c, R, t."""
    n, d = points.shape
    nv = d + 1 + n * d
    rows, b = [], []
    for sign in (1.0, -1.0):
        block = np.zeros((n * d, nv))
        idx = np.arange(n * d)
        block[idx, d + 1 + idx] = -1.0
        block[idx, np.tile(np.arange(d), n)] = -sign
        rows.append(block)
        b.append((-sign * points).reshape(-1))
    total = np.zeros((n, nv))
    total[:, d] = -1.0
    for i in range(n):
        total[i, d + 1 + i * d:d + 1 + (i + 1) * d] = 1.0
    rows.append(total)
    b.append(np.zeros(n))
    c = np.zeros(nv)
    c[d] = 1.0
    c0 = points.mean(axis=0)
    t0 = np.abs(points - c0) + 1.0
    x0 = np.concatenate([c0, [t0.sum(axis=1).max() + 1.0], t0.reshape(-1)])
    return K.ConeProblem(c, np.vstack(rows), np.concatenate(b), (("l", len(np.concatenate(b))),), x0)


def build_ball_soc(points: np.ndarray) -> K.ConeProblem:
    """Je Punkt s = (R; x_i - c) im Lorentz-Kegel der Größe d + 1."""
    n, d = points.shape
    A = np.zeros((n * (d + 1), d + 1))
    b = np.zeros(n * (d + 1))
    for i in range(n):
        r0 = i * (d + 1)
        A[r0, d] = -1.0
        A[r0 + 1:r0 + 1 + d, :d] = np.eye(d)
        b[r0 + 1:r0 + 1 + d] = points[i]
    c = np.zeros(d + 1)
    c[d] = 1.0
    c0 = points.mean(axis=0)
    x0 = np.concatenate([c0, [_start_radius(points, c0, 2)]])
    return K.ConeProblem(c, A, b, (("q", d + 1),) * n, x0)


def build_ball_psd(points: np.ndarray) -> K.ConeProblem:
    """Je Punkt die Matrix [[R I, v],[v^T, R]] mit v = x_i - c im PSD-Kegel der Größe d + 1 (Schur-Komplement: R >= ||v||)."""
    n, d = points.shape
    k = d + 1
    iu = np.triu_indices(k)
    dim = len(iu[0])
    A = np.zeros((n * dim, d + 1))
    b = np.zeros(n * dim)
    for i in range(n):
        for r, (a, c_) in enumerate(zip(*iu)):
            f = 1.0 if a == c_ else K.SQ2
            row = i * dim + r
            if a == c_:
                A[row, d] = -1.0                        # Diagonale: R
            elif c_ == d:
                A[row, a] = f * 1.0                     # Eintrag (a, d) = x_ia - c_a  =>  s = f (x_ia - c_a) = b - A x mit A = f
                b[row] = f * points[i, a]
    c = np.zeros(d + 1)
    c[d] = 1.0
    c0 = points.mean(axis=0)
    x0 = np.concatenate([c0, [_start_radius(points, c0, 2)]])
    return K.ConeProblem(c, A, b, (("s", k),) * n, x0)


def build_ellipse_axis(points: np.ndarray) -> K.ConeProblem:
    """Variablen p_1..p_d (p_j = 1/Halbachse_j^2) und t_1..t_d; Zeilen 1 - Summe_j p_j x_ij^2 >= 0; t_j >= 1/p_j als gedrehter Lorentz-Kegel (p+t; 2, p-t) in Q^3."""
    n, d = points.shape
    nv = 2 * d
    A = [np.hstack([points ** 2, np.zeros((n, d))])]
    b = [np.ones(n)]
    for j in range(d):
        blk = np.zeros((3, nv))
        blk[0, j] = blk[0, d + j] = -1.0
        blk[2, j], blk[2, d + j] = -1.0, 1.0
        A.append(blk)
        b.append(np.array([0.0, 2.0, 0.0]))
    c = np.zeros(nv)
    c[d:] = 1.0
    alpha = 0.5 / max(float((points ** 2).sum(axis=1).max()), 1e-9)
    x0 = np.concatenate([np.full(d, alpha), np.full(d, 2.0 / alpha)])
    return K.ConeProblem(c, np.vstack(A), np.concatenate(b), (("l", n),) + (("q", 3),) * d, x0)


def build_ellipse_psd(points: np.ndarray) -> K.ConeProblem:
    """Variablen svec(P), svec(T); Zeilen 1 - <x x^T, P> >= 0; [[P, I],[I, T]] im PSD-Kegel der Größe 2d (T >= P^-1); Ziel trace(T)."""
    n, d = points.shape
    iu = np.triu_indices(d)
    m = len(iu[0])
    nv = 2 * m
    pos = {(a, c_): r for r, (a, c_) in enumerate(zip(*iu))}
    # 1 - <x x^T, P> = 1 - sum_(a<=c) w_ac x_a x_c P_ac mit w = 1 (Diagonale) bzw. 2 (Außerdiagonale); P_ac = svec_r / sqrt 2 außerhalb der Diagonale
    cons = np.zeros((n, nv))
    for r, (a, c_) in enumerate(zip(*iu)):
        cons[:, r] = points[:, a] * points[:, c_] * (1.0 if a == c_ else K.SQ2)
    k = 2 * d
    iuk = np.triu_indices(k)
    dimk = len(iuk[0])
    A_psd = np.zeros((dimk, nv))
    b_psd = np.zeros(dimk)
    for r, (a, c_) in enumerate(zip(*iuk)):
        if a < d and c_ < d:                            # P-Block: svec-Koordinate des Blocks entspricht der der Variable (gleiche Skalierung)
            A_psd[r, pos[(a, c_)]] = -1.0
        elif a >= d and c_ >= d:                        # T-Block
            A_psd[r, m + pos[(a - d, c_ - d)]] = -1.0
        elif c_ == a + d:                               # Einheitsmatrix oben rechts: Eintrag 1, svec-Faktor sqrt 2
            b_psd[r] = K.SQ2
    c = np.zeros(nv)
    for r, (a, c_) in enumerate(zip(*iu)):
        if a == c_:
            c[m + r] = 1.0
    alpha = 0.5 / max(float((points ** 2).sum(axis=1).max()), 1e-9)
    x0 = np.zeros(nv)
    for r, (a, c_) in enumerate(zip(*iu)):
        if a == c_:
            x0[r], x0[m + r] = alpha, 2.0 / alpha
    return K.ConeProblem(c, np.vstack([cons, A_psd]), np.concatenate([np.ones(n), b_psd]), (("l", n), ("s", k)), x0)


BUILDERS = {"cube": build_cube, "diamond": build_diamond, "ball_soc": build_ball_soc, "ball_psd": build_ball_psd,
            "ellipse_axis": build_ellipse_axis, "ellipse_psd": build_ellipse_psd}


def build_form(form: str, points: np.ndarray) -> K.ConeProblem:
    if form not in BUILDERS:
        raise ValueError(f"unbekannte Form {form!r}")
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or len(pts) < 1:
        raise ValueError("Punkte müssen eine nicht leere Matrix (n, d) sein")
    return BUILDERS[form](pts)


# ----------------------------------------------------------------------------- Rückumformung

def decode(form: str, points: np.ndarray, x: np.ndarray) -> Shape:
    """Lösungsvektor x zurück in die Form der Ausgangsaufgabe (Mittelpunkt und Radius bzw. Matrix P)."""
    d = points.shape[1]
    if form in ("cube", "diamond", "ball_soc", "ball_psd"):
        kind = {"cube": "box", "diamond": "diamond"}.get(form, "ball")
        return Shape(kind, np.array(x[:d]), float(x[d]))
    if form == "ellipse_axis":
        return Shape("ellipse", np.zeros(d), matrix=np.diag(x[:d]))
    return Shape("ellipse", np.zeros(d), matrix=K.smat(x[:d * (d + 1) // 2], d))


def original_violations(form: str, points: np.ndarray, x: np.ndarray, value: float, tol: float = 1e-6) -> list[str]:
    """Prüft das Ergebnis in der AUSGANGSAUFGABE (nicht im Kegelprogramm): Normen <= R bzw. x^T P x <= 1, P positiv definit, Zielwert passt."""
    shape = decode(form, points, x)
    problems = []
    if shape.kind != "ellipse":
        ordnung = {"box": np.inf, "diamond": 1, "ball": 2}[shape.kind]
        dist = np.linalg.norm(points - shape.center, ord=ordnung, axis=1)
        if dist.max() > shape.radius * (1.0 + tol) + tol:
            problems.append(f"ein Punkt liegt außerhalb (Abstand {dist.max():.6g} > Radius {shape.radius:.6g})")
        if abs(shape.radius - value) > tol * max(1.0, abs(value)):
            problems.append("Radius und Zielwert stimmen nicht überein")
        return problems
    P = shape.matrix
    eig = np.linalg.eigvalsh(P)
    if eig.min() <= 0:
        problems.append("P ist nicht positiv definit")
        return problems
    quad = np.einsum("ij,jk,ik->i", points, P, points)
    if quad.max() > 1.0 + tol:
        problems.append(f"ein Punkt liegt außerhalb der Ellipse (x^T P x = {quad.max():.6g})")
    semi_sq = float((1.0 / eig).sum())
    if semi_sq > value * (1.0 + tol) + tol:
        problems.append("Summe der quadrierten Halbachsen liegt über dem Zielwert")
    return problems


@dataclass(frozen=True, eq=False)
class FormResult:
    form: str
    problem: K.ConeProblem
    result: K.BarrierResult
    shape: Shape
    cone_problems: list         # Kegel, in denen s = b - A x nicht liegt
    original_problems: list     # Verletzungen in der Ausgangsaufgabe


def solve_form(form: str, points: np.ndarray, tol: float = 1e-9) -> FormResult:
    problem = build_form(form, points)
    result = K.solve_barrier(problem, tol=tol)
    pts = np.asarray(points, dtype=float)
    return FormResult(form, problem, result, decode(form, pts, result.x), K.check_solution(problem, result.x),
                      original_violations(form, pts, result.x, result.value))


# ----------------------------------------------------------------------------- LP-Nachbildung der Kugel

def directions(d: int, k: int, seed: int = 0) -> np.ndarray:
    """k Einheitsrichtungen: in der Ebene ein regelmäßiges Vieleck (die beste Wahl), ab d = 3 gleichverteilte Zufallsrichtungen."""
    if k < 1:
        raise ValueError("mindestens eine Richtung")
    if d == 2:
        t = 2.0 * math.pi * (np.arange(k) + 0.5) / k
        return np.stack([np.cos(t), np.sin(t)], axis=1)
    u = np.random.default_rng([seed, d, k]).normal(size=(k, d))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def build_ball_polyhedral(points: np.ndarray, dirs: np.ndarray) -> K.ConeProblem:
    """Die Kugelbedingung ||x_i - c|| <= R ersetzt durch u_j . (x_i - c) <= R für alle Richtungen u_j und R >= 0: n k + 1 Zeilen im Orthanten (ein LP)."""
    n, d = points.shape
    k = len(dirs)
    A = np.empty((n * k + 1, d + 1))
    A[:n * k, :d] = -np.tile(dirs, (n, 1))
    A[:n * k, d] = -1.0
    A[n * k, :] = 0.0
    A[n * k, d] = -1.0                                  # R >= 0: ohne genug Richtungen ist max_j u_j . v keine Norm und das LP unbeschränkt
    b = np.concatenate([-(dirs @ points.T).T.reshape(-1), [0.0]])   # Zeile (i, j): -u_j . x_i
    c = np.zeros(d + 1)
    c[d] = 1.0
    c0 = points.mean(axis=0)
    x0 = np.concatenate([c0, [_start_radius(points, c0, 2)]])
    return K.ConeProblem(c, A, b, (("l", n * k + 1),), x0)


def polyhedral_radius(points: np.ndarray, dirs: np.ndarray, tol: float = 1e-9) -> float:
    """Radius der LP-Nachbildung (immer höchstens der wahre Radius, denn max_j u_j . v <= ||v||)."""
    return K.solve_barrier(build_ball_polyhedral(np.asarray(points, dtype=float), dirs), tol=tol).value
