"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import kgf_algorithm as A
import kgf_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def outline(shape: A.Shape) -> tuple[np.ndarray, np.ndarray]:
    """Umriss der Form als Linienzug (x, y): Würfel = Quadrat, Raute = Quadrat auf der Spitze, Kugel = Kreis, Ellipse aus x^T P x = 1."""
    t = np.linspace(0.0, 2.0 * math.pi, 241)
    cx, cy = shape.center
    if shape.kind == "ball":
        return cx + shape.radius * np.cos(t), cy + shape.radius * np.sin(t)
    if shape.kind == "box":
        r = shape.radius
        return np.array([cx - r, cx + r, cx + r, cx - r, cx - r]), np.array([cy - r, cy - r, cy + r, cy + r, cy - r])
    if shape.kind == "diamond":
        r = shape.radius
        return np.array([cx + r, cx, cx - r, cx, cx + r]), np.array([cy, cy + r, cy, cy - r, cy])
    eig, vec = np.linalg.eigh(shape.matrix)            # Halbachsen 1/sqrt(Eigenwert) entlang der Eigenvektoren
    pts = vec @ (np.stack([np.cos(t), np.sin(t)]) / np.sqrt(eig)[:, None])
    return cx + pts[0], cy + pts[1]


def build_shape(points: np.ndarray, form: str, shape: A.Shape):
    """Punktwolke und die gefundene Form (Farbe je Form), gleicher Maßstab auf beiden Achsen."""
    fig = go.Figure()
    xs, ys = outline(shape)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=A.FORM_NAMES[form], line=dict(color=C.COLORS[form], width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers", name="Punkte", marker=dict(size=8, color=C.COLOR_POINT, line=dict(width=1, color="#444")),
                             hovertemplate="(%{x:.2f}; %{y:.2f})<extra></extra>"))
    fig.add_trace(go.Scatter(x=[shape.center[0]], y=[shape.center[1]], mode="markers", name="Mittelpunkt", marker=dict(symbol="x", size=11, color=C.COLORS[form])))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y", scaleanchor="x", scaleratio=1)
    return _base(fig, 440)


def build_forms(rows: list[dict]):
    """Zeilen des Kegelprogramms je Form (Balken, Farbe je Form) mit der Zahl der Variablen als Beschriftung."""
    fig = go.Figure(go.Bar(x=[r["name"] for r in rows], y=[r["rows"] for r in rows], marker_color=[C.COLORS[r["form"]] for r in rows],
                           text=[f"{r['variables']} Var." for r in rows], textposition="outside", hovertemplate="%{x}: %{y} Zeilen<extra></extra>"))
    fig.update_yaxes(title_text="Zeilen des Kegelprogramms")
    fig.update_layout(showlegend=False)
    return _base(fig)


def build_gap(result):
    """Lückenschranke nu / t über die Newton-Schritte (logarithmisch): sie fällt mit jeder Runde um den Faktor 8."""
    steps = [h[1] for h in result.history]
    gaps = [h[2] for h in result.history]
    fig = go.Figure(go.Scatter(x=steps, y=gaps, mode="lines+markers", line=dict(color=C.COLOR_REF, width=2), name="Lückenschranke ν/t"))
    fig.update_xaxes(title_text="Newton-Schritte bisher")
    fig.update_yaxes(title_text="bewiesene Lücke (logarithmisch)", type="log")
    fig.update_layout(showlegend=False)
    return _base(fig, 300)


def build_lp(rows: list[dict]):
    """Relativer Fehler des Radius der LP-Nachbildung über die Zahl der Richtungen je Dimension (doppelt logarithmisch), dazu die 1-%-Marke."""
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Scatter(x=r["ks"], y=[max(e, 1e-9) for e in r["errors"]], mode="lines+markers", name=f"d = {r['d']}"))
    ks = rows[0]["ks"]
    fig.add_trace(go.Scatter(x=[ks[0], ks[-1]], y=[C.LP_TARGET, C.LP_TARGET], mode="lines", name="1 %", line=dict(color=C.COLOR_REF, dash="dash", width=1.5)))
    fig.update_xaxes(title_text="Richtungen k je Punkt (logarithmisch)", type="log")
    fig.update_yaxes(title_text="relativer Fehler des Radius (logarithmisch)", type="log")
    return _base(fig)


def build_psd(rows: list[dict]):
    """Zeilen des Kegelprogramms für dieselbe Kugel als Lorentz-Kegel und als PSD-Kegel über die Punktzahl."""
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["rows_soc"] for r in rows], mode="lines+markers", name="Lorentz-Kegel", line=dict(color=C.COLORS["ball_soc"], width=3)))
    fig.add_trace(go.Scatter(x=ns, y=[r["rows_psd"] for r in rows], mode="lines+markers", name="PSD-Kegel", line=dict(color=C.COLORS["ball_psd"], width=3)))
    fig.update_xaxes(title_text="Punkte n")
    fig.update_yaxes(title_text="Zeilen des Kegelprogramms")
    return _base(fig)


def build_rotation(rows: list[dict]):
    """Verhältnis allgemein / achsenparallel über den Drehwinkel: Median mit Band (Minimum bis Maximum)."""
    xs = [r["angle"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[r["max"] for r in rows] + [r["min"] for r in rows][::-1], fill="toself", fillcolor="rgba(138,43,39,0.18)",
                             line=dict(width=0), hoverinfo="skip", name="Minimum bis Maximum"))
    fig.add_trace(go.Scatter(x=xs, y=[r["median"] for r in rows], mode="lines+markers", name="Median", line=dict(color=C.COLORS["ellipse_psd"], width=3),
                             hovertemplate="%{x}°: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[xs[0], xs[-1]], y=[1.0, 1.0], mode="lines", name="gleich groß", line=dict(color=C.COLOR_REF, dash="dash", width=1.5)))
    fig.update_xaxes(title_text="Drehung der Wolke (Grad)")
    fig.update_yaxes(title_text="allgemein geteilt durch achsenparallel", rangemode="tozero")
    return _base(fig)
