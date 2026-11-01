"""PDF-Export der aktuellen Wolke mit den sechs Kegelprogrammen (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from kgf_evaluation import Analysis, de, form_rows, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "€": "EUR", "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3",
            "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Kegelform: dieselbe Aufgabe in sechs Kegelprogrammen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Wolke: {s.n} Punkte in der Ebene, Streckung {de(s.stretch, 1)}, um {s.angle} Grad gedreht, Seed {s.seed}; um den Mittelwert verschoben. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Form", "Zielgröße", "Wert", "Variablen", "Zeilen", "Kegel", "Newton", "Lücke <="], [48, 52, 22, 20, 16, 62, 16, 22]
    pdf.set_font("Helvetica", "B", 8)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for r in form_rows(a):
        for v, w in zip([r["name"], r["objective"], de(r["value"], 4), r["variables"], r["rows"], r["cones"], r["newton"], sci(r["gap"])], widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Kegelform: min c^T x mit s = b - A x im Kegel (Orthant, Lorentz-Kegel, PSD-Kegel); Löser: primale Barrierenmethode, Lücke nu/t. "
                            "Quelle: kegelform-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
