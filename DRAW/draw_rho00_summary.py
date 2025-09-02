#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recreate the Eur. Phys. J. C16 (2000) summary figure of helicity matrix element rho_00
using PyROOT. This draws horizontal error-bar points for several vector mesons with
left labels (particle names) and right labels (experiment/selection). A vertical dashed
line at rho_00 = 1/3 is shown.

Usage:
  python3 Draw/draw_rho00_summary.py            # saves PNG/PDF under output/other/

Notes:
  - The numerical values below are reasonable approximations taken from the visual figure
    and are meant to reproduce style. Replace with your exact numbers if needed.
"""

from __future__ import annotations
import os
import math
import array

import ROOT as R


def build_graph(points, marker_style=20, marker_color=1, fill_style=0):
    """
    points: list of dicts with keys x, exl, exh, y
    Returns TGraphAsymmErrors with horizontal bars only.
    """
    n = len(points)
    gx = array.array('d', [p['x'] for p in points])
    gy = array.array('d', [p['y'] for p in points])
    gexl = array.array('d', [p.get('exl', p.get('ex', 0.0)) for p in points])
    gexh = array.array('d', [p.get('exh', p.get('ex', 0.0)) for p in points])
    geyl = array.array('d', [0.0]*n)
    geyh = array.array('d', [0.0]*n)
    g = R.TGraphAsymmErrors(n, gx, gy, gexl, gexh, geyl, geyh)
    g.SetMarkerStyle(marker_style)
    g.SetMarkerColor(marker_color)
    g.SetLineColor(marker_color)
    if fill_style:
        g.SetFillStyle(fill_style)
    g.SetLineWidth(2)
    g.SetMarkerSize(1.5)
    return g


def draw(save_dir: str = "./", basename: str = "rho00_summary"):
    R.gROOT.SetBatch(True)
    R.gStyle.SetOptStat(0)
    # Global font: bold Times New Roman everywhere
    # ROOT font code format: 10*font + precision (2 -> TrueType). 22 ≈ Times-Bold with precision 2.
    FONT_TNR_BOLD = 22
    R.gStyle.SetTextFont(FONT_TNR_BOLD)
    R.gStyle.SetLabelFont(FONT_TNR_BOLD, "XYZ")
    R.gStyle.SetTitleFont(FONT_TNR_BOLD, "XYZ")
    R.gStyle.SetPadTickX(1)
    R.gStyle.SetPadTickY(0)
    R.gStyle.SetNdivisions(510, "X")
    R.gStyle.SetNdivisions(1200, "Y")

    # Roughly reproduce the entries from the figure (x, +/- error, label left, label right)
    # y=0 at bottom; we'll draw from top to bottom, so largest y at top.
    # The y spacing determines vertical layout. Keep gaps between groups.
    y = 0
    rows = []
    def add(label_left, x, ex, label_right, style="filled"):
        nonlocal y
        rows.append({
            'y': y,
            'label_left': label_left,
            'x': x,
            'ex': ex,
            'label_right': label_right,
            'style': style,
        })
        y += 1

    # Top to bottom (reverse later for drawing)
    add("#rho(770)^{#pm}", 0.373, 0.052,  "OPAL 0.3<x<0.6", "filled")
    add("#rho(770)^{0}",    0.43, 0.05,  "DELPHI x>0.3",   "open")
    add("#omega(782)",      0.142, 0.114,  "OPAL 0.3<x<0.6", "filled")
    add("K^{*}(892)^{0}",   0.66, 0.11,  "OPAL x>0.7",     "filled")
    add("K^{*}(892)^{0}",   0.35, 0.067, "OPAL 0.3<x<0.4", "filled")
    add("K^{*}(892)^{0}",   0.46, 0.08,  "DELPHI x>0.4",   "open")
    add("#phi(1020)",       0.54, 0.078,  "OPAL x>0.7",     "filled")
    add("#phi(1020)",       0.55, 0.10,  "DELPHI x>0.7",   "open")
    add("D^{*}(2010)^{#pm}",0.40, 0.022,  "OPAL c#rightarrow D^{*}", "filled")
    add("D^{*}(2010)^{#pm}", 0.327, 0.006, "CLEO II", "open")
    add("D_{s1}(2536)",0.490, 0.126, "Belle x>0.8", "filled" )
    #add("B^{*}",            0.34, 0.04,  "OPAL",           "filled")
    #add("B^{*}",            0.36, 0.05,  "ALEPH",          "open")
    #add("B^{*}",            0.39, 0.05,  "DELPHI",         "open")

    # Reverse for plotting top-to-bottom
    for i, r in enumerate(reversed(rows)):
        r['y'] = i  # compact spacing 1.0
    rows = list(reversed(rows))  # maintain original order for labels; y updated

    nrows = len(rows)
    y_min = -0.8
    y_max = nrows - 0.2

    can = R.TCanvas("c", "rho00", 900, 900)
    # Use a much smaller right margin; labels will be placed just outside the frame
    can.SetMargin(0.16, 1, 0.12, 0.08)  # left, right, bottom, top

    # Frame for axes
    frame = R.TH2F("frame", ";#bf{Helicity matrix element #rho_{00}};", 100, 0.0, 1.05, 100, y_min, y_max)
    frame.GetYaxis().SetLabelSize(0)  # we'll draw our own labels
    frame.GetXaxis().SetTitleFont(FONT_TNR_BOLD)
    frame.GetXaxis().SetLabelFont(FONT_TNR_BOLD)
    frame.GetXaxis().SetTitleOffset(1.1)
    frame.GetXaxis().SetTitleSize(0.045)
    frame.GetXaxis().SetLabelSize(0.04)
    frame.GetXaxis().CenterTitle()
    frame.Draw()

    # Vertical reference line at rho00 = 1/3
    x_ref = 1.0/3.0
    line = R.TLine(x_ref, y_min, x_ref, y_max)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.SetLineColor(R.kGray+2)
    line.Draw()

    # Split points by style
    points_filled = []
    points_open = []
    for r in rows:
        entry = { 'x': r['x'], 'y': r['y'], 'ex': r['ex'], 'exl': r['ex'], 'exh': r['ex'] }
        if r['style'] == 'open':
            points_open.append(entry)
        else:
            points_filled.append(entry)

    g_filled = build_graph(points_filled, marker_style=20, marker_color=1)
    g_open   = build_graph(points_open,   marker_style=24, marker_color=1)

    g_filled.Draw("P same")
    g_open.Draw("P same")

    # Draw caps for the horizontal errors (optional, TGraphAsymmErrors already draws bars)
    # Add y grid lines for readability
    grid = R.TLine()
    grid.SetLineColor(R.kGray)
    grid.SetLineStyle(3)
    #or r in rows:
        #grid.DrawLine(0.0, r['y'], 1.05, r['y'])

    # Labels on left (particle names) and right (experiments)
    latex = R.TLatex()
    latex.SetTextFont(FONT_TNR_BOLD)
    latex.SetTextSize(0.038)
    # left labels
    for r in rows:
        latex.DrawLatexNDC(0.02, can.PadtoY(r['y'], y_min, y_max, useNDC=True), "#bf{%s}" % r['label_left'])
    # right labels
    # Place right labels inside the frame, close to the right axis
    latex.SetTextAlign(32)  # right aligned, vertically centered
    # slightly smaller text size for the right-side experiment labels
    latex.SetTextSize(0.03)
    x_max = frame.GetXaxis().GetXmax()
    x_min = frame.GetXaxis().GetXmin()
    x_right_in = x_max - 0.02*(x_max - x_min)  # 2% inside the frame
    for r in rows:
        latex.DrawLatex(x_right_in, r['y'], "#bf{%s}" % r['label_right'])

    # Big reference text in red (like the figure)
    ref = R.TLatex()
    ref.SetTextColor(R.kRed+1)
    ref.SetTextFont(FONT_TNR_BOLD)
    ref.SetTextSize(0.04)
    #ref.DrawLatexNDC(0.22, 0.10, "Eur. Phys. J. C16. 61 (2000)")

    # Small legend for marker meaning
    leg = R.TLegend(0.65, 0.84, 0.95, 0.93)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(FONT_TNR_BOLD)
    leg.SetTextSize(0.035)
    leg.AddEntry(g_filled, "#bf{filled: OPAL (examples)}", "p")
    leg.AddEntry(g_open,   "#bf{open: DELPHI (examples)}", "p")
    #leg.Draw()

    can.RedrawAxis()

    # Save
    out_png = os.path.join(save_dir, f"{basename}.png")
    out_pdf = os.path.join(save_dir, f"{basename}.pdf")
    can.SaveAs(out_png)
    can.SaveAs(out_pdf)
    print(f"Saved: {out_png}\nSaved: {out_pdf}")


# Utility: map y-value to NDC for TLatex using DrawLatexNDC
def _pad_to_y(self, y, y_min, y_max, useNDC=False):
    uxmin = self.GetUxmax()  # force update of coordinate mapping
    # ROOT doesn't expose a direct mapping for Y to NDC; compute linear map with margins.
    # We know margins from current pad.
    pad = R.gPad
    l, r, b, t = pad.GetLeftMargin(), pad.GetRightMargin(), pad.GetBottomMargin(), pad.GetTopMargin()
    # Convert y in [y_min, y_max] to NDC in [b, 1-t]
    if y_max == y_min:
        return 0.5
    y_ndc = b + (y - y_min) / float(y_max - y_min) * (1.0 - b - t)
    return y_ndc


# Monkey patch helper into TCanvas for convenience
setattr(R.TCanvas, 'PadtoY', _pad_to_y)


if __name__ == "__main__":
    draw()
