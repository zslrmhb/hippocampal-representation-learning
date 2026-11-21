"""Render the reviewed poster packet without importing the scientific suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

from jinja2 import Environment, FileSystemLoader, select_autoescape
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

if __package__:
    from .evidence_tools import load_evidence
else:
    from evidence_tools import load_evidence

HERE = Path(__file__).resolve().parent
TARGETS = {"odor": "Odor", "position": "Position"}
MODELS = {"historical_anchor": "Amplitude + L1", "cebra_combined_mean": "Wavelet CEBRA-Time + L1"}
RATS = ("Mitt", "Barat", "Stella", "Superchris", "Buchanan")
COLORS = ("#147b73", "#a04220", "#3f64a2", "#834492", "#766017")
MUTED = "#486071"
INK = "#142f3b"
DIVERGING = [(0, "#245699"), (.5, "#f6f6f1"), (1, "#a23c1c")]
GRAPH_CONFIG = {"displaylogo": False, "responsive": True, "scrollZoom": False}
QUESTION = "Can a learned temporal representation improve odor or position decoding over a direct wavelet-amplitude baseline?"
COMPARISON_NOTE = (
    "Direct averages keep time and frequency explicit; CEBRA-Time learns a compact representation from local temporal relationships. "
    "Both use an L1-regularized linear head, but the inputs and selection procedures differ. This compares complete recipes, not architecture alone."
)
RESULT_NOTE = (
    "The direct amplitude baseline scored higher on both targets on validation and on the previously viewed test. "
    "These five-rat results are descriptive, not independent confirmation or evidence of neural causality."
)
TASK_NOTE = (
    "Odor is A–E on correct InSeq trials. Position is ordinal sequence position (1–5), not physical maze location, "
    "on correct InSeq and OutSeq trials. Odors appear at their usual position on InSeq trials and out of order on OutSeq trials."
)
PROTOCOL = (
    ("Input", "TRAIN-scaled Morlet log amplitude", "Same scaled amplitude plus aligned phase sine and cosine"),
    ("Representation", "10 × 50 ms averages per electrode and center frequency",
     "offset10-model; 1,000 updates; mean over valid centers; 8 dimensions"),
    ("Head and selection", "TRAIN-fitted class-balanced L1 logistic head; frozen historical C per rat",
     "Same head family; validation-selected C per target; frozen TRAIN-trained checkpoint per rat and target"),
)
CONTROLS_NOTE = (
    "Both recipes use five within-rat heads per target, matched chronological partitions and trials, seed 42 "
    "for heads and applicable encoders, and TRAIN-only head standardization. Test did not determine selection."
)
SPLIT_NOTE = (
    "At 1 kHz, a 70/15/15 chronological raw sample-index split is intersected with timestamp segments. "
    "Trials belong to their onset’s split; the 500 ms odor-aligned clip must fit before withdrawal and inside "
    "a guarded clean block. These are sample fractions, not class-balanced trial fractions."
)
FILTER_NOTE = (
    "TRAIN-only QA uses median/MAD thresholds (SD fallback; multiplier 20) and screens training-constant channels, "
    "nonfinite samples, excessive amplitudes, both sides of excessive jumps, and exact flatlines ≥100 ms. "
    "The forward/backward 60 Hz Q30 notch uses a 2,199-sample guard; 12 Morlet centers from 4 to 150 Hz require "
    "a combined 2,677-sample guard. Amplitude scaling uses unique retained TRAIN samples. Filtering uses "
    "continuous context, including future samples, within the owning block: this is offline analysis."
)
SIGNAL_NOTE = (
    "Legacy spectra use overlapping 4 s Welch windows; high-resolution spectra use nonoverlapping 16 s Hann windows. "
    "Attenuation is a power dB ratio, not a physiological-origin test. A dash marks a peak below the reporting "
    "threshold. Mitt’s legacy spectrum was independently reviewed; high-resolution results are runner-reported."
)
SLIDE_NOTE = (
    "The reviewed four-rat column averages strict five-class BA over the same four rats, excluding Buchanan, "
    "without pooling trials. The archived finalist scores are included in the evidence packet; the historical CEBRA position model "
    "used 200 ms, not 500 ms. This is a cross-check, not the primary estimand or a paired test."
)
REPO_URL = "https://github.com/zslrmhb/hippocampal-representation-learning#1-poster-rebuild-from-reviewed-aggregates"
METRIC_NOTE = (
    "Validation BA is the mean of the five class recalls within each rat, then an equal mean across rats. "
    "On the previously viewed test, Buchanan has four observed classes; other rats have five. "
    "The primary five-rat test score averages each rat's recall over only its observed classes, "
    "then weights the five rats equally. Strict five-class BA is unavailable for the equal-five-rat test cohort. "
    "The displayed trial counts describe support; trials are not pooled to calculate these scores."
)
OUTSEQ_NOTE = (
    "Compare each position prediction with the observed position and the odor's usual position from correct TRAIN InSeq trials. "
    "Only correct OutSeq trials where those positions differ enter this contrast. Both models are scored "
    "on the same eight previously viewed test trials. This is descriptive: elapsed time, movement and "
    "sequence context remain possible confounds."
)


def _pct(value):
    return "Unavailable" if value is None else f"{value:.1%}"


def _number(value, decimals=1):
    return "—" if value is None else f"{value:.{decimals}f}"


def headline_table(evidence):
    """Four model/target rows pairing validation BA with the primary test summary."""
    scores = {(r["target"], r["model"], r["split"]): r for r in evidence["headline"]}
    table = []
    for target, label in TARGETS.items():
        for model, name in MODELS.items():
            val = scores[target, model, "val"]
            test = scores[target, model, "test"]
            table.append({
                "Target": label, "Model": name, "Validation BA": _pct(val["ba"]),
                "Previously viewed test recall": _pct(test["observed_macro_recall"]),
                "Trials (val / test)": f"{val['n']} / {test['n']}",
                "Full-five-class rats (test)": f"{test['n_rat_ba']} / {test['n_rat']}",
            })
    return table


def slide_table(evidence):
    return [{
        "Target": TARGETS[r["target"]], "Model family": MODELS[r["model"]],
        "Reviewed four-rat BA · 500 ms": _pct(r["current_ba"]),
        "Archived finalist BA": _pct(r["slide_ba"]),
        "Archived window": f"{r['slide_window_ms']} ms", "Archived configuration": r["slide_configuration"],
    } for r in evidence["slide_check"]]


def outseq_table(evidence):
    rows = {r["model"]: r for r in evidence["outseq"] if r["split"] == "test" and r["scope"] == "pooled_trials"}
    return [{
        "Model": label, "Follows observed position": rows[model]["follows_actual"],
        "Follows usual position": rows[model]["follows_usual"], "Neither": rows[model]["neither"],
        "Matched trials": rows[model]["n"],
    } for model, label in MODELS.items()]


def signal_table(evidence):
    status = {"independently_reviewed_mitt": "Mitt legacy QA reviewed", "runner_reported": "Runner reported"}
    return [{
        "Rat": r["session"], "Legacy windows": r["legacy_windows"],
        "Legacy peak (Hz)": _number(r["legacy_peak_hz"], 2),
        "Legacy attenuation (dB)": _number(r["legacy_attenuation_db"]),
        "High-res. windows": r["windows"], "High-res. peak (Hz)": _number(r["peak_hz"], 2),
        "High-res. attenuation (dB)": _number(r["attenuation_db"]),
        "Evidence status": status[r["evidence_status"]],
    } for r in evidence["signal_qa"]]


def winner_table(evidence):
    names = {("amp_mean", "hist_boost"): "Mean amplitude + histogram boosting",
             ("wave_tangent", "rbf_svm"): "Morlet tangent features + RBF SVM"}
    return [{"Target": TARGETS[r["target"]], "Validation winner": names[r["encoder"], r["decoder"]],
             "Validation BA": _pct(r["val_ba"])} for r in evidence["matrix"] if r["recommended"]]


def _style(fig, height, *, margin=None):
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": INK, "size": 12},
        margin=margin or dict(l=48, r=24, b=48, t=35), hoverlabel={"font_size": 12},
    )
    return fig


def _diagram(nodes, edges, height, *, yrange):
    fig = _style(go.Figure(), height, margin=dict(l=1, r=1, t=1, b=1))
    for (x0, y0), (x1, y1) in edges:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, axref="x", ayref="y", xref="x", yref="y",
                           text="", showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=MUTED)
    for x, y, label, color in nodes:
        fig.add_annotation(x=x, y=y, text=label, showarrow=False, align="center", bgcolor=color,
                           bordercolor="#9aafb5", borderwidth=1, borderpad=6)
    fig.update_xaxes(visible=False, range=[0, 1], fixedrange=True)
    fig.update_yaxes(visible=False, range=yrange, fixedrange=True)
    return fig


def workflow_diagram():
    """Chronological trial ownership and the matched two-branch comparison."""
    common, amp, cebra, score = "#edf4f3", "#dbf0e8", "#f9e8df", "#e8edf6"
    levels = (5.40, 4.60, 3.80, 3.00, 2.20, 1.32, .51)
    x = .5
    nodes = [
        (x, levels[0], "<b>Digitized LFP (1 kHz) + behavior</b>", common),
        (x, levels[1], "70/15/15 chronological sample split ∩ timestamp segments", common),
        (x, levels[2], "TRAIN QA → 60 Hz notch (Q30) + 12 Morlet centers", common),
        (x, levels[3], "Split-owned 0–500 ms trials → unique-TRAIN amplitude scaling", common),
        (.27, levels[4], "<b>Direct log amplitude</b><br>10 × 50 ms averages", amp),
        (.73, levels[4], "<b>Amplitude + phase → CEBRA-Time</b><br>offset10 · 1,000 updates · mean valid centers · 8D", cebra),
        (x, levels[5], "TRAIN-fitted within-rat L1 heads", score),
        (x, levels[6], "Frozen, previously viewed test", score),
    ]
    edges = [((x, levels[i] - .25), (x, levels[i + 1] + .25)) for i in (0, 1, 2, 5)]
    for branch in (.27, .73):
        edges.extend([((x, levels[3] - .25), (branch, levels[4] + .33)),
                      ((branch, levels[4] - .33), (x, levels[5] + .25))])
    return _diagram(nodes, edges, 354, yrange=(.07, 5.80))


def outseq_diagram():
    return _diagram([
        (.50, 1.63, "<b>Correct OutSeq</b> · observed position ≠ usual position", "#edf4f3"),
        (.16, .44, "Observed position", "#dbf0e8"),
        (.50, .44, "Usual position", "#f9e8df"),
        (.84, .44, "Prediction = neither", "#e8edf6"),
    ], [((.5, 1.36), (x, .74)) for x in (.16, .50, .84)], 128, yrange=(0, 2.05))


def _paired(evidence):
    rows = {(r["target"], r["model"], r["session"]): r for r in evidence["session_scores"] if r["split"] == "test"}
    classes = {}
    for r in evidence["class_support"]:
        if r["split"] == "test" and r["support"] > 0:
            key = (r["target"], r["model"], r["session"])
            classes[key] = classes.get(key, 0) + 1
    rat_labels = ["Buchanan · 4 classes" if rat == "Buchanan" else rat for rat in RATS]
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=list(TARGETS.values()), horizontal_spacing=.09)
    for col, target in enumerate(TARGETS, 1):
        lines_x, lines_y = [], []
        for rat, name in zip(RATS, rat_labels):
            lines_x.extend([rows[target, model, rat]["observed_macro_recall"] for model in MODELS] + [None])
            lines_y.extend((name, name, None))
        fig.add_trace(go.Scatter(x=lines_x, y=lines_y, mode="lines", line={"color": "#8b9ca5"},
                                 showlegend=False, hoverinfo="skip"), row=1, col=col)
        for (model, label), color, symbol in zip(MODELS.items(), COLORS, ("circle", "diamond")):
            fig.add_trace(go.Scatter(
                x=[rows[target, model, rat]["observed_macro_recall"] for rat in RATS], y=rat_labels,
                mode="markers", name=label, legendgroup=model, showlegend=col == 1,
                marker={"color": color, "symbol": symbol, "size": 10, "line": {"color": "white", "width": 1}},
                customdata=[[rows[target, model, rat]["n"], classes[target, model, rat]] for rat in RATS],
                hovertemplate="%{y}<br>Recall: %{x:.1%}<br>Trials: %{customdata[0]}<br>Observed classes: %{customdata[1]}",
            ), row=1, col=col)
    fig.update_xaxes(tickformat=".0%", range=[0, 1], title_text="Observed-class recall")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=rat_labels)
    fig.update_layout(legend=dict(orientation="h", x=0, y=1.43, yanchor="top"))
    return _style(fig, 295, margin=dict(l=128, r=15, b=47, t=62))


def interpretation_notes(evidence):
    weights = evidence["l1_weights"]
    peak = max(weights, key=lambda row: abs(row["mean_weight"]))
    start = peak["bin_start_ms"]
    zero = sum(row["n_nonzero"] == 0 for row in weights)
    projection = evidence["cebra_projection"]
    explained = projection[0]["pc1_variance"] + projection[0]["pc2_variance"]
    outseq = sum(row["n"] for row in projection if row["cohort"] == "outseq" and row["grouping"] == "position")
    return {
        "weights": (
            f'The largest absolute signed mean is {peak["mean_weight"]:+.4f} at '
            f'{peak["frequency_hz"]:.1f} Hz and {start}–{start + 50} ms '
            f'({peak["n_nonzero"]}/{peak["n_electrodes"]} nonzero electrode coefficients). '
            f'{zero} of {len(weights)} tiles have no nonzero coefficients. '
            'These model weights are not causal importance estimates.'
        ),
        "projection": (
            f'The first two PCs explain {explained:.1%} of TRAIN latent variance. '
            f'The OutSeq view contains only {outseq} validation trials; the centroids are descriptive.'
        ),
    }


def _weights(evidence):
    rows = {(r["frequency_hz"], r["bin_start_ms"]): r for r in evidence["l1_weights"]}
    frequencies = sorted({f for f, _ in rows})
    bins = sorted({b for _, b in rows})
    z = [[rows[f, b]["mean_weight"] for b in bins] for f in frequencies]
    extent = max(abs(v) for row in z for v in row) or 1
    fig = go.Figure(go.Heatmap(
        x=bins, y=list(range(len(frequencies))), z=z,
        customdata=[[[f, rows[f, b]["n_nonzero"], rows[f, b]["n_electrodes"]] for b in bins] for f in frequencies],
        colorscale=DIVERGING, zmin=-extent, zmax=extent, colorbar={"thickness": 10, "len": .86},
        hovertemplate="%{x} ms bin start · %{customdata[0]:.2f} Hz<br>Signed mean: %{z:.4g}"
                      "<br>Nonzero: %{customdata[1]}/%{customdata[2]}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Bin start (ms)", tickmode="array", tickvals=bins, side="bottom")
    fig.update_yaxes(title_text="Center (Hz)", tickmode="array", tickvals=list(range(len(frequencies))),
                     ticktext=[f"{f:.1f}" for f in frequencies], tickfont_size=10)
    return _style(fig, 310, margin=dict(l=70, r=10, t=8, b=42))


def _mask_label(group, frequencies):
    kind, low, high = group.split("_")
    lo, hi = int(low), int(high)
    if kind == "freq":
        return f"{frequencies[lo]:.3g}–{frequencies[hi - 1]:.3g} Hz"
    return f"{lo * 50}–{hi * 50} ms"


def _masking(evidence):
    rows = {(r["target"], r["group"], r["session"]): r for r in evidence["l1_masking"]}
    frequencies = sorted({r["frequency_hz"] for r in evidence["l1_weights"]})
    groups = sorted({group for _, group, _ in rows}, key=lambda group: (group.startswith("time"), group))
    labels = [_mask_label(g, frequencies) for g in groups]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=list(TARGETS.values()),
                        horizontal_spacing=.11)
    bound = max(.01, max(abs(r["delta"]) for r in rows.values())) * 1.15
    for col, target in enumerate(TARGETS, 1):
        for i, (rat, color, symbol) in enumerate(zip(RATS, COLORS, ("circle", "diamond", "square", "cross", "triangle-up"))):
            series = [rows[target, g, rat] for g in groups]
            fig.add_trace(go.Scatter(
                x=[r["delta"] for r in series], y=[j + (i - 2) * .105 for j in range(len(groups))],
                mode="markers", name=rat, legendgroup=rat, showlegend=col == 1,
                marker={"color": color, "size": 7, "symbol": symbol},
                customdata=[[label, r["intact_ba"], r["masked_ba"], r["n_classes"], r["n"]]
                            for label, r in zip(labels, series)],
                hovertemplate="%{customdata[0]}<br>Intact: %{customdata[1]:.1%} · masked: %{customdata[2]:.1%}"
                              "<br>Change: %{x:+.1%}<br>%{customdata[3]} classes · %{customdata[4]} trials",
            ), row=1, col=col)
    fig.update_xaxes(tickformat=".0%", range=[-bound, bound], zeroline=True, zerolinecolor="#8b9ca5")
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(groups))), ticktext=labels,
                     range=[len(groups) - .5, -.5], tickfont_size=10)
    fig.update_layout(legend=dict(orientation="h", x=0, y=-.15, yanchor="top", font_size=10))
    return _style(fig, 310, margin=dict(l=88, r=9, t=28, b=57))


def _padded_range(values):
    lo, hi = min(values), max(values)
    pad = .14 * (hi - lo) if hi > lo else 1
    return [lo - pad, hi + pad]


def _projection(evidence):
    rows = evidence["cebra_projection"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(TARGETS.values()), horizontal_spacing=.10)
    max_n = max(r["n"] for r in rows)
    for cohort in ("all", "outseq"):
        for col, group in enumerate(TARGETS, 1):
            selected = sorted((r for r in rows if r["grouping"] == group and r["cohort"] == cohort),
                              key=lambda r: r["class"])
            fig.add_trace(go.Scatter(
                x=[r["pc1"] for r in selected], y=[r["pc2"] for r in selected], text=[r["class"] for r in selected],
                textposition="middle center", textfont={"color": "white", "size": 10, "weight": 700},
                customdata=[r["n"] for r in selected], visible=cohort == "all",
                mode="markers+text", showlegend=False, marker={
                    "size": [r["n"] for r in selected], "sizemode": "area", "sizeref": 2 * max_n / 24**2,
                    "sizemin": 5, "color": [COLORS[int(r["class"]) - 1] for r in selected],
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate="Class %{text}<br>PC1 %{x:.3f} · PC2 %{y:.3f}<br>Validation trials: %{customdata}<extra></extra>",
            ), row=1, col=col)
    fig.update_xaxes(title_text=f"PC1 · {_pct(rows[0]['pc1_variance'])} TRAIN variance",
                     range=_padded_range([r["pc1"] for r in rows]))
    fig.update_yaxes(title_text=f"PC2 · {_pct(rows[0]['pc2_variance'])} TRAIN variance", row=1, col=1)
    fig.update_yaxes(range=_padded_range([r["pc2"] for r in rows]))
    fig.update_layout(updatemenus=[dict(type="buttons", direction="right", x=0, y=1.25, xanchor="left",
        buttons=[dict(label=label, method="restyle", args=[{"visible": visible}]) for label, visible in (
            ("All validation", [True, True, False, False]), ("OutSeq validation", [False, False, True, True]),
        )])])
    return _style(fig, 335, margin=dict(l=78, r=24, t=72, b=47))


def _matrix(evidence):
    rows = [r for r in evidence["matrix"] if not r["is_anchor"]]
    encoders = list(dict.fromkeys(r["encoder"] for r in rows))
    decoders = list(dict.fromkeys(r["decoder"] for r in rows))
    lookup = {(r["target"], r["encoder"], r["decoder"]): r for r in rows}
    fig = go.Figure()
    for target in TARGETS:
        grid = [[lookup[target, enc, dec] for dec in decoders] for enc in encoders]
        for is_scored in (False, True):
            z = [[r["val_ba"] if is_scored else (0 if r["val_ba"] is None else None) for r in row] for row in grid]
            details = [[[r["input_view"], r["status"], r["window_ms"], r["validation_rank"] or "—"]
                        for r in row] for row in grid]
            fig.add_trace(go.Heatmap(
                x=decoders, y=encoders, z=z, customdata=details, hoverongaps=False, xgap=1, ygap=1,
                visible=target == "odor", zmin=0 if not is_scored else .20, zmax=1 if not is_scored else .60,
                colorscale="Teal" if is_scored else [(0, "#dce3e7"), (1, "#dce3e7")], showscale=is_scored,
                colorbar={"title": "VAL BA", "tickformat": ".0%", "thickness": 13},
                hovertemplate="%{y} × %{x}<br>Input view: %{customdata[0]}<br>Status: %{customdata[1]}"
                              "<br>Window (ms): %{customdata[2]}<br>Validation rank: %{customdata[3]}"
                              + ("<br>Validation BA: %{z:.2%}" if is_scored else "") + "<extra></extra>",
            ))
    fig.update_yaxes(autorange="reversed", title_text="Encoder")
    fig.update_xaxes(tickangle=-45, side="bottom", title_text="Decoder")
    fig.update_layout(updatemenus=[dict(type="buttons", direction="right", x=0, y=1.11, xanchor="left",
        buttons=[dict(label=label, method="restyle", args=[{"visible": visible}]) for label, visible in (
            (TARGETS["odor"], [True, True, False, False]), (TARGETS["position"], [False, False, True, True]),
        )])])
    return _style(fig, 630, margin=dict(l=175, r=72, t=60, b=127))


def figures(evidence):
    """Named figures in report order; the matrix is interactive-only in print."""
    return {
        "workflow": workflow_diagram(), "paired": _paired(evidence), "weights": _weights(evidence),
        "masking": _masking(evidence), "projection": _projection(evidence), "outseq": outseq_diagram(),
        "matrix": _matrix(evidence),
    }


def _atomic_write(path, payload):
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        staged = Path(temp.name)
    try:
        staged.write_bytes(payload)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def _output_dir(path, evidence_root):
    path, evidence_root = path.resolve(), evidence_root.resolve()
    if path.is_relative_to(evidence_root):
        raise ValueError("The output directory cannot be inside the evidence packet")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _print_pdf(html):
    from playwright.sync_api import sync_playwright

    with TemporaryDirectory(prefix="poster-print-") as temp, sync_playwright() as playwright:
        source = Path(temp, "poster-results.html")
        source.write_text(html, encoding="utf-8")
        browser = playwright.chromium.launch()
        try:
            # Match the CSS @page content box so Plotly lays out before printing at the final width.
            page = browser.new_page(viewport={"width": 680, "height": 1016})
            errors = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.emulate_media(media="print")
            page.goto(source.as_uri())
            page.wait_for_function("window.posterReady === true || Boolean(window.posterRenderError)")
            render_error = page.evaluate("window.posterRenderError || null")
            if render_error or errors:
                raise RuntimeError(f"Poster did not render: {render_error or '; '.join(errors)}")
            pdf = page.pdf(format="A4", print_background=True, prefer_css_page_size=True)
            if errors:
                raise RuntimeError(f"Browser errors during print: {'; '.join(errors)}")
            return pdf
        finally:
            browser.close()


def build(evidence_root, output, *, pdf=False):
    evidence = load_evidence(evidence_root)
    env = Environment(loader=FileSystemLoader(HERE), autoescape=select_autoescape(("html", "jinja")),
                      trim_blocks=True, lstrip_blocks=True)
    html = env.get_template("report.html.jinja").render(
        figures={name: json.loads(fig.to_json()) for name, fig in figures(evidence).items()},
        plotlyjs=get_plotlyjs(), config=GRAPH_CONFIG, metrics_note=METRIC_NOTE, outseq_note=OUTSEQ_NOTE,
        question=QUESTION, comparison_note=COMPARISON_NOTE, result_note=RESULT_NOTE, task_note=TASK_NOTE,
        protocol=PROTOCOL, models=MODELS.values(), controls_note=CONTROLS_NOTE,
        split_note=SPLIT_NOTE, filter_note=FILTER_NOTE, signal_note=SIGNAL_NOTE, slide_note=SLIDE_NOTE,
        interpretation=interpretation_notes(evidence),
        repo_url=REPO_URL,
        headline=headline_table(evidence), slide=slide_table(evidence), outseq=outseq_table(evidence),
        signal=signal_table(evidence), winners=winner_table(evidence), manifest=evidence["manifest"],
    )
    pdf_bytes = _print_pdf(html) if pdf else None
    output = _output_dir(output, evidence_root)
    _atomic_write(output / "poster-results.html", html.encode("utf-8"))
    if pdf_bytes is not None:
        _atomic_write(output / "poster-results.pdf", pdf_bytes)


def notebook(evidence_root, output):
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    load_evidence(evidence_root)
    repo = HERE.parents[1]
    source = repo / "notebooks" / "poster_results.ipynb"
    nb = nbformat.read(source, as_version=4)
    with TemporaryDirectory(prefix="poster-kernel-") as kernels:
        kernel_name = "poster-release"
        spec_dir = Path(kernels, kernel_name)
        spec_dir.mkdir()
        (spec_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Locked poster Python", "language": "python",
        }), encoding="utf-8")
        manager = KernelManager(kernel_name=kernel_name,
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[kernels], ensure_native_kernel=False))
        client = NotebookClient(nb, km=manager, kernel_name=kernel_name, timeout=120)
        try:
            client.execute(cwd=str(repo), env={**os.environ, "POSTER_EVIDENCE": str(evidence_root.resolve())})
        finally:
            # The caller supplied the kernel manager; nbclient does not own its lifecycle.
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    output = _output_dir(output, evidence_root)
    _atomic_write(output / "poster-results.executed.ipynb", nbformat.writes(nb).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "notebook"):
        command = commands.add_parser(name)
        command.add_argument("--evidence", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "build":
            command.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        build(args.evidence, args.output, pdf=args.pdf)
    else:
        notebook(args.evidence, args.output)


if __name__ == "__main__":
    main()
