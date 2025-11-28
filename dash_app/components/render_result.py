# dash_app/components/render_result.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from dash import html
from i18n import T

# ----------------- helpers -----------------
def _pct(x) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "—"

def _bar(weights: List[float] | None):
    """
    Dark-theme friendly attention bar.
    Uses the CSS classes you already have in styles.css:
      .bar (container), .bar-part (segments)
    """
    if not weights:
        return html.Div(className="bar")
    total = sum(max(0.0, float(w)) for w in weights) or 1.0
    parts = []
    for w in weights:
        pct = 100.0 * max(0.0, float(w)) / total
        parts.append(html.Div(className="bar-part", style={"width": f"{pct:.2f}%"}))
    return html.Div(parts, className="bar")

def _thumb(img_b64: str, label: str):
    if not img_b64:
        return html.Div()
    return html.Div([
        html.Div(label, style={"fontSize": "0.75rem", "opacity": 0.9, "marginBottom": "4px"}),
        html.Img(src=f"data:image/png;base64,{img_b64}", className="thumb")
    ], style={"display": "inline-block", "marginRight": "8px"})

def _explain_gallery(res: Dict[str, Any], lang: str = "en"):
    items = res.get("explain", []) or []
    if not items:
        return None
    cards = []
    for it in items:
        meta = html.Div([
            html.Div(f"{T('res.frame', lang)} #{it.get('frame_idx', '—')}", style={"fontSize": "0.8rem"}),
            html.Div(f"Attention: {float(it.get('attn', 0.0)):.3f}", style={"fontSize": "0.8rem", "opacity": 0.9})
        ], style={"marginBottom": "4px"})
        imgs = html.Div([
            _thumb(it.get("thumb_b64", ""), T("res.frame", lang)),
            _thumb(it.get("cam_b64", ""), "Grad-CAM"),
        ], className="thumbs")
        # Themed card (no bright whites)
        cards.append(html.Div([meta, imgs], style={
            "display": "inline-block",
            "padding": "8px",
            "border": "1px solid var(--border)",
            "borderRadius": "10px",
            "marginRight": "10px",
            "marginBottom": "10px",
            "background": "var(--panel)"
        }))
    return html.Div([
        html.Div(T("res.expl", lang), style={"fontWeight": 700, "marginTop": "10px", "marginBottom": "4px"}),
        html.Div(cards)
    ])

def render_empty_result(msg="Drop a file to begin."):
    return html.Div([html.Div(msg, className="muted")], className="result-empty")

def _kv(label: str, value: str):
    # small key-value chip
    return html.Span([
        html.Span(f"{label}: ", style={"fontWeight": 600}),
        html.Span(value)
    ], style={"marginRight": "12px"})

def _timeline(values: List[float] | None):
    """Small horizontal bar made of segments, showing per-window scores."""
    if not values:
        return None

    segments = []
    for v in values:
        try:
            alpha = 0.25 + 0.6 * max(0.0, float(v))
        except Exception:
            alpha = 0.25
        segments.append(
            html.Div(
                style={
                    "flex": "1 1 0",
                    "height": "10px",
                    "borderRadius": "999px",
                    "background": f"rgba(248,113,113,{alpha:.3f})",
                }
            )
        )

    return html.Div(
        segments,
        style={"display": "flex", "gap": "4px", "marginTop": "6px"},
    )


# ----------------- VIDEO / IMAGE -----------------
def render_video_result(res: Dict[str, Any], lang: str = "en"):
    """
    Renders both video and image unified outputs (unchanged data contract).
    """
    if not isinstance(res, dict):
        return html.Div(T("res.invalid", lang), style={"color": "crimson"}, className="result")

    # error surfacing
    if "error" in res and res["error"]:
        return html.Div([
            html.Div(T("res.error", lang), style={"fontWeight": 700, "marginBottom": "6px"}),
            html.Pre(str(res["error"]), style={"whiteSpace": "pre-wrap", "color": "crimson"})
        ], className="result")

    # ----- probabilities / label -----
    probs = res.get("probs") or {}
    fake_prob = float(
        probs.get("fake",
            res.get("fake_prob",
                res.get("prob_fake",
                    1.0 - float(res.get("real_prob", probs.get("real", 0.0)))
                )
            )
        )
    )
    real_prob = float(
        probs.get("real",
            res.get("real_prob",
                1.0 - fake_prob
            )
        )
    )

    label = (res.get("label") or ("FAKE" if fake_prob >= real_prob else "REAL")).upper()
    low_conf = bool(res.get("low_confidence"))
    color = "#f87171" if label == "FAKE" else "#60a5fa" if label == "REAL" else "#cbd5e1"

    # ----- header line -----
    header = html.Div(T("res.header", lang), style={"fontWeight": 700, "marginBottom": "8px"})
    
    label_text = T("res.label.fake", lang) if label == "FAKE" else (T("res.label.real", lang) if label == "REAL" else T("res.label.unsure", lang))
    
    line_items = [
        _kv(T("res.fake", lang), _pct(fake_prob)),
        _kv(T("res.real", lang), _pct(real_prob)),
        html.Span(T("res.final", lang), style={"fontWeight": 600}),
        html.Span("●", style={"color": color, "margin": "0 6px"}),
        html.Span(label_text, style={"color": color, "fontWeight": 700}),
    ]
    if low_conf:
        line_items.append(html.Span(T("res.low_conf", lang), style={"color": "#fca5a5", "marginLeft": "6px"}))
    line = html.Div(line_items)

        # ----- window scores (video) -----
    window_section = None
    timeline = None
    if (res.get("type") == "video" or "window_probs" in res) and res.get("window_probs"):
        items = [html.Li(f"Window {i+1}: { _pct(p) }") for i, p in enumerate(res["window_probs"])]
        window_section = html.Div([
            html.Div("Window Scores", style={"fontWeight": 700, "marginTop": "8px"}),
            html.Ul(items, style={"margin": "6px 0 0 18px"})
        ])
        timeline = _timeline(res["window_probs"])


    # ----- attention bar -----
    attn_section = None
    attn_weights = res.get("attn_weights")
    if attn_weights:
        attn_section = html.Div([
            html.Div(T("res.attn", lang), style={"fontWeight": 700, "marginTop": "8px"}),
            _bar(attn_weights)
        ])
    elif res.get("type") == "video" and res.get("windows"):
        w0 = res["windows"][0]
        if w0.get("attn_weights"):
            attn_section = html.Div([
                html.Div(T("res.frame_attn", lang), style={"fontWeight": 700, "marginTop": "8px"}),
                _bar(w0["attn_weights"])
            ])

    # ----- Grad-CAM thumbnails gallery -----
    gallery = _explain_gallery(res, lang)

    # ----- footer -----
    file_info = res.get("path") or res.get("file") or res.get("filename") or ""
    subtype = res.get("type", "").capitalize() if res.get("type") else ""
    footer_bits = []
    if file_info:
        footer_bits.append(html.Span(f"{T('res.file', lang)}{file_info}"))
    if subtype:
        if footer_bits:
            footer_bits.append(html.Span("  •  ", style={"opacity": 0.6, "margin": "0 6px"}))
        footer_bits.append(html.Span(subtype))
    footer = html.Div(footer_bits, className="muted") if footer_bits else html.Div()

    # IMPORTANT: no hardcoded white background; rely on themed container styles
    return html.Div(
        [header, line, timeline, window_section, attn_section, gallery, html.Hr(), footer],
        className="result",
    )

# ----------------- AUDIO -----------------
def render_audio_result(pred: Dict[str, Any], expl: Optional[Dict[str, Any]] = None, show_expl: bool = True, lang: str = "en"):
    """
    Render card for audio prediction (unchanged API).
    """
    probs = pred.get("probs") or {}
    fake_prob = float(probs.get("fake", pred.get("fake_prob", pred.get("prob_fake", 0.0))))
    real_prob = float(probs.get("real", pred.get("real_prob", 1.0 - fake_prob)))
    label = (pred.get("label") or ("FAKE" if fake_prob >= real_prob else "REAL")).upper()
    color = "#f87171" if label == "FAKE" else "#60a5fa" if label == "REAL" else "#cbd5e1"

    header = html.Div(T("res.header.audio", lang), style={"fontWeight": 700, "marginBottom": "8px"})
    
    label_text = T("res.label.fake", lang) if label == "FAKE" else (T("res.label.real", lang) if label == "REAL" else T("res.label.unsure", lang))

    line = html.Div([
        _kv(T("res.fake", lang), _pct(fake_prob)),
        _kv(T("res.real", lang), _pct(real_prob)),
        html.Span(T("res.final", lang), style={"fontWeight": 600}),
        html.Span("●", style={"color": color, "margin": "0 6px"}),
        html.Span(label_text, style={"color": color, "fontWeight": 700}),
    ])

    blocks = [header, line]

    if show_expl and expl:
        spans = [f"{s:.1f}–{e:.1f}s" for (s, e) in expl.get("spans", [])]
        reason_bits = [html.Span(T("res.why", lang), style={"fontWeight": 600}),
                       html.Span(expl.get("reason", ""))]
        if spans:
            reason_bits.append(html.Span(T("res.spans", lang) + ", ".join(spans) + ")", style={"marginLeft": "6px"}))
        blocks.append(html.Div(reason_bits, className="muted", style={"marginTop": "6px"}))

    if pred.get("band"):
        blocks.append(html.Div(f"{T('res.band', lang)}{pred['band']}", className="muted", style={"marginTop": "6px"}))
    if pred.get("verdict"):
        blocks.append(html.Div(f"{T('res.verdict', lang)}{pred['verdict']}", className="muted"))

    # IMPORTANT: no hardcoded white background; rely on themed container styles
    return html.Div(blocks, className="result")
