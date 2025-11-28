# dash_app/callbacks.py
from __future__ import annotations

from dash import html, dcc, Input, Output, State
import base64, io
import torch, torchaudio
import torch.nn.functional as F
import dash_daq as daq

from services.audio_model import load_audio_model, predict_audio, explain_audio_simple
from services.video_model import predict_from_upload
from services.image_model import predict_from_upload as predict_image_from_upload
from components.render_result import render_video_result, render_audio_result
from i18n import T, I18N

_REGISTERED = False
TARGET_SR = 16000
TARGET_LEN = TARGET_SR * 5

# audio model
try:
    audio_model = load_audio_model()
except Exception:
    audio_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if audio_model is not None:
    audio_model.to(device).eval()

def _safe_error(msg): return html.Div(f"❌ {msg}", style={"color":"crimson","fontWeight":600})
def _decode_upload(contents):
    if not contents or "," not in contents: return None
    _, b64 = contents.split(",", 1)
    try: return base64.b64decode(b64)
    except Exception: return None

def preprocess_waveform(wav_bytes: bytes):
    waveform, sr = torchaudio.load(io.BytesIO(wav_bytes))
    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(waveform); sr = TARGET_SR
    if waveform.dim()==2 and waveform.shape[0]>1: waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.dim()==1: waveform = waveform.unsqueeze(0)
    if waveform.shape[1] > TARGET_LEN: waveform = waveform[:, :TARGET_LEN]
    else:
        pad_len = TARGET_LEN - waveform.shape[1]
        if pad_len>0: waveform = F.pad(waveform, (0,pad_len))
    return waveform.to(device), sr

def _strip_heavy_fields(result, explain_on):
    if not isinstance(result, dict) or explain_on: return result
    for k in ("explain","attn_weights","window_probs","windows"): result.pop(k, None)
    return result

def register_callbacks(app):
    global _REGISTERED
    if _REGISTERED: return app
    _REGISTERED = True

    # Theme toggle (value=True/False from daq.ToggleSwitch)
    @app.callback(Output("root","className"), Input("dark-toggle","value"))
    def _set_theme(is_dark): return "theme--dark" if is_dark else "theme--light"

    # Sync theme to <body> and <html> so the WHOLE page changes
    app.clientside_callback(
        """
        function(cls){
          document.body.classList.remove('theme--dark','theme--light');
          document.documentElement.classList.remove('theme--dark','theme--light');
          if(cls){ document.body.classList.add(cls); document.documentElement.classList.add(cls); }
          return '';
        }
        """,
        Output("theme-sync","children"),
        Input("root","className")
    )

    # language dropdown -> store
    @app.callback(Output("lang-store","data"), Input("lang-select","value"))
    def _set_lang(val): return val or "en"

    # localize always-present elements
    @app.callback(
        Output("brand-sub","children"),
        Output("foot-left","children"),
        Output("tab-audio-label","label"),
        Output("tab-image-label","label"),
        Output("tab-video-label","label"),
        Input("lang-store","data"),
    )
    def _localize_static(lang):
        return (T("brand",lang), T("foot",lang), T("tab.audio",lang), T("tab.image",lang), T("tab.video",lang))

    # render current tab with localized text
    @app.callback(Output("tab-content","children"), Input("tabs","value"), State("lang-store","data"))
    def render_tab(tab, lang):
        if tab=="tab-audio":
            return html.Div([
                html.Div([
                    html.Div([
                        html.H3(T("audio.title",lang), className="card-title"),
                        dcc.Upload(
                            id="audio-upload",
                            children=html.Div([
                                html.Div(T("audio.h1",lang), className="muted"),
                                html.Span(T("audio.h2",lang), className="link"),
                                html.Div(T("audio.h3",lang), className="tiny muted")
                            ]),
                            multiple=False, className="dropzone",
                        ),
                        # explanation toggle (boolean)
                        dcc.Store(id="audio-explain-toggle", data=True),  # store to read in callback
                        html.Div(className="toggle-row", children=[
                            dcc.Markdown(f"**{T('show.expl', lang)}**", className="toggle-label"),
                            daq.ToggleSwitch(id="audio-explain-ui", value=True, size=30, color="#3b82f6", className="toggle")
                        ])
                    ], className="card"),
                    html.Div(id="audio-progress", className="progress", children=html.Div(className="bar")),
                    dcc.Loading(html.Div(id="audio-output", className="card result-card"), type="default"),
                ], className="grid grid-2"),
            ], className="tab-pane")

        if tab=="tab-image":
            return html.Div([
                html.Div([
                    html.Div([
                        html.H3(T("image.title",lang), className="card-title"),
                        dcc.Upload(
                            id="image-upload",
                            children=html.Div([
                                html.Div(T("image.h1",lang), className="muted"),
                                html.Span(T("image.h2",lang), className="link"),
                                html.Div(T("image.h3",lang), className="tiny muted")
                            ]),
                            multiple=False, className="dropzone",
                        ),
                        dcc.Store(id="image-explain-toggle", data=True),
                        html.Div(className="toggle-row", children=[
                            dcc.Markdown(f"**{T('show.expl', lang)}**", className="toggle-label"),
                            daq.ToggleSwitch(id="image-explain-ui", value=True, size=30, color="#3b82f6", className="toggle")
                        ])
                    ], className="card"),
                    html.Div(id="image-progress", className="progress", children=html.Div(className="bar")),
                    dcc.Loading(html.Div(id="image-output", className="card result-card"), type="default"),
                ], className="grid grid-2"),
                # html.Div(T("image.note",lang), className="note muted tiny"),
            ], className="tab-pane")

        if tab=="tab-video":
            return html.Div([
                html.Div([
                    html.Div([
                        html.H3(T("video.title",lang), className="card-title"),
                        dcc.Upload(
                            id="video-upload",
                            children=html.Div([
                                html.Div(T("video.h1",lang), className="muted"),
                                html.Span(T("video.h2",lang), className="link"),
                                html.Div(T("video.h3",lang), className="tiny muted")
                            ]),
                            multiple=False, className="dropzone",
                        ),
                        dcc.Store(id="video-explain-toggle", data=True),
                        html.Div(className="toggle-row", children=[
                            dcc.Markdown(f"**{T('show.expl', lang)}**", className="toggle-label"),
                            daq.ToggleSwitch(id="video-explain-ui", value=True, size=30, color="#3b82f6", className="toggle")
                        ])
                    ], className="card"),
                    html.Div(id="video-progress", className="progress", children=html.Div(className="bar")),
                    dcc.Loading(html.Div(id="video-output", className="card result-card"), type="default"),
                ], className="grid grid-2"),
            ], className="tab-pane")
        return html.Div()

    # explanation UI -> store (so we can read a simple boolean)
    app.clientside_callback("function(v){return v}", Output("audio-explain-toggle","data"),  Input("audio-explain-ui","value"))
    app.clientside_callback("function(v){return v}", Output("image-explain-toggle","data"),  Input("image-explain-ui","value"))
    app.clientside_callback("function(v){return v}", Output("video-explain-toggle","data"),  Input("video-explain-ui","value"))

    # AUDIO
    @app.callback(
        Output("audio-result-store", "data"),
        Input("audio-upload", "contents"),
        State("audio-explain-toggle", "data"),
        prevent_initial_call=True,
    )
    def handle_audio(contents, explain_on):
        if contents is None: return None
        try:
            raw = _decode_upload(contents)
            if raw is None: return {"error": "Could not decode uploaded audio."}
            wave, sr = preprocess_waveform(raw)
            if audio_model is None: return {"error": "Audio model is not loaded."}
            pred = predict_audio(audio_model, wave, sr)
            expl = explain_audio_simple(audio_model, wave, sr) if explain_on else None
            return {"pred": pred, "expl": expl, "show_expl": bool(explain_on)}
        except Exception as e:
            return {"error": f"Error while processing audio: {e}"}

    @app.callback(
        Output("audio-output", "children"),
        Input("audio-result-store", "data"),
        Input("audio-upload", "contents"),
        Input("lang-store", "data"),
    )
    def render_audio_output(data, contents, lang):
        # If user uploaded a file but no result yet → show processing
        if contents and not data:
            return html.Div("Processing...", className="processing-card")

        if not data:
            return None

        if "error" in data:
            return _safe_error(data["error"])

        return render_audio_result(
            data.get("pred"), data.get("expl"), data.get("show_expl"), lang=lang
        )


    @app.callback(Output("audio-progress","style"),
        Input("audio-upload","contents"), Input("audio-result-store","data"),
        prevent_initial_call=True)
    def audio_progress_style(contents, data): 
        return {"display":"block"} if contents and (data is None) else {"display":"none"}

     # IMAGE
    @app.callback(
        Output("image-result-store", "data"),
        Input("image-upload", "contents"),
        State("image-upload", "filename"),
        State("image-explain-toggle", "data"),
        prevent_initial_call=True,
    )
    def handle_image(contents, filename, explain_on):
        if contents is None or filename is None: return None
        try:
            res = predict_image_from_upload(contents, filename, explain=bool(explain_on))
            res = _strip_heavy_fields(res, bool(explain_on))
            return res
        except Exception as e:
            return {"error": f"Error while processing image: {e}"}

    @app.callback(
        Output("image-output", "children"),
        Input("image-result-store", "data"),
        Input("image-upload", "contents"),
        Input("lang-store", "data"),
)
    def render_image_output(data, contents, lang):
        if contents and not data:
            return html.Div("Processing...", className="processing-card")

        if not data:
            return None

        if "error" in data:
            return _safe_error(data["error"])

        return render_video_result(data, lang=lang)


    @app.callback(Output("image-progress","style"),
                  Input("image-upload","contents"), Input("image-result-store","data"),
                  prevent_initial_call=True)
    def image_progress_style(contents, data):
        return {"display":"block"} if contents and (data is None) else {"display":"none"}

    # VIDEO
    @app.callback(
        Output("video-result-store", "data"),
        Input("video-upload", "contents"),
        State("video-upload", "filename"),
        State("video-explain-toggle", "data"),
        prevent_initial_call=True,
    )
    def handle_video(contents, filename, explain_on):
        if contents is None or filename is None: return None
        try:
            res = predict_from_upload(contents, filename, explain=bool(explain_on))
            res = _strip_heavy_fields(res, bool(explain_on))
            return res
        except Exception as e:
            return {"error": f"Error while processing video: {e}"}

    @app.callback(
        Output("video-output", "children"),
        Input("video-result-store", "data"),
        Input("video-upload", "contents"),
        Input("lang-store", "data"),
)
    def render_video_output_cb(data, contents, lang):
        if contents and not data:
            return html.Div("Processing...", className="processing-card")

        if not data:
            return None

        if "error" in data:
            return _safe_error(data["error"])

        return render_video_result(data, lang=lang)


    @app.callback(Output("video-progress","style"),
                  Input("video-upload","contents"), Input("video-result-store","data"),
                  prevent_initial_call=True)
    def video_progress_style(contents, data):
        return {"display":"block"} if contents and (data is None) else {"display":"none"}

    return app
