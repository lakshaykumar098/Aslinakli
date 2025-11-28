# dash_app/layout.py
from dash import html, dcc
import dash_daq as daq


def _header():
    return html.Header([
        html.Div([
            html.Img(src="/assets/AsliNakli.jpeg", className="brand-logo"),
            html.Div([
                html.Div("AsliNakli", className="brand-title"),
                html.Div("Deepfake Detection Tool", id="brand-sub", className="brand-subtitle"),
            ], className="brand-text")
        ], className="brand"),

        html.Div([
            dcc.Dropdown(
                id="lang-select",
                options=[
                    {"label": "English", "value": "en"},
                    {"label": "हिंदी", "value": "hi"},
                    {"label": "मराठी", "value": "mr"},
                    {"label": "বাংলা", "value": "bn"},
                    {"label": "தமிழ்", "value": "ta"},
                    {"label": "తెలుగు", "value": "te"},
                    {"label": "ગુજરાતી", "value": "gu"},
                    {"label": "ಕನ್ನಡ", "value": "kn"},
                    {"label": "اردو", "value": "ur"},
                    {"label": "ਪੰਜਾਬੀ", "value": "pa"},
                ],
                value="en",
                clearable=False,
                className="lang-dropdown",
            ),
            daq.ToggleSwitch(
                id="dark-toggle",
                value=True,              # True=dark
                label="Dark Mode 🌙",
                labelPosition="right",
                size=40,
                color="#3b82f6",         # <<< ON color (blue)
                className="toggle"
            ),
        ], className="header-actions")
    ], className="app-header")


def _footer():
    return html.Footer([
        html.Div(id="foot-left", className="foot-left"),
        html.Div(id="foot-right", className="foot-right")
    ], className="app-footer")


def _audio_tab():
    return html.Div([
        html.Div([
            html.Div([
                html.H3([html.I(className="fas fa-microphone"), " Audio Analysis"], id="audio-title", className="card-title"),
                dcc.Upload(
                    id="audio-upload",
                    children=html.Div([
                        html.Div("Drag and Drop or Click to Upload", id="audio-upload-hint-1", className="muted"),
                        html.Span("Supported formats: WAV, MP3, FLAC", id="audio-upload-hint-2", className="tiny muted"),
                    ]),
                    multiple=False,
                    className="dropzone",
                ),
                daq.ToggleSwitch(
                    id="audio-explain-ui",
                    value=True,
                    label="Show AI Explanation",
                    labelPosition="right",
                    size=35,
                    color="#3b82f6",     # <<< ON color (blue)
                    className="toggle"
                ),
                dcc.Store(id="audio-explain-toggle", data=True),
            ], className="card"),

            # progress bar: inner div renamed to "pbar"
            html.Div([
                html.Div(id="audio-progress", className="progress", children=html.Div(className="pbar")),
                dcc.Loading(html.Div(id="audio-output", className="card result-card"), type="default", color="#3b82f6"),
            ]),
        ], className="grid grid-2"),
    ], className="tab-pane")


def _image_tab():
    return html.Div([
        html.Div([
            html.Div([
                html.H3([html.I(className="fas fa-image"), " Image Analysis"], id="image-title", className="card-title"),
                dcc.Upload(
                    id="image-upload",
                    children=html.Div([
                        html.Div("Drag and Drop or Click to Upload", id="image-upload-hint-1", className="muted"),
                        html.Span("Supported formats: JPG, PNG, WEBP", id="image-upload-hint-2", className="tiny muted"),
                    ]),
                    multiple=False,
                    className="dropzone",
                ),
                daq.ToggleSwitch(
                    id="image-explain-ui",
                    value=True,
                    label="Show AI Explanation",
                    labelPosition="right",
                    size=35,
                    color="#3b82f6",
                    className="toggle"
                ),
                dcc.Store(id="image-explain-toggle", data=True),
            ], className="card"),

            html.Div([
                html.Div(id="image-progress", className="progress", children=html.Div(className="pbar")),
                dcc.Loading(html.Div(id="image-output", className="card result-card"), type="default", color="#3b82f6"),
            ]),
        ], className="grid grid-2"),
        html.Div(id="image-training-note", className="note muted tiny", style={"marginTop": "12px"}),
    ], className="tab-pane")


def _video_tab():
    return html.Div([
        html.Div([
            html.Div([
                html.H3([html.I(className="fas fa-video"), " Video Analysis"], id="video-title", className="card-title"),
                dcc.Upload(
                    id="video-upload",
                    children=html.Div([
                        html.Div("Drag and Drop or Click to Upload", id="video-upload-hint-1", className="muted"),
                        html.Span("Supported formats: MP4, AVI, MOV", id="video-upload-hint-2", className="tiny muted"),
                    ]),
                    multiple=False,
                    className="dropzone",
                ),
                daq.ToggleSwitch(
                    id="video-explain-ui",
                    value=True,
                    label="Show AI Explanation",
                    labelPosition="right",
                    size=35,
                    color="#3b82f6",
                    className="toggle"
                ),
                dcc.Store(id="video-explain-toggle", data=True),
            ], className="card"),

            html.Div([
                html.Div(id="video-progress", className="progress", children=html.Div(className="pbar")),
                dcc.Loading(html.Div(id="video-output", className="card result-card"), type="default", color="#3b82f6"),
            ]),
        ], className="grid grid-2"),
    ], className="tab-pane")


def get_layout():
    return html.Div([
        dcc.Store(id="theme-store", storage_type="memory"),
        dcc.Store(id="lang-store", data="en"),
        # Stores for AI results to enable dynamic translation
        dcc.Store(id="audio-result-store", storage_type="memory"),
        dcc.Store(id="image-result-store", storage_type="memory"),
        dcc.Store(id="video-result-store", storage_type="memory"),
        html.Div(id="theme-sync", style={"display": "none"}),
        _header(),
        html.Div([
            dcc.Tabs(
                id="tabs", value="tab-audio",
                className="tabs",
                parent_className="tabs-parent",
                content_className="tabs-content",
                children=[
                    dcc.Tab(id="tab-audio-label", label="Audio Detection", value="tab-audio", className="tab", selected_className="tab--selected"),
                    dcc.Tab(id="tab-image-label", label="Image Detection", value="tab-image", className="tab", selected_className="tab--selected"),
                    dcc.Tab(id="tab-video-label", label="Video Detection", value="tab-video", className="tab", selected_className="tab--selected"),
                ]
            ),
            html.Div(id="tab-content", className="tab-content")
        ], className="container"),
        _footer()
    ], id="root", className="theme--dark")
