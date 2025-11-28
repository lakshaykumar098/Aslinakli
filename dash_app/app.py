from dash import Dash
from layout import get_layout
from callbacks import register_callbacks

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="AsliNakli • Deepfake Detection",
    update_title=None,
    assets_folder="assets",
)

app.layout = get_layout()
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=8050)
