import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

# If your checkpoint needs SpeechBrain's class in scope during load, keep this import:
try:
    from speechbrain.inference.classifiers import EncoderClassifier  # noqa: F401
except Exception:
    EncoderClassifier = None  # not strictly required if your model is a plain torch nn.Module

from torch.serialization import safe_globals  # <-- allowlist context

MODEL_PATH = "models/Rawnetlite_Model_Output.pt"

def load_audio_model():
    """
    Loads your saved RawNetLite model on CPU.
    Works whether it's a plain torch.nn.Module or a SpeechBrain EncoderClassifier wrapper.

    Fixes PyTorch 2.6 behavior by:
      - allowlisting EncoderClassifier for unpickling
      - forcing weights_only=False (since checkpoint stores objects)
    """
    if EncoderClassifier is not None:
        # Allowlist the SpeechBrain class referenced by the pickle
        with safe_globals([EncoderClassifier]):
            model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    else:
        # If the import failed, still try full unpickle (only do this if you trust the file)
        model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    model.eval()
    return model

# ---------------------------
# Core helpers (simple & robust)
# ---------------------------

def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Normalize logits to a flat 1D tensor of length 2 *without* changing class order.
    Expected order in your project: [FAKE, REAL].
    """
    logits = logits.squeeze()
    if logits.ndim > 1:
        logits = logits.reshape(-1)
    if logits.numel() < 2:
        raise RuntimeError(f"Expected at least 2 logits, got shape {tuple(logits.shape)}")
    if logits.numel() > 2:
        logits = logits[:2]  # keep FIRST two (preserve order)
    return logits

def forward_logits(model, wave: torch.Tensor, sr: int, require_grad: bool = False) -> torch.Tensor:
    """
    Returns raw logits for [FAKE, REAL] for a single mono waveform.
    Accepts either:
      - SpeechBrain EncoderClassifier (encode_batch + mods.classifier), or
      - a plain torch model that returns logits directly.
    """
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)  # [1, T]
    wave = wave.float()

    logits = None
    try:
        if hasattr(model, "encode_batch") and hasattr(getattr(model, "mods", None), "classifier"):
            if require_grad:
                emb = model.encode_batch(wave)            # [1, D]
                logits = model.mods.classifier(emb)       # [1, 2]
            else:
                with torch.no_grad():
                    emb = model.encode_batch(wave)        # [1, D]
                    logits = model.mods.classifier(emb)   # [1, 2]
    except Exception:
        logits = None

    if logits is None:
        if require_grad:
            logits = model(wave)  # [1, 2] or [2]
        else:
            with torch.no_grad():
                logits = model(wave)

    return _normalize_logits(logits)

def _confidence_band(p: float) -> str:
    pct = int(round(p * 100))
    if pct < 60:
        return f"Inconclusive ({pct}%)"
    elif pct < 80:
        return f"Possibly ({pct}%)"
    elif pct < 95:
        return f"Likely ({pct}%)"
    else:
        return f"Very likely ({pct}%)"

def predict_audio(model, wave: torch.Tensor, sr: int) -> Dict:
    """
    Human-readable prediction.
    Output:
      {
        "probs": {"real": float, "fake": float},
        "verdict": "Likely REAL|FAKE" | "Inconclusive",
        "band": "Likely (88%)",
        "label": "REAL" | "FAKE" | "UNSURE"
      }
    """
    logits = forward_logits(model, wave, sr, require_grad=False)
    # IMPORTANT: your project uses [FAKE, REAL]
    probs = F.softmax(logits, dim=-1)  # [FAKE, REAL]
    p_fake = float(probs[0].item())
    p_real = float(probs[1].item())

    top_p = max(p_real, p_fake)
    if top_p < 0.60:
        label = "UNSURE"
        verdict = "Inconclusive"
        band = _confidence_band(top_p)
    elif p_fake >= p_real:
        label = "FAKE"
        band = _confidence_band(p_fake)
        verdict = f"{band.split()[0]} FAKE"
    else:
        label = "REAL"
        band = _confidence_band(p_real)
        verdict = f"{band.split()[0]} REAL"

    return {
        "probs": {"real": p_real, "fake": p_fake},
        "verdict": verdict,
        "band": band,
        "label": label,
    }

# ---------------------------
# Simple XAI: Gradient × Input over time
# ---------------------------

def explain_audio_simple(
    model,
    wave: torch.Tensor,
    sr: int,
    window_ms: int = 80,
    hop_ms: int = 40,
) -> Dict:
    """
    Lightweight saliency on the target logit (FAKE if predicted FAKE, else REAL).
    Returns top 1–2 time spans + a one-line reason.
    """
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    wave = wave.float()
    wave = wave.clone().detach().requires_grad_(True)

    with torch.no_grad():
        logits_pre = forward_logits(model, wave.detach(), sr, require_grad=False)  # [FAKE, REAL]
        probs = F.softmax(logits_pre, dim=-1)
        target_idx = int(torch.argmax(probs).item())  # 0=FAKE, 1=REAL

    logits = forward_logits(model, wave, sr, require_grad=True)  # [FAKE, REAL]
    logits = logits.view(-1)
    target_logit = logits[target_idx]  # scalar

    grad = torch.autograd.grad(target_logit, wave, retain_graph=False, create_graph=False)[0]  # [1, T]
    saliency = (grad * wave).abs().detach().squeeze(0)  # [T]

    win = max(int(sr * (window_ms / 1000.0)), 1)
    hop = max(int(sr * (hop_ms / 1000.0)), 1)
    T = saliency.numel()
    scores: List[Tuple[float, int, int]] = []
    for start in range(0, max(0, T - win) + 1, hop):
        end = min(start + win, T)
        if end > start:
            score = float(saliency[start:end].mean().item())
            scores.append((score, start, end))

    if not scores:
        return {"spans": [], "reason": "Audio too short to analyze", "duration_sec": T / float(sr)}

    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[:2]
    spans: List[Tuple[float, float]] = [(s / sr, e / sr) for _, s, e in top]

    if len(top) >= 1:
        reason = "Most influence around {:.1f}–{:.1f}s".format(spans[0][0], spans[0][1])
        if len(top) == 2:
            reason += " (also {:.1f}–{:.1f}s)".format(spans[1][0], spans[1][1])
    else:
        reason = "Most influence concentrated in a short segment"

    return {
        "spans": spans,
        "reason": reason,
        "duration_sec": T / float(sr),
    }
