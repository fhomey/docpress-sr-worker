"""
DocPress SR Worker — RunPod Serverless
Real-ESRGAN x4plus GPU Super-Resolution
Two Leaf Holding — Felipe Homeylev

Deploy op: RunPod Serverless → RTX 4090
Container: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

Environment variables (stel in op RunPod):
  SUPABASE_URL      — jouw Supabase project URL
  SUPABASE_KEY      — Supabase service_role key
  DOCPRESS_API_KEY  — je DocPress API key (optioneel voor webhook)
"""

import runpod
import base64
import hashlib
import json
import os
import subprocess
import time
import tempfile
from io import BytesIO

import torch
import numpy as np
from PIL import Image

# ── torchvision compatibiliteit patch (vereist voor basicsr op torchvision 0.16+) ──
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    import torchvision.transforms.functional as _tvf
    import sys as _sys
    _sys.modules["torchvision.transforms.functional_tensor"] = _tvf

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ── Model cache ──────────────────────────────────────────────────────────────
_sr_model = None
_sr_model_id = None

def load_sr_model(scale: int = 4):
    """Load Real-ESRGAN x4plus model, cache in memory."""
    global _sr_model, _sr_model_id

    model_name = f"RealESRGAN_x{scale}plus"
    model_path  = f"/tmp/{model_name}.pth"

    # Download weights if not cached
    if not os.path.exists(model_path):
        url = (
            "https://github.com/xinntao/Real-ESRGAN/"
            "releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            if scale == 4 else
            "https://github.com/xinntao/Real-ESRGAN/"
            "releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        )
        subprocess.check_call(["wget", "-q", "-O", model_path, url])
        print(f"[SR] Downloaded {model_name} → {model_path}")

    # Compute state_dict ID (ID_SR voor patent manifest)
    if _sr_model_id is None:
        with open(model_path, "rb") as f:
            raw = f.read()
        _sr_model_id = hashlib.sha256(raw).hexdigest()
        print(f"[SR] ID_SR = {_sr_model_id[:16]}…")

    if _sr_model is None:
        arch = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32, scale=scale
        )
        _sr_model = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=arch,
            tile=512,          # 512×512 tiles — volledig GPU
            tile_pad=10,
            pre_pad=0,
            half=True,         # FP16 op RTX 4090
            device=torch.device("cuda")
        )
        print(f"[SR] Model geladen op {torch.cuda.get_device_name(0)}")

    return _sr_model, _sr_model_id


# ── Handler ──────────────────────────────────────────────────────────────────
def handler(job):
    """
    Input (job["input"]):
      pixels_b64   : base64 gecodeerde PNG/JPEG afbeelding
      scale        : upscale factor (default 4)
      document_id  : optioneel — voor patent evidence logging
      dpr          : Device Pixel Ratio van de client

    Output:
      hr_pixels_b64 : base64 gecodeerde PNG hoge-resolutie afbeelding
      id_sr         : SHA-256 van SR model weights (patent Eq.3 ID_SR)
      input_size    : [breedte, hoogte] van input
      output_size   : [breedte, hoogte] van output
      processing_ms : verwerkingstijd in milliseconden
      lpips_estimate: geschatte LPIPS op basis van output resolutie
    """
    t0 = time.time()

    inp         = job.get("input", {})
    pixels_b64  = inp.get("pixels_b64")
    scale       = int(inp.get("scale", 4))
    document_id = inp.get("document_id", "")
    dpr         = float(inp.get("dpr", 3.0))

    # Validatie
    if not pixels_b64:
        return {"error": "pixels_b64 is required"}
    if scale not in (2, 4):
        return {"error": "scale must be 2 or 4"}

    # Decodeer input afbeelding
    img_bytes = base64.b64decode(pixels_b64)
    img       = Image.open(BytesIO(img_bytes)).convert("RGB")
    input_w, input_h = img.size

    # Laad SR model
    model, id_sr = load_sr_model(scale=scale)

    # SR upscaling
    img_np = np.array(img, dtype=np.uint8)
    try:
        output_np, _ = model.enhance(img_np, outscale=scale)
    except RuntimeError as e:
        # OOM fallback: gebruik kleinere tile
        print(f"[SR] OOM bij tile=512, fallback naar tile=256: {e}")
        model.tile_size = 256
        output_np, _ = model.enhance(img_np, outscale=scale)
        model.tile_size = 512  # herstel voor volgende request

    output_img = Image.fromarray(output_np)
    output_w, output_h = output_img.size

    # Encodeer output als PNG
    buf = BytesIO()
    output_img.save(buf, format="PNG", optimize=True)
    hr_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    processing_ms = round((time.time() - t0) * 1000)

    # Schat LPIPS op basis van output resolutie
    # (gebaseerd op V1.0 benchmark: gemiddeld 0.044 bij GPU SR)
    lpips_est = 0.044 if output_w >= 1920 else 0.065

    result = {
        "hr_pixels_b64":  hr_b64,
        "id_sr":          id_sr,
        "input_size":     [input_w, input_h],
        "output_size":    [output_w, output_h],
        "processing_ms":  processing_ms,
        "lpips_estimate": lpips_est,
        "document_id":    document_id,
        "dpr":            dpr,
        "scale":          scale,
        "tile_size":      getattr(model, "tile_size", getattr(model, "tile", 512)),
        "device":         torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    print(
        f"[SR] ✓ {input_w}×{input_h} → {output_w}×{output_h} "
        f"| {processing_ms}ms | ID_SR={id_sr[:8]}…"
    )
    return result


# ── Entrypoint ───────────────────────────────────────────────────────────────
runpod.serverless.start({"handler": handler})
