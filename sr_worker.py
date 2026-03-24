#!/usr/bin/env python3
"""
DocPress GPU SR Worker — RunPod Serverless Handler
Real-ESRGAN x4 inference on GPU (RTX 3080 / T4 / A100)

Input  (JSON via RunPod):
  { "input": { "image_b64": "<base64 PNG/JPEG>", "scale": 4 } }

Output (JSON):
  { "output": { "image_b64": "<base64 PNG>", "sr_model_id": "...", "elapsed_ms": ... } }

Deploy on RunPod:
  Container: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
  Handler:   python -u /handler.py
  Env vars:  SR_WEIGHTS_X4 (optional, defaults to official GitHub release)
"""

import runpod
import base64, hashlib, io, os, time, logging, numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s")
log = logging.getLogger("sr_worker")

SR_WEIGHTS_URL = os.getenv(
    "SR_WEIGHTS_X4",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
WEIGHTS_PATH = "/tmp/RealESRGAN_x4plus.pth"

# ── Model laden bij container startup ──────────────────────────────────────
_upsampler = None
_model_id: str = ""
_model_id_method: str = ""

def _load_model():
    global _upsampler, _model_id, _model_id_method

    if not os.path.exists(WEIGHTS_PATH):
        log.info("SR weights downloaden van %s", SR_WEIGHTS_URL)
        import urllib.request
        urllib.request.urlretrieve(SR_WEIGHTS_URL, WEIGHTS_PATH)
        log.info("Download klaar: %s (%.1f MB)",
                 WEIGHTS_PATH, os.path.getsize(WEIGHTS_PATH) / 1e6)

    # basicsr torchvision compatibiliteit patch
    try:
        import torchvision.transforms.functional_tensor  # type: ignore
    except ImportError:
        import torchvision.transforms.functional as _f
        import sys
        sys.modules["torchvision.transforms.functional_tensor"] = _f

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                  num_block=23, num_grow_ch=32, scale=4)
    _upsampler = RealESRGANer(
        scale=4,
        model_path=WEIGHTS_PATH,
        model=net,
        tile=512,           # volledige 512×512 tiles op GPU
        tile_pad=10,
        pre_pad=0,
        half=True if device == "cuda" else False,
        device=device,
    )

    # Model ID via state_dict SHA-256 (identiek aan Railway main.py)
    h = hashlib.sha256()
    for arr in _upsampler.model.state_dict().values():
        h.update(np.asarray(arr.cpu().float(), dtype=np.float32).tobytes())
    _model_id = h.hexdigest()
    _model_id_method = "state_dict"
    log.info("Real-ESRGAN x4 geladen  ID_SR=%s…  (%s)", _model_id[:16], _model_id_method)

# Warmup bij container start
_load_model()


# ── RunPod handler ─────────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    """
    RunPod serverless entry point.
    job["input"]:
        image_b64  : str   — base64-encoded PNG or JPEG
        scale      : int   — upscale factor (default 4)
        doc_id     : str   — optional, returned unchanged for correlation
    """
    t0 = time.time()
    inp = job.get("input", {})

    image_b64 = inp.get("image_b64")
    if not image_b64:
        return {"error": "image_b64 is required"}

    scale = int(inp.get("scale", 4))
    doc_id = inp.get("doc_id", "")

    try:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        log.info("SR start  doc_id=%s  input=%dx%d  scale=%d",
                 doc_id[:8] if doc_id else "?", img.width, img.height, scale)

        # Real-ESRGAN inference
        output_np, _ = _upsampler.enhance(img_np, outscale=scale)
        output_img = Image.fromarray(output_np)

        # Encode output als PNG
        buf = io.BytesIO()
        output_img.save(buf, "PNG", optimize=False)
        output_b64 = base64.b64encode(buf.getvalue()).decode()

        elapsed_ms = int((time.time() - t0) * 1000)
        log.info("SR klaar  doc_id=%s  output=%dx%d  elapsed=%dms",
                 doc_id[:8] if doc_id else "?",
                 output_img.width, output_img.height, elapsed_ms)

        return {
            "output": {
                "image_b64": output_b64,
                "output_width": output_img.width,
                "output_height": output_img.height,
                "input_width": img.width,
                "input_height": img.height,
                "scale": scale,
                "sr_model_id": _model_id,
                "sr_model_id_method": _model_id_method,
                "elapsed_ms": elapsed_ms,
                "doc_id": doc_id,
            }
        }

    except Exception as e:
        log.exception("SR handler fout: %s", e)
        return {"error": str(e)}


# ── Lokale test (zonder RunPod) ────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # Maak een test-image aan en stuur die door de SR pipeline
        test_img = Image.new("RGB", (128, 128), color=(100, 150, 200))
        buf = io.BytesIO()
        test_img.save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        result = handler({"input": {"image_b64": b64, "scale": 4, "doc_id": "test-local"}})
        if "output" in result:
            out = result["output"]
            print(f"✅ SR test OK: {out['input_width']}x{out['input_height']} → "
                  f"{out['output_width']}x{out['output_height']} in {out['elapsed_ms']}ms")
            print(f"   ID_SR: {out['sr_model_id'][:16]}… ({out['sr_model_id_method']})")
        else:
            print(f"❌ SR test FOUT: {result}")
        sys.exit(0)

    # RunPod serverless loop
    log.info("RunPod serverless worker starten…")
    runpod.serverless.start({"handler": handler})
