import os
import json
import uuid
import base64
import logging
import pathlib
import subprocess
from pathlib import Path

import runpod
from runpod.serverless.utils import rp_upload
from huggingface_hub import snapshot_download, hf_hub_download


# =========================
# Defaults for SERVERLESS
# =========================
# In RunPod Serverless, your Network Volume mounts at /runpod-volume.
# We'll cache/download models there on first run and reuse thereafter.
WEIGHTS_DIR   = os.getenv("WEIGHTS_DIR", "/runpod-volume/weights")
CKPT_DIR      = os.getenv("WAN_CKPT_DIR", f"{WEIGHTS_DIR}/Wan2.1-I2V-14B-480P")
W2V_DIR       = os.getenv("WAV2VEC_DIR", f"{WEIGHTS_DIR}/chinese-wav2vec2-base")
IT_WEIGHTS    = os.getenv("INF_TALK_WEIGHTS", f"{WEIGHTS_DIR}/InfiniteTalk/single/infinitetalk.safetensors")

PYTHON_BIN    = os.getenv("PYTHON_BIN", "python")
DEFAULT_SIZE  = os.getenv("DEFAULT_SIZE", "infinitetalk-480")  # or "infinitetalk-720"
DEFAULT_MODE  = os.getenv("DEFAULT_MODE", "streaming")         # "streaming" | "clip"
SAMPLE_STEPS  = int(os.getenv("SAMPLE_STEPS", "40"))
MOTION_FRAME  = int(os.getenv("MOTION_FRAME", "9"))


# -------------------------
# One-time model bootstrap
# -------------------------
def ensure_weights():
    """
    Ensure required model folders/files exist on the mounted volume.
    Downloads from Hugging Face on first run, then reuses cached files.
    """
    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

    # 1) Wan 2.1 I2V 14B (480p)
    if not Path(CKPT_DIR).exists():
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir=CKPT_DIR,
            local_dir_use_symlinks=False
        )

    # 2) wav2vec2 audio encoder (+ extra model.safetensors on refs/pr/1)
    if not Path(W2V_DIR).exists():
        snapshot_download(
            repo_id="TencentGameMate/chinese-wav2vec2-base",
            local_dir=W2V_DIR,
            local_dir_use_symlinks=False
        )
        try:
            hf_hub_download(
                repo_id="TencentGameMate/chinese-wav2vec2-base",
                filename="model.safetensors",
                revision="refs/pr/1",
                local_dir=W2V_DIR,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            # Optional file — keep going if this specific revision isn't available.
            logging.warning(f"Optional wav2vec2 model.safetensors fetch failed: {e}")

    # 3) InfiniteTalk weights (repo contains single/multi/quant variants)
    it_root = Path(WEIGHTS_DIR) / "InfiniteTalk"
    if not it_root.exists():
        snapshot_download(
            repo_id="MeiGen-AI/InfiniteTalk",
            local_dir=str(it_root),
            local_dir_use_symlinks=False
        )

    # Sanity: ensure the expected single-model file exists
    if not Path(IT_WEIGHTS).exists():
        # Try to locate any suitable *.safetensors under InfiniteTalk/single
        candidates = list((it_root / "single").glob("*.safetensors"))
        if candidates:
            # Use the first found path if env default doesn't exist yet
            global IT_WEIGHTS
            IT_WEIGHTS = str(candidates[0])
        else:
            raise FileNotFoundError(
                "InfiniteTalk 'single' weights not found after download. "
                f"Expected something under: {it_root/'single'}"
            )


# -------------------------
# Helpers
# -------------------------
def _download_to(path: str, url_or_b64: str):
    """
    Save input (http(s) URL or base64 string) to path.
    """
    if url_or_b64.startswith(("http://", "https://")):
        import requests  # lazy import to keep cold starts lean
        r = requests.get(url_or_b64, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    else:
        # assume base64
        with open(path, "wb") as f:
            f.write(base64.b64decode(url_or_b64))


# -------------------------
# Handler
# -------------------------
def handler(event):
    """
    Input:
    {
      "image": "<https url or base64>",     # required
      "audio": "<https url or base64>",     # required (wav recommended)
      "size": "infinitetalk-480|infinitetalk-720",
      "mode": "streaming|clip",
      "sample_steps": 40,
      "motion_frame": 9,
      "extra_args": ["--num_persistent_param_in_dit","0"]  # optional
    }

    Output (on success):
    {
      "video_url": "https://.../file.mp4",
      "args": {...},
      "logs": "tail of process logs"
    }
    """
    # 0) Make sure weights exist (first run will download & cache them)
    ensure_weights()

    job_id = event.get("id", str(uuid.uuid4()))
    inp = event.get("input", {})

    # Validate required inputs
    if "image" not in inp or "audio" not in inp:
        return {"error": "Both 'image' and 'audio' are required."}

    size         = inp.get("size", DEFAULT_SIZE)
    mode         = inp.get("mode", DEFAULT_MODE)
    # sample_steps = int(inp.get("sample_steps", SAMPLED_STEPS)) if "sample_steps" in inp else SAMPLE_STEPS  # type: ignore
    sample_steps = int(inp.get("sample_steps", SAMPLE_STEPS))
    motion_frame = int(inp.get("motion_frame", MOTION_FRAME))
    extra_args   = [str(x) for x in inp.get("extra_args", [])]

    work = pathlib.Path(f"/tmp/{job_id}")
    work.mkdir(parents=True, exist_ok=True)

    # 1) Materialize inputs
    img_path = str(work / "input.jpg")
    wav_path = str(work / "input.wav")
    _download_to(img_path, inp["image"])
    _download_to(wav_path, inp["audio"])

    # 2) Build the JSON spec the repo expects
    spec = {"type": "image", "image_path": img_path, "audio_path": wav_path}
    spec_path = str(work / "single_example_image.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f)

    # 3) Call the repo's generator
    out_stem = str(work / "infinitetalk_res")
    cmd = [
        PYTHON_BIN, "generate_infinitetalk.py",
        "--ckpt_dir", CKPT_DIR,
        "--wav2vec_dir", W2V_DIR,
        "--infinitetalk_dir", IT_WEIGHTS,
        "--input_json", spec_path,
        "--size", size,
        "--sample_steps", str(sample_steps),
        "--mode", mode,
        "--motion_frame", str(motion_frame),
        "--save_file", out_stem,
        *extra_args  # pass-through any extra CLI flags
    ]

    logging.info("Running: %s", " ".join(cmd))

    try:
        run = subprocess.run(
            cmd,
            cwd="/workspace",              # repo root in our container
            capture_output=True,
            text=True,
            timeout=60 * 60
        )
        logs = (run.stdout or "") + "\n\n" + (run.stderr or "")
        if run.returncode != 0:
            return {"error": f"Generator exited with code {run.returncode}", "logs": logs[-5000:]}
    except Exception as e:
        return {"error": f"subprocess failed: {e}"}

    # 4) Find the output video
    candidates = list(pathlib.Path(work).glob("infinitetalk_res*.mp4"))
    if not candidates:
        # sometimes tools emit to CWD—scan repo dir as a fallback
        candidates = list(pathlib.Path("/workspace").glob("infinitetalk_res*.mp4")) or list(pathlib.Path(work).glob("*.mp4"))
    if not candidates:
        return {"error": "No MP4 produced.", "logs": logs[-5000:]}

    video_path = str(candidates[0])

    # 5) Upload to RunPod's CDN helper and return URL
    uploaded = rp_upload.upload_file(job_id=job_id, file_path=video_path)  # {"file_id": "...", "url": "..."}
    return {
        "video_url": uploaded.get("url"),
        "args": {
            "size": size,
            "mode": mode,
            "sample_steps": sample_steps,
            "motion_frame": motion_frame,
            "extra_args": extra_args
        },
        "logs": logs[-5000:]
    }


# Start the serverless worker loop
runpod.serverless.start({"handler": handler})
