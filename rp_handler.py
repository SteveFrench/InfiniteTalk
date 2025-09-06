import os
import json
import uuid
import base64
import logging
import subprocess
from pathlib import Path
from typing import Iterable, List

import runpod
from runpod.serverless.utils import rp_upload
from huggingface_hub import snapshot_download, hf_hub_download


# =========================
# Config (Serverless-first)
# =========================
# In RunPod **Serverless**, your Network Volume mounts at /runpod-volume.
# We default all weights there and auto-download on first run.
WEIGHTS_DIR   = os.getenv("WEIGHTS_DIR", "/runpod-volume/weights")
CKPT_DIR      = os.getenv("WAN_CKPT_DIR", f"{WEIGHTS_DIR}/Wan2.1-I2V-14B-480P")
W2V_DIR       = os.getenv("WAV2VEC_DIR", f"{WEIGHTS_DIR}/chinese-wav2vec2-base")
IT_WEIGHTS    = os.getenv("INF_TALK_WEIGHTS", f"{WEIGHTS_DIR}/InfiniteTalk/single/infinitetalk.safetensors")

PYTHON_BIN    = os.getenv("PYTHON_BIN", "python")
DEFAULT_SIZE  = os.getenv("DEFAULT_SIZE", "infinitetalk-480")   # or "infinitetalk-720"
DEFAULT_MODE  = os.getenv("DEFAULT_MODE", "streaming")          # "streaming" | "clip"
SAMPLE_STEPS  = int(os.getenv("SAMPLE_STEPS", "40"))
MOTION_FRAME  = int(os.getenv("MOTION_FRAME", "9"))

HF_TOKEN      = os.getenv("HF_TOKEN")  # optional; required if repos are gated

# Logging level (INFO by default)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# -------------------------
# Utilities
# -------------------------
def _download_to(path: Path, url_or_b64: str):
    """
    Save input (http(s) URL or base64 string) to path.
    """
    if url_or_b64.startswith(("http://", "https://")):
        import requests  # lazy import for smaller cold starts
        r = requests.get(url_or_b64, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    else:
        # assume base64
        path.write_bytes(base64.b64decode(url_or_b64))


def _tree_snapshot(root: Path, depth: int = 3, file_cap_per_dir: int = 25) -> str:
    import os as _os
    lines: List[str] = []
    if not root.exists():
        return f"(missing) {root}"
    root = root.resolve()
    for cur_dir, subdirs, files in _os.walk(root):
        rel_parts = Path(cur_dir).resolve().relative_to(root).parts
        if len(rel_parts) > depth:
            continue
        indent = "  " * len(rel_parts)
        lines.append(f"{indent}{Path(cur_dir).name}/")
        for f in sorted(files)[:file_cap_per_dir]:
            lines.append(f"{indent}  {f}")
    return "\n".join(lines)


# -------------------------
# One-time model bootstrap (robust)
# -------------------------
def ensure_weights_and_get_it_path() -> str:
    """
    Ensure required model folders/files exist on the mounted volume.
    Downloads from Hugging Face on first run, then returns the resolved
    path to an InfiniteTalk *single* weights file (.safetensors).

    Search order prefers:
      - 'single/**/infinitetalk*.safetensors'
      - 'single/**/*.safetensors'
      - '**/infinitetalk*.safetensors'
      - '**/*.safetensors'
    """
    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)

    # 1) Wan 2.1 I2V 14B (480p) base video model
    if not Path(CKPT_DIR).exists():
        logging.info("[weights] downloading Wan-AI/Wan2.1-I2V-14B-480P -> %s", CKPT_DIR)
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir=CKPT_DIR,
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )

    # 2) wav2vec2 audio encoder (+ optional extra model.safetensors on refs/pr/1)
    if not Path(W2V_DIR).exists():
        logging.info("[weights] downloading TencentGameMate/chinese-wav2vec2-base -> %s", W2V_DIR)
        snapshot_download(
            repo_id="TencentGameMate/chinese-wav2vec2-base",
            local_dir=W2V_DIR,
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )
        try:
            hf_hub_download(
                repo_id="TencentGameMate/chinese-wav2vec2-base",
                filename="model.safetensors",
                revision="refs/pr/1",
                local_dir=W2V_DIR,
                local_dir_use_symlinks=False,
                token=HF_TOKEN
            )
        except Exception as e:
            logging.warning("[weights] optional wav2vec2 model.safetensors fetch failed: %s", e)

    # 3) InfiniteTalk weights (download likely filenames explicitly, then snapshot)
    it_root = Path(WEIGHTS_DIR) / "InfiniteTalk"
    it_root.mkdir(parents=True, exist_ok=True)

    # Try a repo snapshot first (harmless if already populated)
    try:
        if not list(it_root.glob("**/*")):
            logging.info("[weights] downloading MeiGen-AI/InfiniteTalk -> %s", it_root)
            snapshot_download(
                repo_id="MeiGen-AI/InfiniteTalk",
                local_dir=str(it_root),
                local_dir_use_symlinks=False,
                token=HF_TOKEN
            )
    except Exception as e:
        logging.warning("[weights] snapshot_download MeiGen-AI/InfiniteTalk failed: %s", e)

    # Then try targeted file downloads for common names/paths
    candidate_names = [
        "comfyui/infinitetalk_single.safetensors",
        "infinitetalk_single.safetensors",
        "single/infinitetalk.safetensors",
        "Wan2_1-InfiniTetalk-Single_fp16.safetensors",
    ]
    for name in candidate_names:
        try:
            hf_hub_download(
                repo_id="MeiGen-AI/InfiniteTalk",
                filename=name,
                local_dir=str(it_root),
                local_dir_use_symlinks=False,
                token=HF_TOKEN
            )
            logging.info("[weights] downloaded explicit file: %s", name)
        except Exception:
            pass  # try the next candidate

    # If INF_TALK_WEIGHTS explicitly points to a file and it exists, use it.
    explicit = Path(IT_WEIGHTS)
    if explicit.is_file():
        logging.info("[weights] using explicit InfiniteTalk weights: %s", explicit)
        return str(explicit)

    # Otherwise, search recursively for any *.safetensors under InfiniteTalk
    candidates = sorted([p.resolve() for p in it_root.rglob("*.safetensors") if p.is_file()])

    if candidates:
        # Prefer 'single' > non-quant > shortest path
        def score(path: Path) -> tuple[int, int]:
            s = 0
            pl = str(path).lower()
            if "single" in pl:
                s -= 3
            if "quant" in pl or "fp8" in pl or "int8" in pl:
                s += 1
            return (s, len(str(path)))
        best = sorted(candidates, key=score)[0]
        logging.info("[weights] resolved InfiniteTalk weights to: %s", best)
        return str(best)

    # Nothing found — print a small directory tree to help debugging.
    snapshot = _tree_snapshot(it_root, depth=4)
    logging.error("[weights] No *.safetensors found under %s\n%s", it_root, snapshot)
    raise FileNotFoundError(
        "InfiniteTalk weights not found after download. "
        f"Searched recursively under: {it_root}. "
        "If you have a specific file path, set INF_TALK_WEIGHTS to that path."
    )


# -------------------------
# Handler
# -------------------------
def handler(event):
    """
    Expects either of these input styles:

    A) rp_handler-native (preferred)
       {
         "image": "<https url or base64>",
         "audio": "<https url or base64>",
         "size": "infinitetalk-480|infinitetalk-720",
         "mode": "streaming|clip",
         "sample_steps": 40,
         "motion_frame": 9,
         "extra_args": ["--num_persistent_param_in_dit","0"]
       }

    B) Alternate (maps to A for convenience)
       {
         "image_path": "<https url>",
         "audio_path": "<https url>" OR
         "audio_paths": {"person1": "<https url>", ...},
         "num_frames": 201,
         "max_frames": 201,
         "mode": "streaming"
       }
    """
    # 0) Ensure models exist (first run will download and cache on the volume)
    it_weights_path = ensure_weights_and_get_it_path()

    job_id = event.get("id", str(uuid.uuid4()))
    inp = event.get("input", {}) or {}

    # --- Accept both input styles ---
    image = inp.get("image") or inp.get("image_path")
    audio = inp.get("audio") or inp.get("audio_path")

    if not audio and isinstance(inp.get("audio_paths"), dict) and inp["audio_paths"]:
        # pick the first URL in the dict
        audio = next(iter(inp["audio_paths"].values()))

    if not image or not audio:
        return {"error": "Both 'image' (or 'image_path') and 'audio' (or 'audio_path'/'audio_paths') are required."}

    size         = inp.get("size", DEFAULT_SIZE)
    mode         = inp.get("mode", DEFAULT_MODE)
    sample_steps = int(inp.get("sample_steps", SAMPLE_STEPS))
    motion_frame = int(inp.get("motion_frame", MOTION_FRAME))

    # Build extra args from optional fields
    extra_args = [str(x) for x in inp.get("extra_args", [])]
    if "num_frames" in inp:
        extra_args += ["--num_frames", str(int(inp["num_frames"]))]
    if "max_frames" in inp:
        extra_args += ["--max_frames", str(int(inp["max_frames"]))]

    # Work directory for this job
    work = Path(f"/tmp/{job_id}")
    work.mkdir(parents=True, exist_ok=True)

    # 1) Materialize inputs to files
    img_path = work / "input.jpg"
    wav_path = work / "input.wav"
    _download_to(img_path, image)
    _download_to(wav_path, audio)

    # 2) Build the JSON spec the repo expects
    spec = {"type": "image", "image_path": str(img_path), "audio_path": str(wav_path)}
    spec_path = work / "single_example_image.json"
    spec_path.write_text(json.dumps(spec))

    # 3) Call the repo's generator script
    out_stem = work / "infinitetalk_res"
    cmd = [
        PYTHON_BIN, "generate_infinitetalk.py",
        "--ckpt_dir", CKPT_DIR,
        "--wav2vec_dir", W2V_DIR,
        "--infinitetalk_dir", it_weights_path,
        "--input_json", str(spec_path),
        "--size", size,
        "--sample_steps", str(sample_steps),
        "--mode", mode,
        "--motion_frame", str(motion_frame),
        "--save_file", str(out_stem),
        *extra_args,   # pass-through any extra CLI flags
    ]

    logging.info("[run] %s", " ".join(cmd))

    try:
        run = subprocess.run(
            cmd,
            cwd="/workspace",          # repo root inside the container
            capture_output=True,
            text=True,
            timeout=60 * 60
        )
        logs = (run.stdout or "") + "\n\n" + (run.stderr or "")
        if run.returncode != 0:
            return {"error": f"Generator exited with code {run.returncode}", "logs": logs[-5000:]}
    except Exception as e:
        return {"error": f"subprocess failed: {e}"}

    # 4) Find the output video (usual name pattern: infinitetalk_res*.mp4)
    candidates = list(work.glob("infinitetalk_res*.mp4"))
    if not candidates:
        # sometimes tools emit to CWD — scan repo dir as a fallback
        candidates = list(Path("/workspace").glob("infinitetalk_res*.mp4")) or list(work.glob("*.mp4"))
    if not candidates:
        return {"error": "No MP4 produced.", "logs": logs[-5000:]}

    video_path = str(candidates[0])

    # 5) Upload to RunPod CDN helper so caller gets a downloadable URL
    uploaded = rp_upload.upload_file(job_id=job_id, file_path=video_path)  # -> {"file_id": "...", "url": "..."}
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
