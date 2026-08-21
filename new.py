from __future__ import annotations

import asyncio
import glob
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from typing import Any

import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image

try:
    import spaces
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(**_kwargs):
            def decorate(function):
                return function
            return decorate
    spaces = _SpacesFallback()


ROOT = pathlib.Path(__file__).resolve().parent
COMFY = ROOT / "ComfyUI"
MODELS = COMFY / "models"
INPUT = COMFY / "input"
OUTPUT = COMFY / "output"
CUSTOM_NODES = COMFY / "custom_nodes"

# ============================================================
# Krea 2 model setup
# ============================================================
KREA_REPO = "Comfy-Org/Krea-2"
KREA_DIFFUSION_FILE = "krea2_turbo_fp8_scaled.safetensors"
DEFAULT_BASE_MODEL = KREA_DIFFUSION_FILE

TEXT_ENCODER_FILE = "qwen3vl_4b_fp8_scaled.safetensors"
VAE_FILE = "qwen_image_vae.safetensors"
KREA_EDIT_NODES = "https://github.com/lbouaraba/comfyui-krea2edit.git"

DOWNLOADS = [
    (
        KREA_REPO,
        "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        MODELS / "text_encoders" / TEXT_ENCODER_FILE,
        "Qwen3-VL text encoder",
    ),
    (
        KREA_REPO,
        "vae/qwen_image_vae.safetensors",
        MODELS / "vae" / VAE_FILE,
        "Krea 2 VAE",
    ),
]

# User-installed LoRAs are discovered automatically from this directory.
LORA_ROOT = MODELS / "loras"

SAMPLERS = [
    "euler", "euler_ancestral", "euler_a", "dpmpp_2m", "dpmpp_2m_sde",
    "dpmpp_sde", "heun", "lms",
]
SCHEDULERS = ["beta", "normal", "karras", "exponential", "sgm_uniform", "simple"]

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
MAX_WIDTH = 2048
MAX_HEIGHT = 2048
DEFAULT_TARGET_MP = 1.4
MAX_TARGET_MP = 4.0
DEFAULT_GROUNDING = 768
DEFAULT_REF_BOOST = 1.0
DEFAULT_STEPS = 8
DEFAULT_CFG = 1.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "beta"
DEFAULT_SEED = 2
MIN_GPU_SECONDS = int(os.environ.get("MIN_GPU_SECONDS", "45"))
MAX_GPU_SECONDS = int(os.environ.get("MAX_GPU_SECONDS", "300"))

_comfy_ready = False
_nodes_ready = False


def _run(command: list[str], cwd: pathlib.Path | None = None, check: bool = True) -> None:
    print("[setup]", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=check)


def _pip_install(arguments: list[str]) -> None:
    _run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *arguments], check=False)


def _install_filtered_requirements(path: pathlib.Path) -> None:
    if not path.exists():
        return
    blocked = {"torch", "torchvision", "torchaudio", "transformers", "huggingface-hub", "accelerate"}
    requirements: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        package = __import__("re").split(r"[<>=!~;\[\s]", item.lower().replace("_", "-"), maxsplit=1)[0]
        if package not in blocked:
            requirements.append(item)
    if requirements:
        _pip_install(requirements)


def _ensure_repo(path: pathlib.Path, url: str) -> None:
    if not path.exists():
        _run(["git", "clone", "--depth", "1", url, str(path)])


def _restore_utils_namespace() -> None:
    source = COMFY / "utils"
    target = COMFY / "utilities"
    if not source.exists() and target.exists():
        target.rename(source)
    if not source.exists():
        return
    import re
    for path in COMFY.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = re.sub(r"\bfrom utilities\b", "from utils", text)
        updated = re.sub(r"\bimport utilities\b", "import utils", updated)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _ensure_comfy() -> None:
    global _comfy_ready
    if _comfy_ready:
        return
    _ensure_repo(COMFY, "https://github.com/comfyanonymous/ComfyUI.git")
    _install_filtered_requirements(COMFY / "requirements.txt")
    CUSTOM_NODES.mkdir(parents=True, exist_ok=True)
    _ensure_repo(CUSTOM_NODES / "comfyui-krea2edit", KREA_EDIT_NODES)
    _restore_utils_namespace()
    for folder in ("diffusion_models", "text_encoders", "vae", "loras"):
        (MODELS / folder).mkdir(parents=True, exist_ok=True)
    INPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _comfy_ready = True


def _init_comfy_nodes() -> None:
    global _nodes_ready
    if _nodes_ready:
        return
    comfy_path = str(COMFY)
    sys.path = [item for item in sys.path if item != comfy_path]
    sys.path.insert(0, comfy_path)
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]
    os.chdir(COMFY)
    import execution
    import nodes
    import server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    loop.run_until_complete(nodes.init_extra_nodes())
    _nodes_ready = True


def _download_to_dest(repo: str, filename: str, destination: pathlib.Path, label: str) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    path = pathlib.PurePosixPath(filename)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    downloaded = pathlib.Path(
        hf_hub_download(
            repo_id=repo,
            filename=path.name,
            subfolder=None if str(path.parent) == "." else str(path.parent),
            local_dir=str(destination.parent),
            token=token,
        )
    )
    if downloaded.resolve() != destination.resolve():
        shutil.move(str(downloaded), str(destination))
    print(f"[models] ready: {label}", flush=True)


def _ensure_models(progress: gr.Progress | None = None) -> str:
    # Only the official Krea 2 Turbo diffusion model is used as the base model.
    base_destination = MODELS / "diffusion_models" / KREA_DIFFUSION_FILE
    _download_to_dest(
        KREA_REPO,
        f"diffusion_models/{KREA_DIFFUSION_FILE}",
        base_destination,
        "Krea 2 Turbo diffusion model",
    )
    for index, (repo, filename, destination, label) in enumerate(DOWNLOADS, start=1):
        if progress:
            progress(index / (len(DOWNLOADS) + 1), desc=f"downloading {label}")
        _download_to_dest(repo, filename, destination, label)
    return KREA_DIFFUSION_FILE


def _discover_loras() -> list[str]:
    """Return every user-installed .safetensors LoRA, recursively."""
    if not LORA_ROOT.exists():
        return []
    files = []
    for path in LORA_ROOT.rglob("*.safetensors"):
        if path.is_file():
            try:
                relative = path.resolve().relative_to(LORA_ROOT.resolve())
            except ValueError:
                continue
            files.append(relative.as_posix())
    return sorted(set(files), key=str.lower)


def _lora_display_name(relative_path: str) -> str:
    return pathlib.PurePosixPath(relative_path).name


def _ref(node: str, output: int = 0) -> list[Any]:
    return [node, output]


def _t2i_workflow() -> dict[str, Any]:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": DEFAULT_BASE_MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER_FILE, "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_FILE}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": _ref("1"), "shift": 4.0}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("2"), "text": ""}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("2"), "text": ""}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": _ref("4"), "positive": _ref("5"), "negative": _ref("6"),
            "latent_image": _ref("7"), "seed": DEFAULT_SEED, "steps": DEFAULT_STEPS,
            "cfg": DEFAULT_CFG, "sampler_name": DEFAULT_SAMPLER, "scheduler": DEFAULT_SCHEDULER,
            "denoise": 1.0,
        }},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": _ref("8"), "vae": _ref("3")}},
        "10": {"class_type": "SaveImage", "inputs": {"images": _ref("9"), "filename_prefix": "krea2_turbo"}},
    }


def _edit_workflow(has_second_reference: bool) -> dict[str, Any]:
    workflow: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER_FILE, "type": "krea2", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_FILE}},
        "5": {"class_type": "UNETLoader", "inputs": {"unet_name": DEFAULT_BASE_MODEL, "weight_dtype": "default"}},
        "6": {"class_type": "Krea2EditModelPatch", "inputs": {
            "model": _ref("5"), "source_latent": _ref("7"), "vae": _ref("4"),
            "source_image": _ref("1"), "target_latent": _ref("8"),
            "ref_boost": DEFAULT_REF_BOOST, "ref_boost_a": DEFAULT_REF_BOOST, "fit_mode": "fit",
        }},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": _ref("1"), "vae": _ref("4")}},
        "8": {"class_type": "EmptySD3LatentImage", "inputs": {"width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT, "batch_size": 1}},
        "9": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": _ref("6"), "shift": 4.0}},
        "10": {"class_type": "Krea2EditGroundedEncode", "inputs": {"clip": _ref("3"), "image": _ref("1"), "prompt": "", "grounding_px": DEFAULT_GROUNDING}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"clip": _ref("3"), "text": ""}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": _ref("9"), "positive": _ref("10"), "negative": _ref("11"),
            "latent_image": _ref("8"), "seed": DEFAULT_SEED, "steps": DEFAULT_STEPS,
            "cfg": DEFAULT_CFG, "sampler_name": DEFAULT_SAMPLER, "scheduler": DEFAULT_SCHEDULER,
            "denoise": 1.0,
        }},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": _ref("12"), "vae": _ref("4")}},
        "14": {"class_type": "SaveImage", "inputs": {"images": _ref("13"), "filename_prefix": "krea2_edit"}},
    }
    if has_second_reference:
        workflow["2"] = {"class_type": "LoadImage", "inputs": {"image": ""}}
        workflow["15"] = {"class_type": "VAEEncode", "inputs": {"pixels": _ref("2"), "vae": _ref("4")}}
        workflow["6"]["inputs"]["source_latent_b"] = _ref("15")
        workflow["6"]["inputs"]["source_image_b"] = _ref("2")
        workflow["10"]["inputs"]["image_b"] = _ref("2")
    return workflow


def _inject_lora_chain(
    workflow: dict[str, Any],
    enabled_loras: list[tuple[str, float]],
    *,
    model_source: list[Any],
    clip_source: list[Any],
    model_consumers: list[tuple[str, str]],
    clip_consumers: list[tuple[str, str]],
) -> None:
    if not enabled_loras:
        for node_id, input_name in model_consumers:
            workflow[node_id]["inputs"][input_name] = model_source
        for node_id, input_name in clip_consumers:
            workflow[node_id]["inputs"][input_name] = clip_source
        return

    previous_model = model_source
    previous_clip = clip_source
    for index, (filename, strength) in enumerate(enabled_loras):
        node_id = f"user_lora_{index}"
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": previous_model,
                "clip": previous_clip,
                "lora_name": filename,
                "strength_model": float(strength),
                "strength_clip": float(strength),
            },
        }
        previous_model = _ref(node_id)
        previous_clip = _ref(node_id, 1)

    for node_id, input_name in model_consumers:
        workflow[node_id]["inputs"][input_name] = previous_model
    for node_id, input_name in clip_consumers:
        workflow[node_id]["inputs"][input_name] = previous_clip


def _prepare_edit_image(path: str, target_megapixels: float) -> tuple[str, int, int]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        megapixels = max(0.25, min(MAX_TARGET_MP, float(target_megapixels)))
        scale = (megapixels * 1_000_000 / max(1, image.width * image.height)) ** 0.5
        width = max(64, int(round(image.width * scale / 64) * 64))
        height = max(64, int(round(image.height * scale / 64) * 64))
        width = min(MAX_WIDTH, width)
        height = min(MAX_HEIGHT, height)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        name = f"input_{uuid.uuid4().hex[:12]}.png"
        image.save(INPUT / name, format="PNG")
    return name, width, height


def _stage_image(path: str, prefix: str) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        name = f"{prefix}_{uuid.uuid4().hex[:12]}.png"
        image.save(INPUT / name, format="PNG")
    return name


def _execute_workflow(workflow: dict[str, Any]) -> list[str]:
    import execution
    import server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    executor = execution.PromptExecutor(
        server_instance,
        cache_type=execution.CacheType.RAM_PRESSURE,
        cache_args={"lru": 0, "ram": 2.0, "ram_inactive": 8.0},
    )
    prompt_id = str(uuid.uuid4())
    save_id = next(node_id for node_id, node in workflow.items() if node.get("class_type") == "SaveImage")
    executor.execute(workflow, prompt_id, extra_data={}, execute_outputs=[save_id])
    if not executor.success:
        message = executor.status_messages[-1] if executor.status_messages else "ComfyUI execution failed"
        raise RuntimeError(str(message))

    paths: list[pathlib.Path] = []
    for output in executor.history_result.get("outputs", {}).values():
        for items in output.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                base = OUTPUT if item.get("type", "output") == "output" else COMFY / item.get("type", "output")
                candidate = base / item.get("subfolder", "") / item["filename"]
                if candidate.exists():
                    paths.append(candidate)
    if not paths:
        paths = sorted(
            [pathlib.Path(item) for item in glob.glob(str(OUTPUT / "**" / "*.png"), recursive=True)],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    if not paths:
        raise RuntimeError("ComfyUI finished without an output image")
    return [str(path) for path in paths]


def _prepare_runtime(progress: gr.Progress | None = None) -> str:
    _ensure_comfy()
    resolved = _ensure_models(progress)
    _init_comfy_nodes()
    return resolved


def get_gpu_duration(*args: Any, **kwargs: Any) -> int:
    steps = kwargs.get("steps", args[11] if len(args) > 11 else DEFAULT_STEPS)
    width = kwargs.get("width", args[5] if len(args) > 5 else DEFAULT_WIDTH)
    height = kwargs.get("height", args[6] if len(args) > 6 else DEFAULT_HEIGHT)
    gen_budget = kwargs.get("gen_budget", args[17] if len(args) > 17 else 0)
    if gen_budget and int(gen_budget) > 0:
        return max(MIN_GPU_SECONDS, min(MAX_GPU_SECONDS, int(gen_budget)))
    lora_weights = kwargs.get("lora_weights", args[18] if len(args) > 18 else {}) or {}
    lora_count = sum(1 for value in lora_weights.values() if abs(float(value)) > 1e-6)
    estimate = int(35 + (int(width) * int(height) / 1_000_000) * int(steps) * 3.0 * (1 + 0.05 * lora_count))
    return max(MIN_GPU_SECONDS, min(MAX_GPU_SECONDS, estimate))


@spaces.GPU(duration=get_gpu_duration)
def generate(
    mode: str,
    prompt: str,
    negative_prompt: str,
    edit_prompt: str,
    primary_image: str | None,
    second_image: str | None,
    width: int,
    height: int,
    target_megapixels: float,
    grounding_px: int,
    ref_boost: float,
    ref_boost_a: float,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    randomize_seed: bool,
    gen_budget: float,
    lora_weights: dict[str, float] | None = None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[list[str], str, int]:
    effective_seed = random.randint(0, 2**32 - 1) if randomize_seed or int(seed) < 0 else int(seed)
    staged: list[pathlib.Path] = []
    try:
        if mode not in {"text2image", "edit"}:
            raise ValueError("unsupported generation mode")
        if mode == "text2image" and not (prompt or "").strip():
            raise ValueError("enter a prompt")
        if mode == "edit":
            if not primary_image:
                raise ValueError("upload a primary image for edit mode")
            if not (edit_prompt or prompt or "").strip():
                raise ValueError("enter an edit instruction")
        if sampler not in SAMPLERS or scheduler not in SCHEDULERS:
            raise ValueError("unsupported sampler or scheduler")

        _prepare_runtime(progress)

        enabled_loras: list[tuple[str, float]] = []
        available = set(_discover_loras())
        for filename, weight in (lora_weights or {}).items():
            numeric = float(weight)
            if abs(numeric) <= 1e-6:
                continue
            if filename not in available:
                raise ValueError(f"LoRA file is no longer available: {filename}")
            enabled_loras.append((filename.replace("\\", "/"), numeric))

        if mode == "text2image":
            width = max(512, min(MAX_WIDTH, int(width) // 64 * 64))
            height = max(512, min(MAX_HEIGHT, int(height) // 64 * 64))
            workflow = _t2i_workflow()
            _inject_lora_chain(
                workflow,
                enabled_loras,
                model_source=_ref("1"),
                clip_source=_ref("2"),
                model_consumers=[("4", "model")],
                clip_consumers=[("5", "clip"), ("6", "clip")],
            )
            workflow["5"]["inputs"]["text"] = (prompt or "").strip()
            workflow["6"]["inputs"]["text"] = (negative_prompt or "").strip()
            workflow["7"]["inputs"].update(width=width, height=height)
            workflow["8"]["inputs"].update(
                seed=effective_seed, steps=int(steps), cfg=float(cfg),
                sampler_name=sampler, scheduler=scheduler, denoise=1.0,
            )
        else:
            primary_name, width, height = _prepare_edit_image(primary_image, target_megapixels)
            staged.append(INPUT / primary_name)
            second_name = _stage_image(second_image, "reference") if second_image else None
            if second_name:
                staged.append(INPUT / second_name)
            workflow = _edit_workflow(bool(second_name))
            workflow["1"]["inputs"]["image"] = primary_name
            if second_name:
                workflow["2"]["inputs"]["image"] = second_name
            _inject_lora_chain(
                workflow,
                enabled_loras,
                model_source=_ref("5"),
                clip_source=_ref("3"),
                model_consumers=[("6", "model")],
                clip_consumers=[("10", "clip"), ("11", "clip")],
            )
            workflow["8"]["inputs"].update(width=width, height=height)
            workflow["6"]["inputs"].update(ref_boost=float(ref_boost), ref_boost_a=float(ref_boost_a))
            workflow["10"]["inputs"].update(
                prompt=(edit_prompt or prompt or "").strip(),
                grounding_px=int(grounding_px),
            )
            workflow["11"]["inputs"]["text"] = (negative_prompt or "").strip()
            workflow["12"]["inputs"].update(
                seed=effective_seed, steps=int(steps), cfg=float(cfg),
                sampler_name=sampler, scheduler=scheduler, denoise=1.0,
            )

        progress(0.35, desc=f"generating {mode}")
        result_paths = _execute_workflow(workflow)
        destination_dir = pathlib.Path(tempfile.mkdtemp(prefix="krea2_outputs_"))
        output_paths: list[str] = []
        for index, source in enumerate(result_paths):
            destination = destination_dir / f"output_{index}.png"
            shutil.copy2(source, destination)
            output_paths.append(str(destination))
        return output_paths, f"done — {len(output_paths)} image(s), seed {effective_seed}", effective_seed
    except Exception as exc:
        print(traceback.format_exc(), flush=True)
        raise gr.Error(f"generation failed: {str(exc)[:500]}") from exc
    finally:
        for path in staged:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _render_lora_panel(previous_json: str = "") -> str:
    try:
        previous = json.loads(previous_json or "{}")
        if not isinstance(previous, dict):
            previous = {}
    except Exception:
        previous = {}

    files = _discover_loras()
    rows = []
    for filename in files:
        value = previous.get(filename, 0)
        try:
            value = float(value)
        except Exception:
            value = 0
        if value.is_integer():
            value = int(value)
        safe_file = filename.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
        display = _lora_display_name(filename).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        rows.append(
            f'<div class="lora-row"><div class="lora-name" title="{safe_file}">{display}</div>'
            f'<input class="lora-strength" data-lora="{safe_file}" type="number" step="1" value="{value}"></div>'
        )
    if not rows:
        body = '<div class="lora-empty">No .safetensors files found in ComfyUI/models/loras yet.</div>'
    else:
        body = "".join(rows)
    return f'<div class="lora-list">{body}</div>'


LORA_JS = r"""
() => {
  const sync = () => {
    const panel = document.querySelector('#lora-panel');
    const target = document.querySelector('#lora-values-json textarea');
    if (!panel || !target) return;
    const data = {};
    panel.querySelectorAll('.lora-strength').forEach(el => {
      const name = el.dataset.lora;
      const value = Number(el.value || 0);
      data[name] = Number.isFinite(value) ? value : 0;
    });
    target.value = JSON.stringify(data);
    target.dispatchEvent(new Event('input', {bubbles: true}));
    target.dispatchEvent(new Event('change', {bubbles: true}));
  };
  if (!window.__kreaLoraBound) {
    window.__kreaLoraBound = true;
    document.addEventListener('input', e => {
      if (e.target && e.target.classList && e.target.classList.contains('lora-strength')) sync();
    });
    document.addEventListener('change', e => {
      if (e.target && e.target.classList && e.target.classList.contains('lora-strength')) sync();
    });
  }
  setTimeout(sync, 100);
}
"""


CSS = """
.lora-list { display:flex; flex-direction:column; gap:8px; max-height:430px; overflow-y:auto; padding:4px 2px; }
.lora-row { display:flex; align-items:center; gap:12px; padding:8px 10px; border:1px solid var(--border-color-primary); border-radius:8px; }
.lora-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-family:monospace; }
.lora-strength { width:100px; min-width:100px; }
.lora-empty { opacity:.7; padding:12px 4px; }
"""


def create_ui() -> gr.Blocks:
    # The HTML LoRA panel is regenerated periodically. Existing strengths are
    # carried through the hidden JSON state, while newly downloaded files get 0.
    lora_state = gr.Textbox(value="{}", visible=False, elem_id="lora-values-json")
    with gr.Blocks(title="Krea 2 Turbo Image Generator", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown("# 🖌️ Krea 2 Turbo Image Generator\nOfficial **Krea 2 Turbo** model with user-installed LoRAs. No built-in LoRA catalog or base-model selector.")

        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(["text2image", "edit"], value="text2image", label="mode")
                gr.Markdown(f"**Base model:** `Comfy-Org/Krea-2/diffusion_models/{KREA_DIFFUSION_FILE}`")

                with gr.Column(visible=False) as image_inputs:
                    primary = gr.Image(type="filepath", label="primary image / scene")
                    second = gr.Image(type="filepath", label="optional second reference")

                prompt = gr.Textbox(value="A cinematic portrait in soft natural light", label="prompt", lines=4)
                negative_prompt = gr.Textbox(label="negative prompt", lines=3, placeholder="blurry, low quality, distorted, artifacts...")
                edit_prompt = gr.Textbox(label="edit instruction", lines=3, visible=False, placeholder="recolor the jacket to matte black")

                with gr.Column() as t2i_resolution:
                    with gr.Row():
                        width = gr.Slider(512, MAX_WIDTH, value=DEFAULT_WIDTH, step=64, label="width")
                        height = gr.Slider(512, MAX_HEIGHT, value=DEFAULT_HEIGHT, step=64, label="height")

                with gr.Column(visible=False) as edit_controls:
                    target_mp = gr.Slider(0.25, MAX_TARGET_MP, value=DEFAULT_TARGET_MP, step=0.05, label="target megapixels")
                    grounding = gr.Slider(384, 1536, value=DEFAULT_GROUNDING, step=32, label="grounding resolution")
                    ref_boost = gr.Slider(0.0, 12.0, value=DEFAULT_REF_BOOST, step=0.1, label="primary reference strength")
                    ref_boost_a = gr.Slider(0.0, 12.0, value=DEFAULT_REF_BOOST, step=0.1, label="second reference strength")

                with gr.Accordion("LoRAs", open=True):
                    gr.Markdown(
                        "LoRAs are read directly from `ComfyUI/models/loras`. "
                        "**0 = disabled**. Any positive or negative value is used as the LoRA strength. "
                        "The number input arrows change strength by exactly **1**. New `.safetensors` files are picked up automatically."
                    )
                    lora_panel = gr.HTML(
                        value=lambda: _render_lora_panel("{}"),
                        every=2,
                        inputs=[lora_state],
                        elem_id="lora-panel",
                        js_on_load=LORA_JS,
                    )

                with gr.Accordion("sampling", open=False):
                    steps = gr.Slider(4, 40, value=DEFAULT_STEPS, step=1, label="steps")
                    cfg = gr.Slider(1.0, 5.0, value=DEFAULT_CFG, step=0.1, label="CFG")
                    with gr.Row():
                        sampler = gr.Dropdown(SAMPLERS, value=DEFAULT_SAMPLER, label="sampler")
                        scheduler = gr.Dropdown(SCHEDULERS, value=DEFAULT_SCHEDULER, label="scheduler")

                with gr.Row():
                    seed = gr.Number(value=DEFAULT_SEED, precision=0, label="seed")
                    randomize = gr.Checkbox(value=False, label="randomize seed")

                gen_budget = gr.Slider(0, MAX_GPU_SECONDS, value=0, step=10, label="GPU budget (0 = automatic)")
                button = gr.Button("generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                gallery = gr.Gallery(label="output", columns=2, height=600)
                status = gr.Textbox(label="status", interactive=False)
                used_seed = gr.Number(label="used seed", interactive=False)

        def on_mode_change(value: str):
            editing = value == "edit"
            return (
                gr.update(visible=editing),
                gr.update(visible=not editing),
                gr.update(visible=editing),
                gr.update(visible=editing),
            )

        mode.change(on_mode_change, inputs=[mode], outputs=[image_inputs, t2i_resolution, edit_controls, edit_prompt])

        def parse_lora_state(value: str) -> dict[str, float]:
            try:
                data = json.loads(value or "{}")
                if not isinstance(data, dict):
                    return {}
                result = {}
                for name, strength in data.items():
                    try:
                        number = float(strength)
                    except Exception:
                        continue
                    if abs(number) > 1e-6:
                        result[str(name)] = number
                return result
            except Exception:
                return {}

        def generate_wrapper(*values):
            *base, lora_json = values
            return generate(*base, lora_weights=parse_lora_state(lora_json))

        generation_inputs = [
            mode, prompt, negative_prompt, edit_prompt, primary, second,
            width, height, target_mp, grounding, ref_boost, ref_boost_a,
            steps, cfg, sampler, scheduler, seed, randomize, gen_budget,
            lora_state,
        ]
        button.click(generate_wrapper, inputs=generation_inputs, outputs=[gallery, status, used_seed], js=LORA_JS)

    return demo


def _on_startup() -> None:
    if os.environ.get("KREA_SKIP_STARTUP") == "1":
        return
    try:
        _ensure_comfy()
        _ensure_models()
        _init_comfy_nodes()
        print(f"[startup] LoRAs found: {len(_discover_loras())}", flush=True)
    except Exception as exc:
        print(f"[startup] setup incomplete ({type(exc).__name__}: {exc}); generation will retry", flush=True)


_on_startup()
demo = create_ui()
demo.queue()

if __name__ == "__main__":
    demo.launch()
