import os, re, time, uuid, torch
import numpy as np
from PIL import Image
import gradio as gr
from nodes import NODE_CLASS_MAPPINGS

# ============================================================
# KREA-2 EDIT
# Reproduces the supplied Krea2Edit ComfyUI workflow:
#
# source -> 1.4MP scale -> VAEEncode -> Krea2EditModelPatch
# source -> 1.4MP scale ----------------> GroundedEncode
# UNET -> LoRAs -> AuraFlow shift 4 -> Krea2Edit patch
# GroundedEncode -> KSampler positive
# ConditioningZeroOut -> KSampler negative
# EmptySD3LatentImage -> KSampler latent
# KSampler -> VAE Decode
#
# The workflow's second reference is supported as source_b.
# ============================================================

ROOT = "/root/ComfyUI"
LORA_DIR = f"{ROOT}/models/loras"
OUT_DIR = f"{ROOT}/output"
os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

UNET_NAME = "krea2_turbo_fp8_scaled.safetensors"
CLIP_NAME = "qwen3vl_4b_fp8_scaled.safetensors"
VAE_NAME = "qwen_image_vae.safetensors"

DEFAULT_ID_LORA = "krea/krea2_identity_edit_v1_2.safetensors"

# Workflow values
MEGAPIXELS = 1.4
GROUNDING_PX = 768
REF_BOOST = 1.0
REF_BOOST_A = 1.0
AURA_SHIFT = 4.0
STEPS = 10
CFG = 1.0
DENOISE = 1.0

def node(name):
    if name not in NODE_CLASS_MAPPINGS:
        raise RuntimeError(
            f'Missing ComfyUI node: "{name}". '
            "Update ComfyUI and restart the notebook."
        )
    return NODE_CLASS_MAPPINGS[name]()

UNETLoader = node("UNETLoader")
CLIPLoader = node("CLIPLoader")
VAELoader = node("VAELoader")
VAEEncode = node("VAEEncode")
VAEDecode = node("VAEDecode")
KSampler = node("KSampler")
ImageScale = node("ImageScaleToTotalPixels")
Aura = node("ModelSamplingAuraFlow")
ZeroOut = node("ConditioningZeroOut")
Patch = node("Krea2EditModelPatch")
Grounded = node("Krea2EditGroundedEncode")

if "LoraLoaderModelOnly" in NODE_CLASS_MAPPINGS:
    LoraModelOnly = node("LoraLoaderModelOnly")
else:
    LoraModelOnly = None

print("=" * 65)
print("                 KREA-2 EDIT")
print("=" * 65)

with torch.inference_mode():
    print("[1/3] Loading Krea-2 UNet...")
    BASE_MODEL = UNETLoader.load_unet(UNET_NAME, "default")[0]

    print("[2/3] Loading Qwen3-VL...")
    BASE_CLIP = CLIPLoader.load_clip(
        CLIP_NAME, "krea2", "default"
    )[0]

    print("[3/3] Loading VAE...")
    BASE_VAE = VAELoader.load_vae(VAE_NAME)[0]

print("Base models loaded.")

def lora_list():
    out = [""]
    for root, _, files in os.walk(LORA_DIR):
        for f in files:
            if f.lower().endswith(
                (".safetensors", ".ckpt", ".pt")
            ):
                out.append(
                    os.path.relpath(
                        os.path.join(root, f), LORA_DIR
                    ).replace("\\", "/")
                )
    return [""] + sorted(set(out[1:]))

LORAS = lora_list()
print(f"LoRAs found: {len(LORAS)-1}")

def load_image(path):
    if not path:
        raise ValueError("Input image is required.")
    if isinstance(path, Image.Image):
        im = path.convert("RGB")
    else:
        im = Image.open(path).convert("RGB")
    return torch.from_numpy(
        np.asarray(im).astype(np.float32) / 255.0
    )[None, ...]

def scale_1_4mp(image):
    # Same purpose/settings as workflow ImageScaleToTotalPixels:
    # lanczos, 1.4 MP, 64-pixel rounding.
    h, w = image.shape[1], image.shape[2]
    target = MEGAPIXELS * 1000000
    s = (target / (w * h)) ** 0.5
    nw = max(64, int(round(w * s / 64)) * 64)
    nh = max(64, int(round(h * s / 64)) * 64)

    im = Image.fromarray(
        (image[0].cpu().numpy() * 255).clip(0,255).astype(np.uint8)
    )
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)

    return torch.from_numpy(
        np.asarray(im).astype(np.float32) / 255.0
    )[None, ...]

def empty_sd3_latent(width, height):
    # Exact EmptySD3LatentImage behavior:
    # 16 channels, /8 spatial size, initialized to 0.0609.
    return {
        "samples": torch.ones(
            [1, 16, height // 8, width // 8],
            device=BASE_MODEL.model_management.intermediate_device()
            if hasattr(BASE_MODEL, "model_management")
            else "cpu"
        ) * 0.0609
    }

def make_latent(width, height):
    # Prefer the actual node when this ComfyUI exposes it.
    if "EmptySD3LatentImage" in NODE_CLASS_MAPPINGS:
        return node("EmptySD3LatentImage").generate(
            width, height, 1
        )[0]

    # Compatible fallback for builds where the node mapping is absent.
    import comfy.model_management
    return {
        "samples": torch.ones(
            [1, 16, height // 8, width // 8],
            device=comfy.model_management.intermediate_device()
        ) * 0.0609
    }

def apply_loras(model, names, strengths):
    if not any(names):
        return model, []

    if LoraModelOnly is None:
        raise RuntimeError(
            "LoraLoaderModelOnly is unavailable in this ComfyUI build."
        )

    applied = []

    for i, (name, strength) in enumerate(
        zip(names, strengths), 1
    ):
        if not name:
            continue

        strength = float(strength)
        if strength == 0:
            continue

        print(
            f"  LoRA {i}: {name} @ {strength}"
        )

        model = LoraModelOnly.load_lora_model_only(
            model,
            name.replace("\\", "/"),
            strength
        )[0]

        applied.append((name, strength))

    return model, applied

@torch.inference_mode()
def generate(
    source_path,
    source_b_path,
    prompt,
    grounding_px,
    ref_boost,
    ref_boost_a,
    steps,
    cfg,
    seed,
    lora_names,
    lora_strengths
):
    start = time.time()

    if not source_path:
        raise gr.Error("Upload an input image first.")

    seed = int(seed)
    if seed == 0:
        seed = int(
            torch.randint(
                0, 1125899906842624, (1,)
            ).item()
        )

    print("\n" + "=" * 65)
    print("NEW KREA-2 EDIT")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load images
    # --------------------------------------------------------
    source = load_image(source_path)
    source_b = load_image(source_b_path) if source_b_path else None

    # --------------------------------------------------------
    # 2. Exact workflow preprocessing: 1.4 MP
    # --------------------------------------------------------
    source_scaled = scale_1_4mp(source)
    source_b_scaled = (
        scale_1_4mp(source_b)
        if source_b is not None else None
    )

    height = source_scaled.shape[1]
    width = source_scaled.shape[2]

    print(f"Output grid: {width} x {height}")

    # --------------------------------------------------------
    # 3. Apply LoRAs BEFORE Krea2EditModelPatch
    # --------------------------------------------------------
    model, applied = apply_loras(
        BASE_MODEL,
        lora_names,
        lora_strengths
    )

    # --------------------------------------------------------
    # 4. VAEEncode source(s)
    # --------------------------------------------------------
    source_latent = VAEEncode.encode(
        BASE_VAE,
        source_scaled
    )[0]

    source_latent_b = None
    if source_b_scaled is not None:
        source_latent_b = VAEEncode.encode(
            BASE_VAE,
            source_b_scaled
        )[0]

    # --------------------------------------------------------
    # 5. Exact EmptySD3LatentImage target
    # --------------------------------------------------------
    target_latent = make_latent(width, height)

    # --------------------------------------------------------
    # 6. Krea2EditModelPatch
    # Exact workflow:
    # ref_boost=1, ref_boost_a=1, fit_mode=fit,
    # VAE + source_image + target_latent connected.
    # --------------------------------------------------------
    model = Patch.patch(
        model=model,
        source_latent=source_latent,
        source_latent_b=source_latent_b,
        ref_boost=float(ref_boost),
        ref_boost_a=float(ref_boost_a),
        ref_boost_mask=None,
        vae=BASE_VAE,
        source_image=source_scaled,
        source_image_b=source_b_scaled,
        fit_mode="fit",
        target_latent=target_latent
    )[0]

    # --------------------------------------------------------
    # 7. ModelSamplingAuraFlow shift=4
    # --------------------------------------------------------
    model = Aura.patch_aura(
        model,
        AURA_SHIFT
    )[0]

    # --------------------------------------------------------
    # 8. Krea2EditGroundedEncode
    # --------------------------------------------------------
    if not prompt:
        prompt = "restore the image quality to 4k HDR."

    positive = Grounded.encode(
        clip=BASE_CLIP,
        image=source_scaled,
        image_b=source_b_scaled,
        prompt=prompt,
        grounding_px=int(grounding_px)
    )[0]

    # Exact supplied workflow uses ConditioningZeroOut.
    negative = ZeroOut.zero_out(positive)[0]

    # --------------------------------------------------------
    # 9. KSampler
    # Exact workflow: euler / beta / denoise 1 / CFG 1.
    # --------------------------------------------------------
    samples = KSampler.sample(
        model,
        seed,
        int(steps),
        float(cfg),
        "euler",
        "beta",
        positive,
        negative,
        target_latent,
        float(DENOISE)
    )[0]

    # --------------------------------------------------------
    # 10. VAE Decode
    # --------------------------------------------------------
    decoded = VAEDecode.decode(
        BASE_VAE,
        samples
    )[0]

    arr = (
        decoded[0]
        .detach()
        .cpu()
        .clamp(0, 1)
        .numpy()
    )

    image = Image.fromarray(
        (arr * 255).astype(np.uint8),
        "RGB"
    )

    safe = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        prompt
    )[:50]

    path = os.path.join(
        OUT_DIR,
        f"{safe}_{seed}_{uuid.uuid4().hex[:6]}.png"
    )

    image.save(path)

    print(f"Saved: {path}")
    print(f"Seed: {seed}")
    print(f"Time: {time.time()-start:.1f}s")
    print("=" * 65)

    return path, seed

def run(
    source, source_b, prompt,
    grounding, ref_boost, ref_boost_a,
    steps, cfg, seed,
    *lora_values
):
    names = list(lora_values[0::2])
    strengths = list(lora_values[1::2])

    selected = [x for x in names if x]
    if len(selected) != len(set(selected)):
        raise gr.Error(
            "Each LoRA can only be selected once."
        )

    path, used_seed = generate(
        source,
        source_b,
        prompt,
        grounding,
        ref_boost,
        ref_boost_a,
        steps,
        cfg,
        seed,
        names,
        strengths
    )

    return path, path, str(used_seed)

# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="Krea-2 Edit",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("# Krea-2 Image Editor")

    with gr.Row():

        with gr.Column():

            source = gr.Image(
                label="Input Image",
                type="filepath",
                height=430
            )

            source_b = gr.Image(
                label="Second Reference (optional)",
                type="filepath",
                height=280
            )

            prompt = gr.Textbox(
                value="restore the image quality to 4k HDR.",
                label="Edit Instruction",
                lines=5
            )

            with gr.Row():

                grounding = gr.Slider(
                    384, 1536,
                    value=GROUNDING_PX,
                    step=64,
                    label="Grounding Resolution"
                )

                ref_boost = gr.Slider(
                    0, 10,
                    value=REF_BOOST,
                    step=0.1,
                    label="Reference Fidelity"
                )

                ref_boost_a = gr.Slider(
                    0, 10,
                    value=REF_BOOST_A,
                    step=0.1,
                    label="Second Reference Fidelity"
                )

            with gr.Row():

                steps = gr.Slider(
                    1, 30,
                    value=STEPS,
                    step=1,
                    label="Steps"
                )

                cfg = gr.Slider(
                    0.5, 5,
                    value=CFG,
                    step=0.1,
                    label="CFG"
                )

                seed = gr.Number(
                    value=0,
                    precision=0,
                    label="Seed (0 = random)"
                )

            with gr.Accordion(
                "10 LoRA Slots",
                open=True
            ):

                lora_inputs = []

                for i in range(10):

                    with gr.Row():

                        d = gr.Dropdown(
                            choices=LORAS,
                            value=(
                                DEFAULT_ID_LORA
                                if i == 0 and
                                DEFAULT_ID_LORA in LORAS
                                else ""
                            ),
                            label=f"LoRA {i+1}",
                            scale=4
                        )

                        s = gr.Number(
                            value=1.0 if i == 0 else 0.0,
                            label="Strength",
                            precision=2,
                            scale=1
                        )

                        lora_inputs += [d, s]

            button = gr.Button(
                "🚀 Edit Image",
                variant="primary",
                size="lg"
            )

        with gr.Column():

            output = gr.Image(
                label="Edited Image",
                height=680
            )

            file = gr.File(
                label="Output File"
            )

            used_seed = gr.Textbox(
                label="Seed Used",
                interactive=False
            )

    button.click(
        fn=run,
        inputs=[
            source,
            source_b,
            prompt,
            grounding,
            ref_boost,
            ref_boost_a,
            steps,
            cfg,
            seed,
            *lora_inputs
        ],
        outputs=[
            output,
            file,
            used_seed
        ]
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True
    )
