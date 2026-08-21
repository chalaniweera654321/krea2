import os
import time
import shutil
import re
import uuid

import torch
import numpy as np
from PIL import Image

import sys
import os
import importlib.util

COMFYUI_PATH = "/root/ComfyUI"

if COMFYUI_PATH not in sys.path:
    sys.path.insert(0, COMFYUI_PATH)

from nodes import NODE_CLASS_MAPPINGS

# ============================================================
# Load Krea2Edit directly from its __init__.py
# ============================================================

KREA2EDIT_DIR = (
    "/root/ComfyUI/custom_nodes/comfyui-krea2edit"
)

KREA2EDIT_INIT = os.path.join(
    KREA2EDIT_DIR,
    "__init__.py"
)

spec = importlib.util.spec_from_file_location(
    "krea2edit",
    KREA2EDIT_INIT,
    submodule_search_locations=[KREA2EDIT_DIR]
)

krea2edit = importlib.util.module_from_spec(spec)

# Make the custom node directory importable
if KREA2EDIT_DIR not in sys.path:
    sys.path.insert(0, KREA2EDIT_DIR)

spec.loader.exec_module(krea2edit)

# Register Krea2Edit nodes
NODE_CLASS_MAPPINGS.update(
    krea2edit.NODE_CLASS_MAPPINGS
)

print(
    "[Krea2Edit] Loaded:",
    list(krea2edit.NODE_CLASS_MAPPINGS.keys())
)

# Verify
if "Krea2EditModelPatch" not in NODE_CLASS_MAPPINGS:
    raise RuntimeError(
        "Krea2EditModelPatch failed to register."
    )

if "Krea2EditGroundedEncode" not in NODE_CLASS_MAPPINGS:
    raise RuntimeError(
        "Krea2EditGroundedEncode failed to register."
    )


# ============================================================
# KREA-2 EDIT + 10 LoRA
# Based on the supplied Krea-2 edit workflow.
#
# Required custom node:
#   https://github.com/lbouaraba/comfyui-krea2edit
#
# The edit path mirrors the workflow:
#   source image
#      -> VAE encode --------------------+
#      -> grounded Qwen3-VL conditioning |
#                                        v
#   UNET -> LoRAs -> Krea2EditModelPatch -> KSampler -> VAE decode
#
# Krea2Edit uses the source image twice:
#   1) VAE latent = appearance/identity path
#   2) Qwen3-VL grounded encode = semantic/image understanding path
# ============================================================

print("\n" + "=" * 60)
print("              KREA-2 EDIT + 10 LoRA")
print("=" * 60)


# ============================================================
# COMFYUI NODES
# ============================================================

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()

Krea2EditModelPatch = NODE_CLASS_MAPPINGS["Krea2EditModelPatch"]()
Krea2EditGroundedEncode = NODE_CLASS_MAPPINGS["Krea2EditGroundedEncode"]()


# ============================================================
# MODEL SETTINGS
# ============================================================

BASE_MODEL = "krea2_turbo_fp8_scaled.safetensors"
CLIP_MODEL = "qwen3vl_4b_fp8_scaled.safetensors"
VAE_MODEL = "qwen_image_vae.safetensors"

LORA_DIR = "./models/loras"

# The Krea 2 Identity Edit LoRA is required for true identity/edit
# behavior. It can also be selected as one of the normal LoRA slots.
DEFAULT_EDIT_LORA = "krea\\krea2_identity_edit_v1_2.safetensors"


# ============================================================
# LOAD BASE MODELS ONCE
# ============================================================

startup_start = time.time()

with torch.inference_mode():

    print("\n[1/3] Loading Krea-2 UNet... ", end="", flush=True)
    t0 = time.time()

    base_model = UNETLoader.load_unet(
        BASE_MODEL,
        "default"
    )[0]

    print(f"done ({time.time() - t0:.1f}s)")

    print("[2/3] Loading Qwen3-VL CLIP... ", end="", flush=True)
    t0 = time.time()

    base_clip = CLIPLoader.load_clip(
        CLIP_MODEL,
        type="krea2"
    )[0]

    print(f"done ({time.time() - t0:.1f}s)")

    print("[3/3] Loading Qwen Image VAE... ", end="", flush=True)
    t0 = time.time()

    vae = VAELoader.load_vae(
        VAE_MODEL
    )[0]

    print(f"done ({time.time() - t0:.1f}s)")


print(
    f"\nBase models loaded in "
    f"{time.time() - startup_start:.1f}s"
)
print("=" * 60)


# ============================================================
# LORA DIRECTORY
# ============================================================

def get_lora_files():
    if not os.path.exists(LORA_DIR):
        print(
            f"\nLoRA directory not found:"
            f"\n{os.path.abspath(LORA_DIR)}"
        )
        return [""]

    files = []

    for root, dirs, filenames in os.walk(LORA_DIR):
        for filename in filenames:
            if filename.lower().endswith(
                (".safetensors", ".pt", ".ckpt")
            ):
                relative_path = os.path.relpath(
                    os.path.join(root, filename),
                    LORA_DIR
                )
                files.append(relative_path)

    files.sort()

    if not files:
        print(
            "\nNo LoRAs found in:"
            f"\n{os.path.abspath(LORA_DIR)}"
        )
        return [""]

    print(f"\nFound {len(files)} LoRA(s)")
    for f in files:
        print(f"   • {f}")

    return files


LORA_FILES = get_lora_files()


# ============================================================
# IMAGE CONVERSION
# ============================================================

def image_to_tensor(image_path):
    """
    Convert a Gradio/PIL image path to ComfyUI IMAGE:
        B,H,W,C float32 in [0,1]
    """
    if not image_path:
        raise ValueError("Please upload an input image.")

    if isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None, ...]
    return tensor


def resize_for_target(image, width, height):
    """
    Keep the source image intact for the grounded encoder.
    Krea2Edit's pixel path performs the training-matched fit when
    source_image + VAE + target_latent are supplied to the patch node.
    """
    return image


# ============================================================
# APPLY MULTIPLE LORAS
# ============================================================

def apply_loras(lora_names, lora_strengths):
    model = base_model
    clip = base_clip
    applied = []

    for i in range(len(lora_names)):
        lora_name = lora_names[i]

        if not lora_name:
            continue

        try:
            strength = float(lora_strengths[i])
        except Exception:
            strength = 1.0

        if strength == 0:
            continue

        print(f"\n   [{i + 1}] Applying LoRA:")
        print(f"       {lora_name}")
        print(f"       Strength: {strength}")

        t0 = time.time()

        model, clip = LoraLoader.load_lora(
            model,
            clip,
            lora_name,
            strength,
            strength
        )

        applied.append(lora_name)
        print(f"       done ({time.time() - t0:.1f}s)")

    return model, clip, applied


# ============================================================
# SAVE HELPERS
# ============================================================

save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)


def get_save_path(prompt):
    safe_prompt = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        prompt or "krea2_edit"
    )[:45]

    uid = uuid.uuid4().hex[:6]

    return os.path.join(
        save_dir,
        f"{safe_prompt}_{uid}.png"
    )


# ============================================================
# GENERATION / EDIT
# ============================================================

@torch.inference_mode()
def generate(input_data):

    values = input_data["input"]

    source_image_path = values["source_image"]
    source_image_b_path = values.get("source_image_b")

    edit_prompt = values["positive_prompt"]
    negative_prompt = values["negative_prompt"]

    width = int(values["width"])
    height = int(values["height"])
    seed = int(values["seed"])
    steps = int(values["steps"])
    cfg = float(values["cfg"])
    sampler_name = values["sampler_name"]
    scheduler = values["scheduler"]

    grounding_px = int(values["grounding_px"])
    ref_boost = float(values["ref_boost"])
    ref_boost_a = float(values["ref_boost_a"])
    fit_mode = values["fit_mode"]

    lora_names = values["lora_names"]
    lora_strengths = values["lora_strengths"]

    print("\n" + "=" * 60)
    print("                  NEW KREA-2 EDIT")
    print("=" * 60)

    total_start = time.time()

    # --------------------------------------------------------
    # SOURCE IMAGE
    # --------------------------------------------------------

    print("\n[1/7] Loading source image...")

    source_image = image_to_tensor(source_image_path)

    source_image_b = None
    if source_image_b_path:
        source_image_b = image_to_tensor(source_image_b_path)

    print(
        f"      Source: "
        f"{source_image.shape[2]}x{source_image.shape[1]}"
    )

    if source_image_b is not None:
        print(
            f"      Reference B: "
            f"{source_image_b.shape[2]}x{source_image_b.shape[1]}"
        )

    # --------------------------------------------------------
    # LORAS
    # --------------------------------------------------------

    print("\n[2/7] Applying LoRAs...")

    model, clip, applied_loras = apply_loras(
        lora_names,
        lora_strengths
    )

    print(
        f"\n      Applied {len(applied_loras)} LoRA(s)"
    )

    # --------------------------------------------------------
    # TARGET LATENT / OUTPUT SIZE
    # --------------------------------------------------------

    print("\n[3/7] Creating target latent...")

    target_latent = EmptyLatentImage.generate(
        width,
        height,
        batch_size=1
    )[0]

    # --------------------------------------------------------
    # SOURCE LATENTS
    # --------------------------------------------------------

    print("\n[4/7] Encoding source image...")

    source_latent = VAEEncode.encode(
        vae,
        source_image
    )[0]

    source_latent_b = None

    if source_image_b is not None:
        source_latent_b = VAEEncode.encode(
            vae,
            source_image_b
        )[0]

    # --------------------------------------------------------
    # KREA EDIT MODEL PATCH
    #
    # This is the important difference from the old app.
    # The normal text-to-image KSampler is NOT used directly.
    # The model is patched with Krea2EditModelPatch so the
    # source image becomes in-context appearance tokens.
    # --------------------------------------------------------

    print("\n[5/7] Patching Krea-2 for image editing...")

    patched_model = Krea2EditModelPatch.patch(
        model,
        source_latent,
        source_latent_b=source_latent_b,
        ref_boost=ref_boost,
        ref_boost_a=ref_boost_a,
        ref_boost_mask=None,
        vae=vae,
        source_image=source_image,
        source_image_b=source_image_b,
        fit_mode=fit_mode,
        target_latent=target_latent
    )[0]

    # --------------------------------------------------------
    # GROUNDED POSITIVE CONDITIONING
    #
    # Qwen3-VL sees the source image + edit instruction.
    # --------------------------------------------------------

    print("\n[6/7] Encoding grounded edit prompt...")

    positive = Krea2EditGroundedEncode.encode(
        clip,
        edit_prompt,
        image=source_image,
        image_b=source_image_b,
        grounding_px=grounding_px,
        system_prompt=""
    )[0]

    # CFG=1 is the recommended Turbo path. If the user chooses
    # a higher CFG, ground the negative conditioning on the same
    # source image as recommended by the Krea2Edit workflow.
    negative = Krea2EditGroundedEncode.encode(
        clip,
        negative_prompt or "",
        image=source_image,
        image_b=source_image_b,
        grounding_px=grounding_px,
        system_prompt=""
    )[0]

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    print(
        f"\n      Sampling: {steps} steps, "
        f"CFG {cfg}, {sampler_name}/{scheduler}"
    )

    samples = KSampler.sample(
        patched_model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        target_latent,
        denoise=1.0
    )[0]

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    print("\n[7/7] Decoding output...")

    decoded = VAEDecode.decode(
        vae,
        samples
    )[0].detach().cpu()

    image_array = np.clip(
        decoded[0].numpy() * 255.0,
        0,
        255
    ).astype(np.uint8)

    output = Image.fromarray(image_array)

    save_path = get_save_path(edit_prompt)
    output.save(save_path)

    # Optional Google Drive copy used by the original app.
    drive_path = "/root/gdrive/MyDrive/krea2_turbo"

    if os.path.exists(drive_path):
        shutil.copy(save_path, drive_path)
        print(f"\nCopied to Google Drive: {drive_path}")

    print(f"\nSaved: {save_path}")

    print("\nLoRAs used:")
    if applied_loras:
        for lora in applied_loras:
            print(f"   • {lora}")
    else:
        print("   • None")

    print(f"\nSeed: {seed}")
    print(
        f"Total time: "
        f"{time.time() - total_start:.1f}s"
    )

    print("=" * 60 + "\n")

    return save_path, seed


# ============================================================
# GRADIO
# ============================================================

import gradio as gr


# ============================================================
# UI FUNCTION
# ============================================================

def generate_ui(
    source_image,
    source_image_b,

    positive_prompt,
    negative_prompt,

    width,
    height,

    seed,
    steps,

    cfg,

    grounding_px,
    ref_boost,
    ref_boost_a,
    fit_mode,

    lora1,
    lora1_strength,

    lora2,
    lora2_strength,

    lora3,
    lora3_strength,

    lora4,
    lora4_strength,

    lora5,
    lora5_strength,

    lora6,
    lora6_strength,

    lora7,
    lora7_strength,

    lora8,
    lora8_strength,

    lora9,
    lora9_strength,

    lora10,
    lora10_strength,

    sampler_name="euler",
    scheduler="simple"
):

    if not source_image:
        raise gr.Error("Please upload an input image.")

    # Random seed when 0 is selected.
    if int(seed) == 0:
        seed = int(torch.randint(
            0,
            1125899906842624,
            (1,)
        ).item())

    lora_names = [
        lora1, lora2, lora3, lora4, lora5,
        lora6, lora7, lora8, lora9, lora10
    ]

    lora_strengths = [
        lora1_strength, lora2_strength,
        lora3_strength, lora4_strength,
        lora5_strength, lora6_strength,
        lora7_strength, lora8_strength,
        lora9_strength, lora10_strength
    ]

    input_data = {
        "input": {
            "source_image": source_image,
            "source_image_b": source_image_b,

            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,

            "width": int(width),
            "height": int(height),

            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),

            "sampler_name": sampler_name,
            "scheduler": scheduler,

            "grounding_px": int(grounding_px),
            "ref_boost": float(ref_boost),
            "ref_boost_a": float(ref_boost_a),
            "fit_mode": fit_mode,

            "lora_names": lora_names,
            "lora_strengths": lora_strengths
        }
    }

    image_path, used_seed = generate(input_data)

    return image_path, image_path, str(used_seed)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_EDIT = """
Edit the input image according to this instruction while preserving
the subject's identity, pose, composition, lighting and realistic
appearance unless the instruction explicitly changes them.
"""

DEFAULT_NEGATIVE = ""

custom_css = """
.gradio-container {
    font-family:
        'SF Pro Display',
        -apple-system,
        BlinkMacSystemFont,
        sans-serif;
}
"""


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks() as demo:

    gr.HTML("""
    <div style="
        width:100%;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        margin:20px 0;
    ">
        <h1 style="font-size:2.5em;margin-bottom:10px;">
            Krea-2 Image Edit + 10 LoRA
        </h1>
        <div>
            Identity-preserving image editing using Krea2Edit
        </div>
    </div>
    """)

    with gr.Row():

        # ====================================================
        # LEFT
        # ====================================================

        with gr.Column():

            # ------------------------------------------------
            # SOURCE IMAGE
            # ------------------------------------------------

            source_image = gr.Image(
                label="Input / Source Image",
                type="filepath",
                height=420
            )

            source_image_b = gr.Image(
                label="Second Reference Image (optional)",
                type="filepath",
                height=260
            )

            # ------------------------------------------------
            # EDIT PROMPT
            # ------------------------------------------------

            positive = gr.Textbox(
                DEFAULT_EDIT,
                label="Edit Instruction",
                lines=6
            )

            # ------------------------------------------------
            # NEGATIVE
            # ------------------------------------------------

            negative = gr.Textbox(
                DEFAULT_NEGATIVE,
                label="Negative Prompt (optional)",
                lines=4
            )

            # ------------------------------------------------
            # IMAGE SETTINGS
            # ------------------------------------------------

            with gr.Row():

                width = gr.Number(
                    value=1024,
                    label="Output Width",
                    precision=0
                )

                height = gr.Number(
                    value=1024,
                    label="Output Height",
                    precision=0
                )

                seed = gr.Number(
                    value=0,
                    label="Seed (0 = random)",
                    precision=0
                )

            # ------------------------------------------------
            # STEPS / CFG
            # ------------------------------------------------

            with gr.Row():

                steps = gr.Slider(
                    4,
                    25,
                    value=8,
                    step=1,
                    label="Steps"
                )

                cfg = gr.Slider(
                    0.5,
                    4.0,
                    value=1.0,
                    step=0.1,
                    label="CFG"
                )

            # ------------------------------------------------
            # KREA EDIT SETTINGS
            # ------------------------------------------------

            with gr.Accordion(
                "🧠 Krea-2 Edit Settings",
                open=True
            ):

                with gr.Row():

                    grounding_px = gr.Slider(
                        384,
                        1536,
                        value=768,
                        step=64,
                        label="Grounding Resolution"
                    )

                    ref_boost = gr.Slider(
                        0.25,
                        8.0,
                        value=1.0,
                        step=0.25,
                        label="Reference Fidelity"
                    )

                with gr.Row():

                    ref_boost_a = gr.Slider(
                        0.25,
                        8.0,
                        value=1.0,
                        step=0.25,
                        label="Reference B Fidelity"
                    )

                    fit_mode = gr.Dropdown(
                        choices=["fit", "crop"],
                        value="fit",
                        label="Reference Fit Mode"
                    )

                gr.Markdown(
                    """
                    **Turbo recommendation:** 8 steps, CFG 1.

                    **Grounding:** 768 is a good default. Around 1024 can
                    help people/identity likeness.

                    **Reference fidelity:** 1.0 is neutral. Higher values
                    pull harder toward the source appearance.

                    **Fit mode:** `fit` is the recommended v1.2 mode.
                    """
                )

            # ------------------------------------------------
            # LORA SETTINGS
            # ------------------------------------------------

            with gr.Accordion(
                "🎨 LoRA Settings",
                open=True
            ):

                gr.Markdown(
                    """
                    Select up to **10 LoRAs**. Leave unused slots empty.
                    The Krea-2 identity-edit LoRA should normally be enabled.
                    """
                )

                with gr.Row():
                    lora1 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value=(
                            DEFAULT_EDIT_LORA
                            if DEFAULT_EDIT_LORA in LORA_FILES
                            else ""
                        ),
                        label="LoRA 1",
                        scale=4
                    )
                    lora1_strength = gr.Textbox(
                        value="1",
                        label="Strength",
                        scale=1
                    )

                with gr.Row():
                    lora2 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA 2",
                        scale=4
                    )
                    lora2_strength = gr.Textbox(
                        value="1",
                        label="Strength",
                        scale=1
                    )

                with gr.Row():
                    lora3 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA 3",
                        scale=4
                    )
                    lora3_strength = gr.Textbox(
                        value="1",
                        label="Strength",
                        scale=1
                    )

                with gr.Row():
                    lora4 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA 4",
                        scale=4
                    )
                    lora4_strength = gr.Textbox(
                        value="1",
                        label="Strength",
                        scale=1
                    )

                with gr.Row():
                    lora5 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA 5",
                        scale=4
                    )
                    lora5_strength = gr.Textbox(
                        value="1",
                        label="Strength",
                        scale=1
                    )

                with gr.Row():
                    lora6 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA 6",
                        scale=4
                    )
                    lora6_strength = gr.Textbox(
                        value="1",
                        label="Strength",
                        scale=1
                    )

                with gr.Row():
                    lora7 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA 7",
                        scale=4
                    )
                    lora7_strength = gr.Textbox(
                        value="1",
                        label="Strength",
                        scale=1
                    )

                with gr.Row():
                    lora8 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA 8",
                        scale=4
                    )
                    lora8_strength = gr.Textbox(
                        value="1",
                        label="Strength",
                        scale=1
                    )

                with gr.Row():
                    lora9 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA 9",
                        scale=4
                    )
                    lora9_strength = gr.Textbox(
                        value="1",
                        label="Strength",
                        scale=1
                    )

                with gr.Row():
                    lora10 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA 10",
                        scale=4
                    )
                    lora10_strength = gr.Textbox(
                        value="1",
                        label="Strength",
                        scale=1
                    )

            # ------------------------------------------------
            # GENERATE
            # ------------------------------------------------

            run = gr.Button(
                "🚀 Edit Image",
                variant="primary",
                size="lg"
            )

        # ====================================================
        # RIGHT
        # ====================================================

        with gr.Column():

            output_img = gr.Image(
                label="Edited Image",
                height=650
            )

            download_image = gr.File(
                label="Download Image"
            )

            used_seed = gr.Textbox(
                label="Seed Used",
                interactive=False
            )


    # ========================================================
    # LORA DUPLICATE PREVENTION
    # ========================================================

    all_lora_dropdowns = [
        lora1, lora2, lora3, lora4, lora5,
        lora6, lora7, lora8, lora9, lora10
    ]


    def update_lora_choices(*selected):

        updated = []

        for current in selected:

            choices = [""] + [
                x for x in LORA_FILES
                if x == current or x not in selected
            ]

            updated.append(
                gr.update(choices=choices)
            )

        return updated


    for dropdown in all_lora_dropdowns:

        dropdown.change(
            fn=update_lora_choices,
            inputs=all_lora_dropdowns,
            outputs=all_lora_dropdowns
        )


    # ========================================================
    # RUN
    # ========================================================

    run.click(
        fn=generate_ui,

        inputs=[
            source_image,
            source_image_b,

            positive,
            negative,

            width,
            height,

            seed,
            steps,

            cfg,

            grounding_px,
            ref_boost,
            ref_boost_a,
            fit_mode,

            lora1,
            lora1_strength,

            lora2,
            lora2_strength,

            lora3,
            lora3_strength,

            lora4,
            lora4_strength,

            lora5,
            lora5_strength,

            lora6,
            lora6_strength,

            lora7,
            lora7_strength,

            lora8,
            lora8_strength,

            lora9,
            lora9_strength,

            lora10,
            lora10_strength
        ],

        outputs=[
            output_img,
            download_image,
            used_seed
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

demo.launch(
    share=True,
    debug=True,
    theme=gr.themes.Soft(),
    css=custom_css
)
