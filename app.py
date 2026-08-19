import os
import random
import time
import shutil
import re
import uuid

import torch
import numpy as np
from PIL import Image

from nodes import NODE_CLASS_MAPPINGS


# ============================================================
# KREA-2 TURBO + CHECKPOINT + MULTIPLE LORA
# ============================================================

print("\n" + "=" * 70)
print("        KREA-2 TURBO + CHECKPOINT + MULTIPLE LORA")
print("=" * 70)


# ============================================================
# COMFYUI NODES
# ============================================================

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()

# Checkpoint loader
CheckpointLoaderSimple = NODE_CLASS_MAPPINGS[
    "CheckpointLoaderSimple"
]()

# Krea-2 negative conditioning
ConditioningZeroOut = NODE_CLASS_MAPPINGS.get(
    "ConditioningZeroOut",
    None
)()


# ============================================================
# DIRECTORIES
# ============================================================

LORA_DIR = "./models/loras"
CHECKPOINT_DIR = "./models/checkpoints"
SAVE_DIR = "./results"

os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# MODEL NAMES
# ============================================================

ORIGINAL_MODEL_NAME = "Original Krea-2 Turbo"

ORIGINAL_UNET = "krea2_turbo_fp8_scaled.safetensors"
ORIGINAL_CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
ORIGINAL_VAE = "qwen_image_vae.safetensors"


# ============================================================
# FIND CHECKPOINTS
# ============================================================

def get_checkpoint_files():

    files = []

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for root, dirs, filenames in os.walk(CHECKPOINT_DIR):

        for filename in filenames:

            if filename.lower().endswith(
                (
                    ".safetensors",
                    ".ckpt",
                    ".pt"
                )
            ):

                full_path = os.path.join(
                    root,
                    filename
                )

                relative_path = os.path.relpath(
                    full_path,
                    CHECKPOINT_DIR
                )

                files.append(relative_path)

    files.sort()

    return files


CHECKPOINT_FILES = get_checkpoint_files()

print(
    f"\n📦 Found {len(CHECKPOINT_FILES)} checkpoint(s)"
)

for checkpoint in CHECKPOINT_FILES:
    print(f"   • {checkpoint}")


# ============================================================
# FIND LORAS
# ============================================================

def get_lora_files():

    files = []

    if not os.path.exists(LORA_DIR):
        os.makedirs(LORA_DIR, exist_ok=True)

    for root, dirs, filenames in os.walk(LORA_DIR):

        for filename in filenames:

            if filename.lower().endswith(
                (
                    ".safetensors",
                    ".pt",
                    ".ckpt"
                )
            ):

                full_path = os.path.join(
                    root,
                    filename
                )

                relative_path = os.path.relpath(
                    full_path,
                    LORA_DIR
                )

                files.append(relative_path)

    files.sort()

    if not files:

        print(
            "\n⚠️ No LoRAs found in:"
            f"\n{os.path.abspath(LORA_DIR)}"
        )

        return [""]

    print(
        f"\n✅ Found {len(files)} LoRA(s)"
    )

    for f in files:
        print(f"   • {f}")

    return files


LORA_FILES = get_lora_files()


# ============================================================
# LOAD ORIGINAL KREA-2
# ============================================================

print("\n" + "=" * 70)
print("Loading Original Krea-2 Turbo")
print("=" * 70)

startup_start = time.time()

with torch.inference_mode():

    # --------------------------------------------------------
    # UNET
    # --------------------------------------------------------

    print(
        "\n[1/3] Loading Krea-2 UNet... ",
        end="",
        flush=True
    )

    t0 = time.time()

    original_model = UNETLoader.load_unet(
        ORIGINAL_UNET,
        "default"
    )[0]

    print(
        f"done ({time.time() - t0:.1f}s)"
    )


    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    print(
        "[2/3] Loading Qwen3-VL CLIP... ",
        end="",
        flush=True
    )

    t0 = time.time()

    original_clip = CLIPLoader.load_clip(
        ORIGINAL_CLIP,
        type="krea2"
    )[0]

    print(
        f"done ({time.time() - t0:.1f}s)"
    )


    # --------------------------------------------------------
    # VAE
    # --------------------------------------------------------

    print(
        "[3/3] Loading Krea-2 VAE... ",
        end="",
        flush=True
    )

    t0 = time.time()

    original_vae = VAELoader.load_vae(
        ORIGINAL_VAE
    )[0]

    print(
        f"done ({time.time() - t0:.1f}s)"
    )


print(
    "\n✅ Original Krea-2 loaded in "
    f"{time.time() - startup_start:.1f}s"
)

print("=" * 70)


# ============================================================
# CHECKPOINT CACHE
# ============================================================

checkpoint_cache = {}


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

def load_selected_model(model_source):

    # ========================================================
    # ORIGINAL KREA-2
    # ========================================================

    if model_source == ORIGINAL_MODEL_NAME:

        print(
            "\n🎯 Using Original Krea-2 Turbo"
        )

        return (
            original_model,
            original_clip,
            original_vae,
            True,
            ORIGINAL_MODEL_NAME
        )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        model_source
    )

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{checkpoint_path}"
        )


    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if model_source in checkpoint_cache:

        print(
            f"\n🎯 Using cached checkpoint:"
            f"\n   {model_source}"
        )

        model, clip, vae = checkpoint_cache[
            model_source
        ]

        return (
            model,
            clip,
            vae,
            False,
            model_source
        )


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        "\n🎯 Loading checkpoint:"
        f"\n   {checkpoint_path}"
    )

    t0 = time.time()

    with torch.inference_mode():

        model, clip, vae = (
            CheckpointLoaderSimple.load_checkpoint(
                model_source
            )
        )


    checkpoint_cache[model_source] = (
        model,
        clip,
        vae
    )

    print(
        "✅ Checkpoint loaded in "
        f"{time.time() - t0:.1f}s"
    )

    return (
        model,
        clip,
        vae,
        False,
        model_source
    )


# ============================================================
# APPLY MULTIPLE LORAS
# ============================================================

def apply_loras(
    model,
    clip,
    lora_names,
    lora_strengths,
    clip_strengths
):

    applied = []

    for i in range(len(lora_names)):

        lora_name = lora_names[i]

        if not lora_name:
            continue


        # ----------------------------------------------------
        # Strengths
        # ----------------------------------------------------

        model_strength = float(
            lora_strengths[i]
        )

        clip_strength = float(
            clip_strengths[i]
        )


        # ----------------------------------------------------
        # Disabled
        # ----------------------------------------------------

        if (
            model_strength == 0
            and
            clip_strength == 0
        ):
            continue


        print(
            f"\n   [{i + 1}] Applying LoRA:"
        )

        print(
            f"       {lora_name}"
        )

        print(
            f"       Model strength: "
            f"{model_strength}"
        )

        print(
            f"       CLIP strength: "
            f"{clip_strength}"
        )


        t0 = time.time()


        # ----------------------------------------------------
        # Apply
        # ----------------------------------------------------

        model, clip = LoraLoader.load_lora(
            model,
            clip,
            lora_name,
            model_strength,
            clip_strength
        )


        print(
            f"       done "
            f"({time.time() - t0:.1f}s)"
        )


        applied.append(
            lora_name
        )


    return (
        model,
        clip,
        applied
    )


# ============================================================
# SAVE PATH
# ============================================================

def get_save_path(prompt):

    safe_prompt = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        prompt
    )[:25]

    if not safe_prompt:
        safe_prompt = "krea2"

    uid = uuid.uuid4().hex[:6]

    filename = (
        f"{safe_prompt}_{uid}.png"
    )

    return os.path.join(
        SAVE_DIR,
        filename
    )


# ============================================================
# GENERATION
# ============================================================

@torch.inference_mode()
def generate(input_data):

    values = input_data["input"]


    # ========================================================
    # MODEL
    # ========================================================

    model_source = values[
        "model_source"
    ]


    # ========================================================
    # PROMPTS
    # ========================================================

    positive_prompt = values[
        "positive_prompt"
    ]

    negative_prompt = values[
        "negative_prompt"
    ]


    # ========================================================
    # IMAGE SETTINGS
    # ========================================================

    seed = int(
        values["seed"]
    )

    steps = int(
        values["steps"]
    )

    cfg = float(
        values["cfg"]
    )

    sampler_name = values[
        "sampler_name"
    ]

    scheduler = values[
        "scheduler"
    ]

    denoise = float(
        values["denoise"]
    )

    width = int(
        values["width"]
    )

    height = int(
        values["height"]
    )

    batch_size = int(
        values["batch_size"]
    )


    # ========================================================
    # LORA SETTINGS
    # ========================================================

    lora_names = values[
        "lora_names"
    ]

    lora_strengths = values[
        "lora_strengths"
    ]

    clip_strengths = values[
        "clip_strengths"
    ]


    # ========================================================
    # HEADER
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "                    NEW GENERATION"
    )

    print(
        "=" * 70
    )

    total_start = time.time()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\n[1/5] Loading selected model..."
    )

    t0 = time.time()


    (
        model,
        clip,
        current_vae,
        is_original_krea,
        model_name
    ) = load_selected_model(
        model_source
    )


    print(
        f"   Model: {model_name}"
    )

    print(
        f"   Time: "
        f"{time.time() - t0:.1f}s"
    )


    # ========================================================
    # APPLY LORAS
    # ========================================================

    print(
        "\n[2/5] Applying LoRAs..."
    )

    t0 = time.time()


    (
        model,
        clip,
        applied_loras
    ) = apply_loras(
        model,
        clip,
        lora_names,
        lora_strengths,
        clip_strengths
    )


    print(
        f"\n✅ Applied "
        f"{len(applied_loras)} LoRA(s)"
    )

    print(
        f"   Time: "
        f"{time.time() - t0:.1f}s"
    )


    # ========================================================
    # PROMPTS
    # ========================================================

    print(
        "\n[3/5] Encoding prompts... ",
        end="",
        flush=True
    )

    t0 = time.time()


    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    positive = CLIPTextEncode.encode(
        clip,
        positive_prompt
    )[0]


    # --------------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------------

    # Original Krea-2 uses ZeroOut according to
    # the original application.
    #
    # Normal checkpoints use the actual negative prompt.

    if (
        is_original_krea
        and
        ConditioningZeroOut is not None
    ):

        negative = (
            ConditioningZeroOut.zero_out(
                positive
            )[0]
        )

        print(
            "Krea-2 ZeroOut negative conditioning",
            end=" "
        )

    else:

        negative = CLIPTextEncode.encode(
            clip,
            negative_prompt
        )[0]


    print(
        f"done ({time.time() - t0:.1f}s)"
    )


    # ========================================================
    # LATENT
    # ========================================================

    print(
        "[4/5] Creating latent image... ",
        end="",
        flush=True
    )

    t0 = time.time()


    latent_image = EmptyLatentImage.generate(
        width,
        height,
        batch_size=batch_size
    )[0]


    print(
        f"done ({time.time() - t0:.1f}s)"
    )


    # ========================================================
    # SAMPLING
    # ========================================================

    print(
        f"[5/5] Sampling "
        f"({steps} steps)..."
    )

    t0 = time.time()


    samples = KSampler.sample(
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise
    )[0]


    print(
        "      Sampling done "
        f"({time.time() - t0:.1f}s)"
    )


    # ========================================================
    # VAE DECODE
    # ========================================================

    print(
        "      Decoding image... ",
        end="",
        flush=True
    )

    t0 = time.time()


    decoded = VAEDecode.decode(
        current_vae,
        samples
    )[0].detach()


    print(
        f"done ({time.time() - t0:.1f}s)"
    )


    # ========================================================
    # SAVE
    # ========================================================

    save_path = get_save_path(
        positive_prompt
    )


    image_array = np.array(
        decoded[0] * 255,
        dtype=np.uint8
    )


    Image.fromarray(
        image_array
    ).save(
        save_path
    )


    print(
        "\n💾 Saved:"
        f"\n   {save_path}"
    )


    # ========================================================
    # GOOGLE DRIVE
    # ========================================================

    drive_path = (
        "/content/gdrive/MyDrive/"
        "krea2_turbo"
    )


    if os.path.exists(drive_path):

        shutil.copy(
            save_path,
            drive_path
        )

        print(
            "☁️ Copied to Google Drive:"
            f"\n   {drive_path}"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        f"🧠 Model:"
        f"\n   {model_name}"
    )


    print(
        "\n🎨 LoRAs used:"
    )


    if applied_loras:

        for lora in applied_loras:

            print(
                f"   • {lora}"
            )

    else:

        print(
            "   • None"
        )


    print(
        f"\n🌱 Seed: {seed}"
    )

    print(
        f"⏱️ Total: "
        f"{time.time() - total_start:.1f}s"
    )

    print(
        "=" * 70 + "\n"
    )


    return (
        save_path,
        seed
    )


# ============================================================
# GRADIO
# ============================================================

import gradio as gr


# ============================================================
# GENERATE UI FUNCTION
# ============================================================

def generate_ui(

    model_source,

    positive_prompt,
    negative_prompt,

    width,
    height,

    seed,
    steps,
    batch_size,

    cfg,
    denoise,

    lora1,
    lora1_strength,
    lora1_clip,

    lora2,
    lora2_strength,
    lora2_clip,

    lora3,
    lora3_strength,
    lora3_clip,

    lora4,
    lora4_strength,
    lora4_clip,

    lora5,
    lora5_strength,
    lora5_clip,

    sampler_name="euler",
    scheduler="simple"
):


    # ========================================================
    # LORA ARRAYS
    # ========================================================

    lora_names = [

        lora1,
        lora2,
        lora3,
        lora4,
        lora5
    ]


    lora_strengths = [

        lora1_strength,
        lora2_strength,
        lora3_strength,
        lora4_strength,
        lora5_strength
    ]


    clip_strengths = [

        lora1_clip,
        lora2_clip,
        lora3_clip,
        lora4_clip,
        lora5_clip
    ]


    # ========================================================
    # RANDOM SEED
    # ========================================================

    # Keep original behavior:
    # Seed 0 = random.

    if int(seed) == 0:

        seed = random.randint(
            0,
            0xFFFFFFFF
        )


    # ========================================================
    # INPUT DATA
    # ========================================================

    input_data = {

        "input": {

            "model_source":
                model_source,

            "positive_prompt":
                positive_prompt,

            "negative_prompt":
                negative_prompt,

            "width":
                int(width),

            "height":
                int(height),

            "batch_size":
                int(batch_size),

            "seed":
                int(seed),

            "steps":
                int(steps),

            "cfg":
                float(cfg),

            "sampler_name":
                sampler_name,

            "scheduler":
                scheduler,

            "denoise":
                float(denoise),

            "lora_names":
                lora_names,

            "lora_strengths":
                lora_strengths,

            "clip_strengths":
                clip_strengths
        }
    }


    # ========================================================
    # GENERATE
    # ========================================================

    image_path, used_seed = generate(
        input_data
    )


    return (
        image_path,
        image_path,
        str(used_seed)
    )


# ============================================================
# DEFAULT PROMPTS
# ============================================================

DEFAULT_POSITIVE = """
A high-resolution, surreal digital illustration showing a human hand holding a martini glass.
The image is overlaid with whimsical, expressive ink-style doodles, including a cartoon figure
inside the glass, a drawn citrus wedge on the rim, and various abstract sketches and faces
surrounding the glass against a clean, white background. The style seamlessly blends a realistic,
lit photograph with loose, hand-drawn marker artistry, creating a playful and artistic juxtaposition.
"""


DEFAULT_NEGATIVE = """
low quality, blurry, unnatural skin tone, bad lighting,
pixelated, noise, oversharpen, soft focus
"""


# ============================================================
# CSS
# ============================================================

custom_css = """

.gradio-container {

    font-family:
        'SF Pro Display',
        -apple-system,
        BlinkMacSystemFont,
        sans-serif;
}


.lora-box {

    border:
        1px solid #888;

    border-radius:
        10px;

    padding:
        10px;

    margin-bottom:
        10px;
}


.model-box {

    border:
        2px solid #666;

    border-radius:
        12px;

    padding:
        15px;

    margin-bottom:
        15px;
}

"""


# ============================================================
# MODEL CHOICES
# ============================================================

MODEL_CHOICES = [
    ORIGINAL_MODEL_NAME
] + CHECKPOINT_FILES


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:


    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML(
        """
        <div style="
            width:100%;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            margin:20px 0;
        ">

        <h1 style="
            font-size:2.5em;
            margin-bottom:10px;
        ">
            Krea-2 Turbo + Checkpoint + Multiple LoRA
        </h1>

        <div>
            Select Original Krea-2 or a checkpoint
            and stack up to 5 LoRAs.
        </div>

        </div>
        """
    )


    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    with gr.Row():


        # ====================================================
        # LEFT
        # ====================================================

        with gr.Column():


            # =================================================
            # MODEL SELECTION
            # =================================================

            with gr.Group(
                elem_classes="model-box"
            ):

                gr.Markdown(
                    "## 🧠 Model Selection"
                )


                model_source = gr.Dropdown(

                    choices=MODEL_CHOICES,

                    value=ORIGINAL_MODEL_NAME,

                    label="Model Source",

                    info=(
                        "Original Krea-2 Turbo or a "
                        "checkpoint from "
                        "models/checkpoints"
                    )
                )


            # =================================================
            # PROMPTS
            # =================================================

            positive = gr.Textbox(

                DEFAULT_POSITIVE,

                label="Positive Prompt",

                lines=6
            )


            negative = gr.Textbox(

                DEFAULT_NEGATIVE,

                label="Negative Prompt",

                lines=5
            )


            # =================================================
            # IMAGE SETTINGS
            # =================================================

            with gr.Row():

                width = gr.Number(

                    value=1024,

                    label="Width",

                    precision=0
                )


                height = gr.Number(

                    value=1024,

                    label="Height",

                    precision=0
                )


                seed = gr.Number(

                    value=0,

                    label="Seed (0 = random)",

                    precision=0
                )


            with gr.Row():

                steps = gr.Slider(

                    4,
                    25,

                    value=8,

                    step=1,

                    label="Steps"
                )


                batch_size = gr.Number(

                    value=1,

                    label="Batch Size",

                    precision=0
                )


            # =================================================
            # LORA SETTINGS
            # =================================================

            with gr.Accordion(
                "🎨 LoRA Settings",
                open=True
            ):


                gr.Markdown(
                    """
                    ### Stack multiple LoRAs

                    Leave a slot empty if you
                    don't want to use it.
                    """
                )


                # =============================================
                # LORA 1
                # =============================================

                with gr.Group(
                    elem_classes="lora-box"
                ):

                    gr.Markdown(
                        "### LoRA 1"
                    )


                    lora1 = gr.Dropdown(

                        choices=[""] + LORA_FILES,

                        value="",

                        label="LoRA"
                    )


                    with gr.Row():

                        lora1_strength = gr.Slider(

                            -9.0,
                            9.0,

                            value=1,

                            step=0.05,

                            label="Model Strength"
                        )


                        lora1_clip = gr.Slider(

                            -2.0,
                            2.0,

                            value=1,

                            step=0.05,

                            label="CLIP Strength"
                        )


                # =============================================
                # LORA 2
                # =============================================

                with gr.Group(
                    elem_classes="lora-box"
                ):

                    gr.Markdown(
                        "### LoRA 2"
                    )


                    lora2 = gr.Dropdown(

                        choices=[""] + LORA_FILES,

                        value="",

                        label="LoRA"
                    )


                    with gr.Row():

                        lora2_strength = gr.Slider(

                            -9.0,
                            9.0,

                            value=1,

                            step=0.05,

                            label="Model Strength"
                        )


                        lora2_clip = gr.Slider(

                            -2.0,
                            2.0,

                            value=1,

                            step=0.05,

                            label="CLIP Strength"
                        )


                # =============================================
                # LORA 3
                # =============================================

                with gr.Group(
                    elem_classes="lora-box"
                ):

                    gr.Markdown(
                        "### LoRA 3"
                    )


                    lora3 = gr.Dropdown(

                        choices=[""] + LORA_FILES,

                        value="",

                        label="LoRA"
                    )


                    with gr.Row():

                        lora3_strength = gr.Slider(

                            -9.0,
                            9.0,

                            value=1,

                            step=0.05,

                            label="Model Strength"
                        )


                        lora3_clip = gr.Slider(

                            -2.0,
                            2.0,

                            value=1,

                            step=0.05,

                            label="CLIP Strength"
                        )


                # =============================================
                # LORA 4
                # =============================================

                with gr.Group(
                    elem_classes="lora-box"
                ):

                    gr.Markdown(
                        "### LoRA 4"
                    )


                    lora4 = gr.Dropdown(

                        choices=[""] + LORA_FILES,

                        value="",

                        label="LoRA"
                    )


                    with gr.Row():

                        lora4_strength = gr.Slider(

                            -9.0,
                            9.0,

                            value=1,

                            step=0.05,

                            label="Model Strength"
                        )


                        lora4_clip = gr.Slider(

                            -2.0,
                            2.0,

                            value=1,

                            step=0.05,

                            label="CLIP Strength"
                        )


                # =============================================
                # LORA 5
                # =============================================

                with gr.Group(
                    elem_classes="lora-box"
                ):

                    gr.Markdown(
                        "### LoRA 5"
                    )


                    lora5 = gr.Dropdown(

                        choices=[""] + LORA_FILES,

                        value="",

                        label="LoRA"
                    )


                    with gr.Row():

                        lora5_strength = gr.Slider(

                            -9.0,
                            9.0,

                            value=1,

                            step=0.05,

                            label="Model Strength"
                        )


                        lora5_clip = gr.Slider(

                            -2.0,
                            2.0,

                            value=1,

                            step=0.05,

                            label="CLIP Strength"
                        )


            # =================================================
            # ADVANCED
            # =================================================

            with gr.Accordion(
                "⚙️ Advanced Settings",
                open=False
            ):

                with gr.Row():

                    cfg = gr.Slider(

                        0.5,
                        4.0,

                        value=1.0,

                        step=0.1,

                        label="CFG"
                    )


                    denoise = gr.Slider(

                        0.1,
                        1.0,

                        value=1.0,

                        step=0.05,

                        label="Denoise"
                    )


            # =================================================
            # GENERATE
            # =================================================

            run = gr.Button(

                "🚀 Generate",

                variant="primary",

                size="lg"
            )


        # ====================================================
        # RIGHT
        # ====================================================

        with gr.Column():


            output_img = gr.Image(

                label="Generated Image",

                height=600
            )


            download_image = gr.File(

                label="Download Image"
            )


            used_seed = gr.Textbox(

                label="Seed Used",

                interactive=False
            )


    # ========================================================
    # BUTTON
    # ========================================================

    run.click(

        fn=generate_ui,

        inputs=[

            # Model
            model_source,

            # Prompts
            positive,
            negative,

            # Image
            width,
            height,

            # Seed / steps
            seed,
            steps,
            batch_size,

            # Advanced
            cfg,
            denoise,

            # LoRA 1
            lora1,
            lora1_strength,
            lora1_clip,

            # LoRA 2
            lora2,
            lora2_strength,
            lora2_clip,

            # LoRA 3
            lora3,
            lora3_strength,
            lora3_clip,

            # LoRA 4
            lora4,
            lora4_strength,
            lora4_clip,

            # LoRA 5
            lora5,
            lora5_strength,
            lora5_clip
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
    debug=True
)
