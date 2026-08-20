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
# KREA-2 TURBO + 10 LORA
# ============================================================

print("\n" + "=" * 60)
print("        KREA-2 TURBO + 10 LoRA")
print("=" * 60)


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


# ============================================================
# BASE MODEL
# ============================================================

startup_start = time.time()

with torch.inference_mode():

    # --------------------------------------------------------
    # UNET
    # --------------------------------------------------------

    print("\n[1/3] Loading UNet... ", end="", flush=True)

    t0 = time.time()

    base_model = UNETLoader.load_unet(
        "krea2_turbo_fp8_scaled.safetensors",
        "default"
    )[0]

    print(
        f"done ({time.time() - t0:.1f}s)"
    )


    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    print(
        "[2/3] Loading CLIP (Qwen3-VL)... ",
        end="",
        flush=True
    )

    t0 = time.time()

    base_clip = CLIPLoader.load_clip(
        "qwen3vl_4b_fp8_scaled.safetensors",
        type="krea2"
    )[0]

    print(
        f"done ({time.time() - t0:.1f}s)"
    )


    # --------------------------------------------------------
    # VAE
    # --------------------------------------------------------

    print(
        "[3/3] Loading VAE... ",
        end="",
        flush=True
    )

    t0 = time.time()

    vae = VAELoader.load_vae(
        "qwen_image_vae.safetensors"
    )[0]

    print(
        f"done ({time.time() - t0:.1f}s)"
    )


print(
    f"\n✅ Base models loaded in "
    f"{time.time() - startup_start:.1f}s"
)

print("=" * 60)


# ============================================================
# LORA DIRECTORY
# ============================================================

LORA_DIR = "./models/loras"


def get_lora_files():

    if not os.path.exists(LORA_DIR):

        print(
            f"\n⚠️ LoRA directory not found:"
            f"\n{os.path.abspath(LORA_DIR)}"
        )

        return [""]


    files = []


    for root, dirs, filenames in os.walk(LORA_DIR):

        for filename in filenames:

            if filename.lower().endswith(
                (
                    ".safetensors",
                    ".pt",
                    ".ckpt"
                )
            ):

                relative_path = os.path.relpath(
                    os.path.join(root, filename),
                    LORA_DIR
                )

                files.append(
                    relative_path
                )


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

        print(
            f"   • {f}"
        )


    return files


LORA_FILES = get_lora_files()


# ============================================================
# APPLY MULTIPLE LORAS
# ============================================================

def apply_loras(
    lora_names,
    lora_strengths
):

    model = base_model
    clip = base_clip

    applied = []


    for i in range(
        len(lora_names)
    ):

        lora_name = lora_names[i]


        # ----------------------------------------------------
        # Empty slot
        # ----------------------------------------------------

        if not lora_name:

            continue


        # ----------------------------------------------------
        # Read strength
        # ----------------------------------------------------

        try:

            strength = float(
                lora_strengths[i]
            )

        except:

            strength = 1.0


        # ----------------------------------------------------
        # Zero = disabled
        # ----------------------------------------------------

        if strength == 0:

            continue


        print(
            f"\n   [{i + 1}] Applying LoRA:"
        )

        print(
            f"       {lora_name}"
        )

        print(
            f"       Strength: {strength}"
        )


        t0 = time.time()


        # ----------------------------------------------------
        # Apply LoRA
        #
        # Same strength is used for:
        # Model + CLIP
        # ----------------------------------------------------

        model, clip = LoraLoader.load_lora(
            model,
            clip,
            lora_name,
            strength,
            strength
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
# SAVE HELPERS
# ============================================================

save_dir = "/content/drive/MyDrive/results"

os.makedirs(
    save_dir,
    exist_ok=True
)


def get_save_path(
    prompt
):

    safe_prompt = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        prompt
    )[:25]


    uid = uuid.uuid4().hex[:6]


    filename = (
        f"{safe_prompt}_{uid}.png"
    )


    return os.path.join(
        save_dir,
        filename
    )


# ============================================================
# GENERATION
# ============================================================

@torch.inference_mode()
def generate(
    input
):

    values = input["input"]


    # ========================================================
    # BASIC SETTINGS
    # ========================================================

    positive_prompt = values[
        "positive_prompt"
    ]

    negative_prompt = values[
        "negative_prompt"
    ]


    seed = values[
        "seed"
    ]


    steps = values[
        "steps"
    ]


    cfg = values[
        "cfg"
    ]


    sampler_name = values[
        "sampler_name"
    ]


    scheduler = values[
        "scheduler"
    ]


    denoise = values[
        "denoise"
    ]


    width = values[
        "width"
    ]


    height = values[
        "height"
    ]


    batch_size = values[
        "batch_size"
    ]


    # ========================================================
    # LORA SETTINGS
    # ========================================================

    lora_names = values[
        "lora_names"
    ]


    lora_strengths = values[
        "lora_strengths"
    ]


    # ========================================================
    # START
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "              NEW GENERATION"
    )

    print(
        "=" * 60
    )


    total_start = time.time()


    # ========================================================
    # APPLY LORAS
    # ========================================================

    print(
        "\n[1/5] Applying LoRAs..."
    )


    t0 = time.time()


    model, clip, applied_loras = apply_loras(
        lora_names,
        lora_strengths
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
        "\n[2/5] Encoding prompts... ",
        end="",
        flush=True
    )


    t0 = time.time()


    # --------------------------------------------------------
    # Positive
    # --------------------------------------------------------

    positive = CLIPTextEncode.encode(
        clip,
        positive_prompt
    )[0]


    # --------------------------------------------------------
    # Negative
    #
    # IMPORTANT:
    # We actually use the negative prompt here.
    # --------------------------------------------------------

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
        "[3/5] Creating latent image... ",
        end="",
        flush=True
    )


    t0 = time.time()


    latent_image = EmptyLatentImage.generate(
        int(width),
        int(height),
        batch_size=int(batch_size)
    )[0]


    print(
        f"done ({time.time() - t0:.1f}s)"
    )


    # ========================================================
    # SAMPLING
    # ========================================================

    print(
        f"[4/5] Sampling "
        f"({steps} steps)..."
    )


    t0 = time.time()


    samples = KSampler.sample(
        model,
        int(seed),
        int(steps),
        float(cfg),
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=float(denoise)
    )[0]


    print(
        f"      Sampling done "
        f"({time.time() - t0:.1f}s)"
    )


    # ========================================================
    # VAE DECODE
    # ========================================================

    print(
        "[5/5] Decoding image... ",
        end="",
        flush=True
    )


    t0 = time.time()


    decoded = VAEDecode.decode(
        vae,
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
        f"\n💾 Saved:"
        f"\n   {save_path}"
    )


    # ========================================================
    # GOOGLE DRIVE
    # ========================================================

    drive_path = (
        "/content/gdrive/MyDrive/"
        "krea2_turbo"
    )


    if os.path.exists(
        drive_path
    ):

        shutil.copy(
            save_path,
            drive_path
        )


        print(
            f"☁️ Copied to Google Drive:"
            f"\n   {drive_path}"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

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
        "=" * 60 + "\n"
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


    # ========================================================
    # LORA NAMES
    # ========================================================

    lora_names = [

        lora1,
        lora2,
        lora3,
        lora4,
        lora5,
        lora6,
        lora7,
        lora8,
        lora9,
        lora10

    ]


    # ========================================================
    # LORA STRENGTHS
    # ========================================================

    lora_strengths = [

        lora1_strength,
        lora2_strength,
        lora3_strength,
        lora4_strength,
        lora5_strength,
        lora6_strength,
        lora7_strength,
        lora8_strength,
        lora9_strength,
        lora10_strength

    ]


    # ========================================================
    # INPUT DATA
    # ========================================================

    input_data = {

        "input": {

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
                lora_strengths

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
        used_seed
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

"""


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:


    # ========================================================
    # TITLE
    # ========================================================

    gr.HTML("""
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
Krea-2 Turbo + 10 LoRA
</h1>

</div>
""")


    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    with gr.Row:


        # ====================================================
        # LEFT
        # ====================================================

        with gr.Column:


            # =================================================
            # POSITIVE PROMPT
            # =================================================

            positive = gr.Textbox(
                DEFAULT_POSITIVE,
                label="Positive Prompt",
                lines=6
            )


            # =================================================
            # NEGATIVE PROMPT
            # =================================================

            negative = gr.Textbox(
                DEFAULT_NEGATIVE,
                label="Negative Prompt",
                lines=5
            )


            # =================================================
            # IMAGE SETTINGS
            # =================================================

            with gr.Row:


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


            # =================================================
            # STEPS / BATCH
            # =================================================

            with gr.Row:


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
                    Select up to **10 LoRAs**.
                    Leave unused slots empty.
                    Strength defaults to **1**.
                    """
                )


                # =================================================
                # LORA 1
                # =================================================

                with gr.Row:

                    lora1 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA 1",
                        scale=4
                    )


                    lora1_strength = gr.Textbox(
                        value="1",
                        label="Strength",
                        scale=1
                    )


                # =================================================
                # LORA 2
                # =================================================

                with gr.Row:

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


                # =================================================
                # LORA 3
                # =================================================

                with gr.Row:

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


                # =================================================
                # LORA 4
                # =================================================

                with gr.Row:

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


                # =================================================
                # LORA 5
                # =================================================

                with gr.Row:

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


                # =================================================
                # LORA 6
                # =================================================

                with gr.Row:

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


                # =================================================
                # LORA 7
                # =================================================

                with gr.Row:

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


                # =================================================
                # LORA 8
                # =================================================

                with gr.Row:

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


                # =================================================
                # LORA 9
                # =================================================

                with gr.Row:

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


                # =================================================
                # LORA 10
                # =================================================

                with gr.Row:

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


            # =================================================
            # ADVANCED
            # =================================================

            with gr.Accordion(
                "⚙️ Advanced Settings",
                open=False
            ):


                with gr.Row:


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
            # GENERATE BUTTON
            # =================================================

            run = gr.Button(
                "🚀 Generate",
                variant="primary",
                size="lg"
            )


        # ====================================================
        # RIGHT
        # ====================================================

        with gr.Column:


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
    # LORA DUPLICATE PREVENTION
    # ========================================================

    all_lora_dropdowns = [

        lora1,
        lora2,
        lora3,
        lora4,
        lora5,
        lora6,
        lora7,
        lora8,
        lora9,
        lora10

    ]


    def update_lora_choices(*selected):

        updated = []


        for current in selected:

            choices = [""] + [
                x for x in LORA_FILES
                if x == current or x not in selected
            ]


            updated.append(
                gr.update(
                    choices=choices
                )
            )


        return updated


    for dropdown in all_lora_dropdowns:

        dropdown.change(

            fn=update_lora_choices,

            inputs=all_lora_dropdowns,

            outputs=all_lora_dropdowns

        )


    # ========================================================
    # GENERATE BUTTON
    # ========================================================

    run.click(

        fn=generate_ui,

        inputs=[

            positive,
            negative,

            width,
            height,

            seed,
            steps,
            batch_size,

            cfg,
            denoise,

            # ----------------------------------------------
            # LoRA 1
            # ----------------------------------------------

            lora1,
            lora1_strength,

            # ----------------------------------------------
            # LoRA 2
            # ----------------------------------------------

            lora2,
            lora2_strength,

            # ----------------------------------------------
            # LoRA 3
            # ----------------------------------------------

            lora3,
            lora3_strength,

            # ----------------------------------------------
            # LoRA 4
            # ----------------------------------------------

            lora4,
            lora4_strength,

            # ----------------------------------------------
            # LoRA 5
            # ----------------------------------------------

            lora5,
            lora5_strength,

            # ----------------------------------------------
            # LoRA 6
            # ----------------------------------------------

            lora6,
            lora6_strength,

            # ----------------------------------------------
            # LoRA 7
            # ----------------------------------------------

            lora7,
            lora7_strength,

            # ----------------------------------------------
            # LoRA 8
            # ----------------------------------------------

            lora8,
            lora8_strength,

            # ----------------------------------------------
            # LoRA 9
            # ----------------------------------------------

            lora9,
            lora9_strength,

            # ----------------------------------------------
            # LoRA 10
            # ----------------------------------------------

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
    debug=True
)
