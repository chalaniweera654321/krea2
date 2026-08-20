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
# KREA-2 TURBO + MULTIPLE LORA
# ============================================================

print("\n" + "=" * 60)
print("        KREA-2 TURBO + Multiple LoRA")
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
ConditioningZeroOut = NODE_CLASS_MAPPINGS.get("ConditioningZeroOut", None)()


# ============================================================
# BASE MODEL
# ============================================================

startup_start = time.time()

with torch.inference_mode():

    print("\n[1/3] Loading UNet... ", end="", flush=True)

    t0 = time.time()

    base_model = UNETLoader.load_unet(
        "krea2_turbo_fp8_scaled.safetensors",
        "default"
    )[0]

    print(f"done ({time.time() - t0:.1f}s)")


    print("[2/3] Loading CLIP (Qwen3-VL)... ", end="", flush=True)

    t0 = time.time()

    base_clip = CLIPLoader.load_clip(
        "qwen3vl_4b_fp8_scaled.safetensors",
        type="krea2"
    )[0]

    print(f"done ({time.time() - t0:.1f}s)")


    print("[3/3] Loading VAE... ", end="", flush=True)

    t0 = time.time()

    vae = VAELoader.load_vae(
        "qwen_image_vae.safetensors"
    )[0]

    print(f"done ({time.time() - t0:.1f}s)")


print(
    f"\n✅ Base models loaded in "
    f"{time.time() - startup_start:.1f}s"
)

print("=" * 60)


# ============================================================
# LORA DIRECTORY
# ============================================================

# Normal ComfyUI location
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
                (".safetensors", ".pt", ".ckpt")
            ):

                # Return path relative to models/loras
                relative_path = os.path.relpath(
                    os.path.join(root, filename),
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
# APPLY MULTIPLE LORAS
# ============================================================

def apply_loras(
    lora_names,
    lora_strengths,
    clip_strengths
):

    model = base_model
    clip = base_clip

    applied = []

    for i in range(len(lora_names)):

        lora_name = lora_names[i]

        if not lora_name:
            continue

        model_strength = float(
            lora_strengths[i]
        )

        clip_strength = float(
            clip_strengths[i]
        )


        # Skip completely disabled LoRA
        if model_strength == 0 and clip_strength == 0:
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


    return model, clip, applied


# ============================================================
# SAVE HELPERS
# ============================================================

save_dir = "./results"

os.makedirs(
    save_dir,
    exist_ok=True
)


def get_save_path(prompt):

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
def generate(input):

    values = input["input"]


    # --------------------------------------------------------
    # Basic settings
    # --------------------------------------------------------

    positive_prompt = values[
        "positive_prompt"
    ]

    negative_prompt = values[
        "negative_prompt"
    ]

    seed = values["seed"]

    steps = values["steps"]

    cfg = values["cfg"]

    sampler_name = values[
        "sampler_name"
    ]

    scheduler = values[
        "scheduler"
    ]

    denoise = values["denoise"]

    width = values["width"]

    height = values["height"]

    batch_size = values[
        "batch_size"
    ]


    # --------------------------------------------------------
    # LoRA settings
    # --------------------------------------------------------

    lora_names = values[
        "lora_names"
    ]

    lora_strengths = values[
        "lora_strengths"
    ]

    clip_strengths = values[
        "clip_strengths"
    ]


    print("\n" + "=" * 60)
    print("              NEW GENERATION")
    print("=" * 60)


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
        lora_strengths,
        clip_strengths
    )


    print(
        f"\n✅ Applied {len(applied_loras)} LoRA(s)"
    )

    print(
        f"   Time: {time.time() - t0:.1f}s"
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


    positive = CLIPTextEncode.encode(
        clip,
        positive_prompt
    )[0]


    # Krea-2 uses ConditioningZeroOut for negative prompts instead of CLIPTextEncode
    if ConditioningZeroOut is not None:
        negative = ConditioningZeroOut.zero_out(
            positive
        )[0]
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


    if os.path.exists(drive_path):

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
        f"\n🎨 LoRAs used:"
    )

    if applied_loras:

        for lora in applied_loras:
            print(f"   • {lora}")

    else:

        print("   • None")


    print(
        f"\n🌱 Seed: {seed}"
    )

    print(
        f"⏱️ Total: "
        f"{time.time() - total_start:.1f}s"
    )

    print("=" * 60 + "\n")


    return save_path, seed


# ============================================================
# GRADIO
# ============================================================

import gradio as gr


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

    lora_data,

    sampler_name="euler",
    scheduler="simple"
):

    # lora_data is a list like:
    # [{"name": "oil_paint.safetensors", "strength": 1.0}, ...]

    lora_names = []
    lora_strengths = []
    clip_strengths = []

    for item in (lora_data or []):
        name = item.get("name", "")
        strength = float(item.get("strength", 1.0))

        if name:
            lora_names.append(name)
            lora_strengths.append(strength)

            # Keep CLIP strength at 1.0 because the UI only exposes
            # the requested Model Strength control.
            clip_strengths.append(1.0)

    input_data = {
        "input": {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,

            "width": int(width),
            "height": int(height),
            "batch_size": int(batch_size),

            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),

            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": float(denoise),

            "lora_names": lora_names,
            "lora_strengths": lora_strengths,
            "clip_strengths": clip_strengths
        }
    }

    image_path, used_seed = generate(input_data)

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

.lora-panel {
    border: 1px solid #777;
    border-radius: 8px;
    padding: 12px;
}

.lora-row {
    width: 100%;
    align-items: center;
    margin: 6px 0;
}

.lora-list {
    width: 100%;
}

.lora-row {
    display: flex !important;
    width: 100%;
    align-items: center;
    gap: 10px;
    margin: 7px 0;
}

.lora-strength-input {
    width: 72px;
    min-width: 72px;
    height: 42px;
    box-sizing: border-box;
    padding: 0 8px;
    text-align: center;
    border: 1px solid #888;
    border-radius: 6px;
    background: var(--input-background-fill);
    color: var(--body-text-color);
    font-size: 16px;
}

.lora-name-display {
    flex: 1;
    min-width: 0;
    height: 42px;
    box-sizing: border-box;
    display: flex;
    align-items: center;
    padding: 0 12px;
    border: 1px solid #888;
    border-radius: 6px;
    background: var(--input-background-fill);
    color: var(--body-text-color);
    font-size: 16px;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}

.lora-remove-button {
    width: 44px !important;
    min-width: 44px !important;
    max-width: 44px !important;
    height: 42px !important;
    padding: 0 !important;
    border: 1px solid #888 !important;
    border-radius: 6px !important;
    font-size: 20px !important;
    cursor: pointer;
}

.lora-empty {
    padding: 12px;
    opacity: 0.7;
    text-align: center;
    border: 1px dashed #888;
    border-radius: 6px;
}

.lora-picker {
    margin-bottom: 10px;
}
"""



# ============================================================
# DYNAMIC GRADIO UI
# ============================================================

# IMPORTANT:
# Do NOT use gr.render()/gr.State for the LoRA rows here.
# Some Gradio versions can leave dynamically-rendered components in the
# event graph after they are removed, producing: KeyError: 0
# The LoRA rows below are normal HTML controlled by JavaScript. A hidden
# Gradio Textbox stores the current LoRA list as JSON.

import json
import html


def parse_lora_json(value):
    if isinstance(value, list):
        data = value
    else:
        try:
            data = json.loads(value or "[]")
        except Exception:
            data = []

    result = []
    if not isinstance(data, list):
        return result

    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            strength = float(item.get("strength", 1.0))
        except Exception:
            strength = 1.0
        strength = max(-9.0, min(9.0, strength))
        result.append({"name": name, "strength": strength})

    return result


def lora_json_string(loras):
    return json.dumps(parse_lora_json(loras), separators=(",", ":"))


def render_lora_rows(loras):
    loras = parse_lora_json(loras)

    if not loras:
        return '''
        <div class="lora-empty">
            Select a LoRA from the dropdown above to add it.
        </div>
        '''

    rows = []
    for i, item in enumerate(loras):
        name = html.escape(item["name"], quote=True)
        strength = item["strength"]
        rows.append(f'''
        <div class="lora-row" data-index="{i}">
            <input class="lora-strength-input" type="number"
                   min="-9" max="9" step="0.05"
                   value="{strength:g}" title="Model Strength">
            <div class="lora-name-display" title="{name}">{name}</div>
            <button type="button" class="lora-remove-button"
                    data-index="{i}" title="Remove LoRA">✕</button>
        </div>
        ''')

    return '<div class="lora-list">' + "".join(rows) + '</div>'


def add_lora(selected_lora, current_json):
    loras = parse_lora_json(current_json)

    if not selected_lora:
        return render_lora_rows(loras), lora_json_string(loras), gr.update(value=None)

    loras.append({"name": str(selected_lora), "strength": 1.0})

    print(f"➕ Added LoRA: {selected_lora} (Model Strength: 1.0)")

    return (
        render_lora_rows(loras),
        lora_json_string(loras),
        gr.update(value=None)
    )


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
    lora_json,
    sampler_name="euler",
    scheduler="simple"
):
    loras = parse_lora_json(lora_json)

    lora_names = []
    lora_strengths = []
    clip_strengths = []

    for item in loras:
        lora_names.append(item["name"])
        lora_strengths.append(float(item["strength"]))
        clip_strengths.append(1.0)

    input_data = {
        "input": {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": float(denoise),
            "lora_names": lora_names,
            "lora_strengths": lora_strengths,
            "clip_strengths": clip_strengths
        }
    }

    image_path, used_seed = generate(input_data)
    return image_path, image_path, used_seed


custom_js = r'''
function getLoraJsonElement() {
    return document.querySelector("#lora_json textarea, #lora_json input");
}

function getLoraList() {
    const el = getLoraJsonElement();
    if (!el) return [];
    try {
        const parsed = JSON.parse(el.value || "[]");
        return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
        console.error("Could not parse LoRA JSON:", e);
        return [];
    }
}

function setLoraList(list) {
    const el = getLoraJsonElement();
    if (!el) return;

    const value = JSON.stringify(list);
    const taSetter = Object.getOwnPropertyDescriptor(
        HTMLTextAreaElement.prototype, "value"
    )?.set;
    const inputSetter = Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype, "value"
    )?.set;

    if (el.tagName === "TEXTAREA" && taSetter) taSetter.call(el, value);
    else if (inputSetter) inputSetter.call(el, value);
    else el.value = value;

    el.dispatchEvent(new Event("input", {bubbles: true}));
    el.dispatchEvent(new Event("change", {bubbles: true}));
}

document.addEventListener("input", function(event) {
    const input = event.target.closest(".lora-strength-input");
    if (!input) return;

    const row = input.closest(".lora-row");
    if (!row) return;

    const index = Number(row.dataset.index);
    const list = getLoraList();
    if (!Number.isInteger(index) || !list[index]) return;

    let strength = Number(input.value);
    if (!Number.isFinite(strength)) strength = 1.0;
    strength = Math.max(-9, Math.min(9, strength));

    list[index].strength = strength;
    setLoraList(list);
});

document.addEventListener("click", function(event) {
    const button = event.target.closest(".lora-remove-button");
    if (!button) return;

    event.preventDefault();
    event.stopPropagation();

    const row = button.closest(".lora-row");
    if (!row) return;

    const index = Number(row.dataset.index);
    const list = getLoraList();
    if (!Number.isInteger(index) || index < 0 || index >= list.length) return;

    list.splice(index, 1);
    setLoraList(list);

    const panel = document.querySelector("#lora_list");
    if (!panel) return;

    const rows = panel.querySelectorAll(".lora-row");
    if (rows[index]) rows[index].remove();

    panel.querySelectorAll(".lora-row").forEach(function(r, i) {
        r.dataset.index = i;
        const btn = r.querySelector(".lora-remove-button");
        if (btn) btn.dataset.index = i;
    });

    if (!panel.querySelector(".lora-row")) {
        panel.innerHTML =
            '<div class="lora-empty">Select a LoRA from the dropdown above to add it.</div>';
    }
});
'''


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css,
    js=custom_js
) as demo:

    gr.HTML("""
    <div style="
        width:100%;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        margin:20px 0;
    ">
        <h1 style="font-size:2.5em; margin-bottom:10px;">
            Krea-2 Turbo + Multiple LoRA
        </h1>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=6)

            negative = gr.Textbox(
                DEFAULT_NEGATIVE,
                label="Negative Prompt (Note: Krea-2 natively uses ZeroOut for negatives)",
                lines=5
            )

            with gr.Row():
                width = gr.Number(value=1024, label="Width", precision=0)
                height = gr.Number(value=1024, label="Height", precision=0)
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)

            with gr.Row():
                steps = gr.Slider(4, 25, value=8, step=1, label="Steps")
                batch_size = gr.Number(value=1, label="Batch Size", precision=0)

            with gr.Group(elem_classes="lora-panel"):
                gr.Markdown("### 🎨 LoRAs")

                lora_picker = gr.Dropdown(
                    choices=LORA_FILES,
                    value=None,
                    label="Loras",
                    elem_classes="lora-picker",
                    allow_custom_value=False
                )

                # Normal hidden component. No gr.State and no gr.render.
                lora_json = gr.Textbox(
                    value="[]",
                    elem_id="lora_json",
                    visible=False
                )

                lora_list = gr.HTML(
                    value=render_lora_rows([]),
                    elem_id="lora_list"
                )

            lora_picker.change(
                fn=add_lora,
                inputs=[lora_picker, lora_json],
                outputs=[lora_list, lora_json, lora_picker]
            )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                    denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")

            run = gr.Button("🚀 Generate", variant="primary", size="lg")

        with gr.Column():
            output_img = gr.Image(label="Generated Image", height=600)
            download_image = gr.File(label="Download Image")
            used_seed = gr.Textbox(label="Seed Used", interactive=False)

    run.click(
        fn=generate_ui,
        inputs=[
            positive, negative,
            width, height,
            seed, steps, batch_size,
            cfg, denoise,
            lora_json
        ],
        outputs=[output_img, download_image, used_seed]
    )


# ============================================================
# LAUNCH
# ============================================================

demo.launch(
    share=True,
    debug=True
)
