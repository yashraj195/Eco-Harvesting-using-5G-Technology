import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import os

from model import load_paddy_model, get_transform, SpatioTemporalModel

LABELS = [
    'normal',
    'dead_heart',
    'downy_mildew',
    'brown_spot',
    'blast',
    'tungro',
    'bacterial_panicle_blight',
    'bacterial_leaf_blight',
    'hispa',
    'bacterial_leaf_streak'
]

LABEL_DISPLAY = {
    'normal':                    'Healthy / Normal',
    'dead_heart':                'Dead Heart',
    'downy_mildew':              'Downy Mildew',
    'brown_spot':                'Brown Spot',
    'blast':                     'Blast (Magnaporthe oryzae)',
    'tungro':                    'Tungro',
    'bacterial_panicle_blight':  'Bacterial Panicle Blight',
    'bacterial_leaf_blight':     'Bacterial Leaf Blight',
    'hispa':                     'Hispa',
    'bacterial_leaf_streak':     'Bacterial Leaf Streak',
}

MODEL_PATH = 'paddy_stdd_model_full.pt'
NUM_CLASSES = len(LABELS)

model = load_paddy_model(MODEL_PATH, num_classes=NUM_CLASSES)
transform = get_transform()


def predict_paddy_disease(image: Image.Image):
    """Returns a dict of {label: confidence} for gr.Label, plus a markdown summary."""
    image = image.convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    sequence_input = torch.stack([input_tensor] * 3, dim=1)

    with torch.no_grad():
        logits = model(sequence_input)
        probs  = F.softmax(logits, dim=1).squeeze(0)  # shape (num_classes,)

    # Build confidence dict for gr.Label  { "Display Name": probability }
    confidence_dict = {
        LABEL_DISPLAY[lbl]: float(probs[i])
        for i, lbl in enumerate(LABELS)
    }

    # Top prediction
    top_idx   = int(probs.argmax())
    top_label = LABELS[top_idx]
    top_conf  = float(probs[top_idx]) * 100

    # Severity hint based on confidence
    if top_conf >= 90:
        severity_badge = "🔴 HIGH confidence"
    elif top_conf >= 70:
        severity_badge = "🟡 MODERATE confidence"
    else:
        severity_badge = "🟢 LOW confidence — consider re-uploading a clearer image"

    summary_md = f"""
## 🌾 Prediction Result

| | |
|---|---|
| **Detected condition** | {LABEL_DISPLAY[top_label]} |
| **Confidence** | **{top_conf:.1f}%** |
| **Status** | {severity_badge} |

> *The model processes the image as a temporal sequence (3 frames) using the Spatio-Temporal Disease Detection (STDD) framework — ResNet-50 spatial backbone + Transformer temporal encoder.*
"""
    return confidence_dict, summary_md


# ── Custom CSS ────────────────────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --green-deep:  #1a3a2a;
    --green-mid:   #2e6b45;
    --green-light: #4caf78;
    --green-pale:  #d6f0e0;
    --gold:        #c9a84c;
    --cream:       #faf8f2;
    --text-dark:   #1c2b22;
    --text-mid:    #3d5a47;
    --radius:      14px;
}

/* ── Force light theme on everything ── */
html, body,
.gradio-container,
.gradio-container *,
.main, .wrap, .panel,
div.svelte-1gfkfd6,
.block, .form,
footer { 
    background-color: transparent !important;
    color: var(--text-dark) !important;
}

body, .gradio-container {
    background: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Force all markdown / paragraph text to be dark */
p, span, label, .prose, .md, markdown,
.output-markdown, .input-markdown,
.svelte-1ed2p3z, .svelte-df6pnx {
    color: var(--text-dark) !important;
}

/* ── Panels & blocks ── */
.gap, .gr-group, .gr-box,
div[data-testid="block"],
div[data-testid="column"] {
    background: transparent !important;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, var(--green-deep) 0%, var(--green-mid) 60%, var(--green-light) 100%) !important;
    border-radius: var(--radius);
    padding: 2.4rem 2.8rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.hero-banner h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.2rem !important;
    color: #ffffff !important;
    margin: 0 0 0.4rem !important;
    letter-spacing: -0.5px;
}
.hero-banner p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.97rem !important;
    margin: 0 !important;
    font-weight: 300;
    max-width: 640px;
}
.hero-tag {
    display: inline-block;
    background: var(--gold);
    color: #1a1a1a !important;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.9rem;
}

/* ── About card ── */
.about-card {
    background: #ffffff !important;
    border: 1px solid #d8ead1;
    border-left: 4px solid var(--green-mid);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
    font-size: 0.92rem;
    line-height: 1.7;
    color: var(--text-mid) !important;
}
.about-card * { color: var(--text-mid) !important; }
.about-card strong { color: var(--green-deep) !important; }

/* ── Metric pills ── */
.metrics-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-bottom: 1.4rem;
}
.metric-pill {
    background: var(--green-deep) !important;
    color: #fff !important;
    border-radius: 40px;
    padding: 0.45rem 1.1rem;
    font-size: 0.82rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 6px;
}
.metric-pill * { color: #fff !important; }
.metric-pill span { color: var(--gold) !important; font-weight: 700; }

/* ── Section label ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--green-mid) !important;
    margin-bottom: 0.5rem;
}

/* ── Upload area ── */
.upload-box,
.upload-box .wrap,
.upload-box [data-testid="image"] {
    border: 2px dashed var(--green-light) !important;
    border-radius: var(--radius) !important;
    background: #f2faf5 !important;
}

/* ── Buttons ── */
button.primary, button[variant="primary"],
.gr-button-primary {
    background: var(--green-mid) !important;
    border: none !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
button.primary:hover { background: var(--green-deep) !important; }

/* ── Result summary markdown ── */
.result-md,
.result-md > *,
.result-md p,
.result-md td,
.result-md th,
.result-md h2 {
    background: #f2faf5 !important;
    color: var(--text-dark) !important;
}
.result-md {
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
}
.result-md table { width: 100%; border-collapse: collapse; }
.result-md td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #d8ead1; font-size: 0.92rem; }
.result-md h2 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--green-deep) !important;
    font-size: 1.35rem !important;
}
.result-md blockquote {
    border-left: 3px solid var(--green-light);
    padding-left: 0.8rem;
    color: #5a7a66 !important;
    font-size: 0.82rem;
}

/* ── Label/confidence output ── */
.label-output,
.label-output *,
.label-output span,
.label-output .output-class,
.label-output .confidence {
    color: var(--text-dark) !important;
    background: transparent !important;
}

/* ── Footer ── */
.footer-note {
    text-align: center;
    font-size: 0.78rem;
    color: #6a907a !important;
    margin-top: 1.6rem;
    padding-top: 1rem;
    border-top: 1px solid #dceee3;
}
.footer-note * { color: #6a907a !important; }
"""

# ── Layout ────────────────────────────────────────────────────────────────────
light_theme = gr.themes.Default(
    primary_hue="green",
    neutral_hue="stone",
).set(
    body_background_fill="#faf8f2",
    body_text_color="#1c2b22",
    block_background_fill="#ffffff",
    block_label_text_color="#1c2b22",
    input_background_fill="#f2faf5",
    table_even_background_fill="#f7fcf9",
    table_odd_background_fill="#ffffff",
)

with gr.Blocks(css=custom_css, theme=light_theme, title="Paddy Disease Detection") as demo:

    # Hero
    gr.HTML("""
    <div class="hero-banner">
      <div class="hero-tag">🌾 Precision Agriculture · AI</div>
      <h1>Paddy Disease Detection</h1>
      <p>Upload a rice leaf image to instantly classify its health condition using a
         Spatio-Temporal Deep Learning model — ResNet-50 spatial features fused with
         a Transformer temporal encoder.</p>
    </div>
    """)

    # About + metrics
    gr.HTML("""
    <div class="about-card">
      <strong>About this project — </strong>
      This tool is built on the <em>Spatio-Temporal Disease Detection (STDD)</em> framework
      developed at GITAM University, Visakhapatnam. It couples a <strong>ResNet-50 CNN</strong>
      backbone for per-frame spatial feature extraction with a <strong>multi-head Transformer encoder</strong>
      that models disease progression across ordered image sequences. A weighted late-fusion layer
      (α = 0.6) combines spatial appearance and temporal progression scores before final classification.
      The model is trained and evaluated on the <strong>PlantVillage rice-leaf subset</strong>
      (5,932 images, 10 classes).
    </div>
    <div class="metrics-row">
      <div class="metric-pill">🎯 Accuracy <span>95.4%</span></div>
      <div class="metric-pill">📌 Precision <span>94.1%</span></div>
      <div class="metric-pill">🔁 Recall <span>93.6%</span></div>
      <div class="metric-pill">⚡ Inference <span>&lt; 3 s</span></div>
      <div class="metric-pill">🦠 Classes <span>10</span></div>
    </div>
    """)

    # Main inference block
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">Upload Leaf Image</div>')
            img_input = gr.Image(
                type="pil",
                label="",
                elem_classes=["upload-box"],
            )
            submit_btn = gr.Button("🔬 Analyse Image", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">Prediction Summary</div>')
            result_md = gr.Markdown(
                value="*Upload an image and click **Analyse Image** to see results.*",
                elem_classes=["result-md"],
            )
            gr.HTML('<div class="section-label" style="margin-top:1rem">Confidence Scores (all classes)</div>')
            label_out = gr.Label(
                num_top_classes=NUM_CLASSES,
                label="",
                elem_classes=["label-output"],
            )

    submit_btn.click(
        fn=predict_paddy_disease,
        inputs=img_input,
        outputs=[label_out, result_md],
    )

    gr.HTML("""
    <div class="footer-note">
      STDD Framework · GITAM School of CSE · Visakhapatnam, Andhra Pradesh, India<br>
      Model: ResNet-50 + Transformer Encoder · Dataset: PlantVillage · Framework: PyTorch 2.2
    </div>
    """)

if __name__ == "__main__":
    demo.launch(debug=True)
