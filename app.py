import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
from datetime import datetime
import io


st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "models" / "pneumonia_model.pth"
CLASS_NAMES = ["normal", "pneumonia"]
IMG_SIZE = 64

if "history" not in st.session_state:
    st.session_state.history = []


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 16 * 16, output_shape),
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(CLASS_NAMES))
    state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def predict(model, device, image: Image.Image):
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1).squeeze(0)
    idx = int(torch.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs.cpu().numpy()



st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #0369a1 100%);
        padding: 2.4rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {font-size: 2rem; font-weight: 800; margin: 0;}
    .hero p {color: #cbd5e1; margin-top: 0.4rem; font-size: 1.02rem;}
    .badge {
        display: inline-block; background: rgba(255,255,255,0.12);
        padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.75rem;
        margin-top: 0.8rem; border: 1px solid rgba(255,255,255,0.2);
    }

    .card {
        background: white; border: 1px solid #e5e7eb; border-radius: 14px;
        padding: 1.4rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .result-positive {
        background: linear-gradient(135deg,#fef2f2,#fee2e2);
        border: 1px solid #fca5a5; border-radius: 16px; padding: 1.6rem; text-align:center;
    }
    .result-negative {
        background: linear-gradient(135deg,#f0fdf4,#dcfce7);
        border: 1px solid #86efac; border-radius: 16px; padding: 1.6rem; text-align:center;
    }
    .result-title {font-size: 1.5rem; font-weight: 800; margin-bottom: 0.2rem;}
    .result-sub {color: #4b5563; font-size: 0.92rem;}

    .metric-box {
        background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
        padding:0.9rem 1rem; text-align:center;
    }
    .metric-box .val {font-size:1.3rem; font-weight:800; color:#0f172a;}
    .metric-box .lab {font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.04em;}

    .disclaimer {
        background:#fffbeb; border:1px solid #fde68a; border-radius:10px;
        padding:0.8rem 1rem; font-size:0.85rem; color:#92400e; margin-top:1rem;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>🫁 Pneumonia Detection AI</h1>
    <p>Deep-learning chest X-ray screening — upload a scan for an instant assessment.</p>
    <span class="badge">CNN · TinyVGG architecture</span>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"Model weights not found at `{MODEL_PATH}`. Train the model and place `pneumonia_model.pth` in the `models/` folder.")
    st.stop()

with st.spinner("Loading model..."):
    model, device = load_model()


with st.sidebar:
    st.subheader("Model Details")
    st.markdown(f"""
    - **Architecture:** TinyVGG (CNN)
    - **Classes:** Normal, Pneumonia
    - **Input size:** {IMG_SIZE}×{IMG_SIZE}
    - **Device:** `{device}`
    """)
    st.divider()
    st.subheader("Session Stats")
    total = len(st.session_state.history)
    pos = sum(1 for h in st.session_state.history if h["label"] == "pneumonia")
    c1, c2 = st.columns(2)
    c1.metric("Scans", total)
    c2.metric("Flagged", pos)
    st.divider()
    st.caption("⚕️ Research / educational tool only. Not a substitute for professional medical diagnosis.")


tab_analyze, tab_history, tab_about = st.tabs(["🔍 Analyze", "🕒 History", "ℹ️ About"])

with tab_analyze:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Upload X-ray")
        uploaded_file = st.file_uploader(
            "JPG or PNG chest X-ray image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        else:
            st.info("Upload a chest X-ray image to begin analysis.")

    with right:
        st.markdown("#### Result")
        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                label, confidence, probs = predict(model, device, image)

            is_pneumonia = label == "pneumonia"
            verdict = "Pneumonia Detected" if is_pneumonia else "No Pneumonia Detected"
            css = "result-positive" if is_pneumonia else "result-negative"
            icon = "⚠️" if is_pneumonia else "✅"

            st.markdown(f"""
            <div class="{css}">
                <div style="font-size:2.2rem;">{icon}</div>
                <div class="result-title">{verdict}</div>
                <div class="result-sub">Model confidence: {confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""<div class="metric-box"><div class="val">{probs[0]*100:.1f}%</div><div class="lab">Normal</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-box"><div class="val">{probs[1]*100:.1f}%</div><div class="lab">Pneumonia</div></div>""", unsafe_allow_html=True)

            st.write("")
            st.progress(float(probs[1]), text="Pneumonia probability")

            st.markdown("""
            <div class="disclaimer">
                ⚠️ <b>Disclaimer:</b> This is an AI screening demo trained on a public dataset — it is
                not a certified diagnostic tool. Always consult a licensed radiologist or physician
                for medical decisions.
            </div>
            """, unsafe_allow_html=True)

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "file": uploaded_file.name,
                "label": label,
                "confidence": confidence,
            })
        else:
            st.markdown("""
            <div class="card">
                <p style="color:#64748b; margin:0;">Your prediction will appear here once you upload an image.</p>
            </div>
            """, unsafe_allow_html=True)

with tab_history:
    st.markdown("#### Session History")
    if st.session_state.history:
        rows = st.session_state.history[::-1]
        st.dataframe(
            [{
                "Time": r["time"],
                "File": r["file"],
                "Result": "Pneumonia" if r["label"] == "pneumonia" else "Normal",
                "Confidence": f"{r['confidence']*100:.1f}%",
            } for r in rows],
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No scans analyzed yet this session.")

with tab_about:
    st.markdown("""
    #### About this project
    A convolutional neural network (TinyVGG) trained to classify chest X-rays as
    **Normal** or **Pneumonia**, built as an applied deep-learning project.

    **Pipeline:**
    - Dataset organized into `normal/` and `pneumonia/` folders (`ImageFolder` labeling)
    - CNN trained with PyTorch, images resized to 64×64
    - Weights exported and served here via a Streamlit interface

    **Limitations:**
    - Trained on a public dataset, not clinically validated
    - Small input resolution (64×64) trades off some diagnostic detail for speed
    - Intended for portfolio / educational demonstration only
    """)
