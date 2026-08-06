import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ---------- Config ----------
st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="centered")
MODEL_PATH = Path(__file__).parents[0] / "models" / "pneumonia_model.pth"
CLASS_NAMES = ["normal", "pneumonia"]  # index 0 -> normal, 1 -> pneumonia
IMG_SIZE = 64
print(MODEL_PATH)
# ---------- Model def (must match training) ----------
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


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(CLASS_NAMES))
    #state = torch.load(MODEL_PATH, map_location=device)
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


# ---------- UI ----------
st.markdown(
    """
    <style>
    .main-title {text-align:center; font-size:2.2rem; font-weight:700; margin-bottom:0;}
    .subtitle {text-align:center; color:#6b7280; margin-top:0; margin-bottom:1.5rem;}
    .result-box {padding:1.5rem; border-radius:12px; text-align:center; margin-top:1rem;}
    .positive {background:#fee2e2; border:1px solid #fca5a5;}
    .negative {background:#dcfce7; border:1px solid #86efac;}
    .result-label {font-size:1.6rem; font-weight:700;}
    .confidence {color:#4b5563; font-size:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🫁 Pneumonia Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a chest X-ray to check for signs of pneumonia</p>', unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"Model file not found at `{MODEL_PATH}`. Place `pneumonia_model.pth` in the `models/` folder next to this app.")
    st.stop()

model, device = load_model()

with st.sidebar:
    st.header("About")
    st.write("CNN (TinyVGG) trained to classify chest X-rays as **Normal** or **Pneumonia**.")
    st.write(f"Device: `{device}`")
    st.divider()
    st.caption("For educational use only — not a medical diagnosis.")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with col1:
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with st.spinner("Analyzing..."):
        label, confidence, probs = predict(model, device, image)

    is_pneumonia = label == "pneumonia"
    verdict = "Pneumonia Detected" if is_pneumonia else "No Pneumonia Detected"
    css_class = "positive" if is_pneumonia else "negative"
    icon = "⚠️" if is_pneumonia else "✅"

    with col2:
        st.markdown(
            f"""
            <div class="result-box {css_class}">
                <div class="result-label">{icon} {verdict}</div>
                <div class="confidence">Confidence: {confidence*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.progress(float(probs[CLASS_NAMES.index("pneumonia")]), text="Pneumonia probability")
        st.caption(f"Normal: {probs[0]*100:.1f}% | Pneumonia: {probs[1]*100:.1f}%")
else:
    st.info("Upload an X-ray image to get a prediction (JPG or PNG).")
