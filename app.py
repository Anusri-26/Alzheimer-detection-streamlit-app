import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import cv2
import torch.nn.functional as F
import io
from datetime import datetime
import gdown
import os

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Alzheimer Detection", layout="wide")

st.title("🧠 AI-Assisted Alzheimer Detection System")
st.caption("⚠️ Screening support tool — not a medical diagnosis")

# ---------------- LOAD LABELS ----------------
with open("labels.json", "r") as f:
    classes = json.load(f)

# ---------------- MODEL ----------------
class HybridModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = models.mobilenet_v2(weights=None).features
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1280, nhead=8, dim_feedforward=2048, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean([-1, -2])
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

@st.cache_resource
def load_model():
    MODEL_PATH = "model.pth"

    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1uPC8rgv2pYCLC_12WfxvYx-RgoU7Tk7W"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = HybridModel(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- GRAD-CAM ----------------
def generate_gradcam(model, image_tensor):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    target_layer = model.cnn[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()

    cam = F.relu(cam)
    cam = cam.detach().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    return cam

# ---------------- NAVIGATION ----------------
st.sidebar.markdown("## 🏥 Navigation Panel")
st.sidebar.markdown("---")
page = st.sidebar.selectbox("Select Section", ["🏠 Home", "🔍 MRI Analysis", "ℹ️ About"])

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.header("Overview")

    st.write("""
    Alzheimer’s disease is a progressive neurological disorder affecting memory and cognition.
    Early detection helps in better treatment planning.
    """)

    st.markdown("---")

    st.subheader("⚙️ How It Works")
    st.write("""
    1. Upload MRI scan  
    2. AI analyzes patterns  
    3. Generates prediction & explanation  
    """)

    st.markdown("---")

    st.subheader("🧾 Symptoms")
    st.markdown("""
    - Memory loss  
    - Confusion  
    - Difficulty solving problems  
    - Mood changes  
    """)

    st.warning("For screening support only — not a medical diagnosis.")

# ---------------- PREDICTION ----------------
elif page == "🔍 MRI Analysis":

    st.header("🔍 MRI Analysis & Clinical Report")

    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original MRI", use_container_width=True)

        img = transform(image).unsqueeze(0)

        if st.button("Analyze MRI"):
            with st.spinner("Generating AI Report..."):

                outputs = model(img)
                probs = torch.softmax(outputs[0], dim=0)
                confidence, pred = torch.max(probs, 0)

                predicted_class = classes[pred.item()]
                confidence_score = confidence.item() * 100

                cam = generate_gradcam(model, img)
                cam = cv2.resize(cam, (224,224))

                heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                overlay = np.array(image.resize((224,224))) * 0.6 + heatmap * 0.4
                overlay = overlay.astype(np.uint8)

            with col2:
                st.success(f"Prediction: {predicted_class}")
                st.info(f"Confidence: {confidence_score:.2f}%")

                if "Moderate" in predicted_class:
                    st.error("⚠️ High Risk")
                elif "Mild" in predicted_class:
                    st.warning("⚠️ Early Stage")
                else:
                    st.success("✅ No significant signs")

                st.subheader("📊 Probability Analysis")
                for i, cls in enumerate(classes):
                    if i == pred.item():
                        st.success(f"{cls}: {probs[i]*100:.2f}% (Predicted)")
                    else:
                        st.write(f"{cls}: {probs[i]*100:.2f}%")

            st.subheader("🔥 Grad-CAM Heatmap")
            st.image(overlay)

            # ---------------- PDF ----------------
            def create_pdf():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()
                story = []

                story.append(Paragraph("AI-Assisted Alzheimer Detection Report", styles["Title"]))
                story.append(Paragraph("<br/>", styles["Normal"]))

                scan_id = "MRI-" + datetime.now().strftime("%Y%m%d%H%M%S")
                story.append(Paragraph(f"Scan ID: {scan_id}", styles["Normal"]))
                story.append(Paragraph(f"Prediction: {predicted_class}", styles["Normal"]))
                story.append(Paragraph(f"Confidence: {confidence_score:.2f}%", styles["Normal"]))
                story.append(Paragraph("<br/>", styles["Normal"]))

                story.append(Paragraph("Clinical Interpretation", styles["Heading2"]))
                story.append(Paragraph(
                    f"The MRI scan shows patterns consistent with {predicted_class}.",
                    styles["Normal"]
                ))

                story.append(Paragraph("<br/>", styles["Normal"]))

                story.append(Paragraph("Probability Analysis", styles["Heading2"]))
                for i, cls in enumerate(classes):
                    story.append(Paragraph(f"{cls}: {probs[i]*100:.2f}%", styles["Normal"]))

                story.append(Paragraph("<br/>", styles["Normal"]))

                image.save("original.png")
                Image.fromarray(overlay).save("heatmap.png")

                story.append(Paragraph("Original MRI", styles["Heading3"]))
                story.append(RLImage("original.png", width=250, height=250))

                story.append(Paragraph("<br/>", styles["Normal"]))

                story.append(Paragraph("Grad-CAM Heatmap", styles["Heading3"]))
                story.append(RLImage("heatmap.png", width=250, height=250))

                story.append(Paragraph("<br/>", styles["Normal"]))

                story.append(Paragraph("Recommendation", styles["Heading2"]))
                story.append(Paragraph(
                    "This AI analysis is for screening purposes only. Consult a medical professional.",
                    styles["Normal"]
                ))

                doc.build(story)
                buffer.seek(0)
                return buffer

            pdf = create_pdf()

            st.download_button(
                label="📄 Download Full Report (PDF)",
                data=pdf,
                file_name="Alzheimer_Report.pdf",
                mime="application/pdf"
            )

# ---------------- ABOUT ----------------
elif page == "ℹ️ About":

    st.header("ℹ️ About the Model")

    st.markdown("""
    Hybrid Model:
    - MobileNet → Feature extraction  
    - Transformer → Global learning  

    Includes Grad-CAM for explainability.
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("AI-Assisted Alzheimer Detection System | Major Project")
