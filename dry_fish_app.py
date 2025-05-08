import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ultralytics import YOLO

# ----------------------------- CONFIG -----------------------------
st.set_page_config(page_title="Dry Fish Classify & Detect", layout="wide")
st.title("🐟 Dry Fish Detection & Classification with Explainable AI")

# ----------------------------- SIDEBAR -----------------------------
mode = st.sidebar.radio("Choose Task", ["Classification", "Detection"])

# Shared class names
class_names = [
    "Corica soborna", "Jamuna ailia", "Clupeidae", "Shrimp", "Chepa",
    "Chela", "Swamp barb", "Silond catfish", "Pale Carplet", "Bombay Duck", "Four-finger threadfin"
]

# ----------------------------- CLASSIFICATION -----------------------------
if mode == "Classification":
    st.header("📊 Dry Fish Classification with Grad-CAM Variants")

    @st.cache_resource
    def load_cls_model():
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
        model.load_state_dict(torch.load("mobilenet_v2.pth", map_location="cpu"))
        model.eval()
        return model.to("cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        input_tensor = transform(image).unsqueeze(0)

        model = load_cls_model()
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax().item()

        st.success(f"Predicted Class: {class_names[pred_class]}")

        # GradCAM Visualizations
        target_layers = [model.features[-1]]
        target = [ClassifierOutputTarget(pred_class)]

        cams = {
            "Grad-CAM": GradCAM(model, target_layers),
            "Grad-CAM++": GradCAMPlusPlus(model, target_layers),
            "EigenCAM": EigenCAM(model, target_layers),
        }

        st.subheader("🧠 XAI Visualizations")
        cols = st.columns(4)
        cams_results = [image_np]
        for method in cams:
            heatmap = cams[method](input_tensor=input_tensor, targets=target)[0]
            result = show_cam_on_image(image_np, cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0])), use_rgb=True)
            cams_results.append(result)

        captions = [
            "**Original**",
            "**Grad-CAM**",
            "**Grad-CAM++**",
            "**EigenCAM**"
        ]

        for i, col in enumerate(cols):
            with col:
                st.image(cv2.resize(cams_results[i], (300, 300)))
                st.markdown(captions[i])

# ----------------------------- DETECTION -----------------------------
elif mode == "Detection":
    st.header("📦 Dry Fish Detection using YOLOv Models + EigenCAM")

    model_options = {
        "YOLOv9": "yolov9.pt",
        "YOLOv10": "yolov10.pt",
        "YOLOv11": "yolov11.pt",
        "YOLOv12": "yolov12.pt"
    }
    selected_model = st.selectbox("Choose YOLO Model", list(model_options.keys()))
    model_path = model_options[selected_model]

    @st.cache_resource
    def load_det_model(path):
        model = YOLO(path)
        model.names = class_names
        return model

    model = load_det_model(model_path)

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="detect")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        if st.button("🔍 Detect Dry Fish"):
            with st.spinner("Detecting..."):
                results = model(image_np)[0]
                annotated = image_np.copy()
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{class_names[cls]} ({conf:.2f})"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                st.image(annotated, caption="Detected Image", use_column_width=True)

                st.subheader("🧠 EigenCAM on Full Image")
                resized = cv2.resize(image_np, (640, 640))
                input_tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                norm_image = np.float32(resized) / 255.0
                cam_layer = [model.model.model[-2]]
                eigen_cam = EigenCAM(model.model, cam_layer)
                grayscale_cam = eigen_cam(input_tensor=input_tensor, eigen_smooth=True)[0]
                cam_image = show_cam_on_image(norm_image, grayscale_cam, use_rgb=True)

                st.image(cam_image, caption="EigenCAM", use_column_width=True)

# ----------------------------- FOOTER -----------------------------
st.markdown("""
<hr style='border: 1px solid gray;'>
<div style='text-align: center;'>© 2025 Dry Fish XAI System | Developed by Md Rifat Ahmmad Rashid</div>
""", unsafe_allow_html=True)
