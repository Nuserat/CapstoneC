import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ===================== Streamlit Page Config =====================
st.set_page_config(page_title="Dry Fish XAI System", layout="wide", page_icon="🐟")

st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stApp {
        background-color: lightcyan;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🐟 Dry Fish Detection & Classification using XAI")
task_option = st.sidebar.radio("Choose Task", ["Detection (YOLO + EigenCAM)", "Classification (MobileNet + GradCAMs)"])

# ===================== Shared Upload Section =====================
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ===================== Detection Task =====================
if task_option.startswith("Detection"):
    model_options = {
        "YOLOv9": "yolov9.pt",
        "YOLOv10": "yolov10.pt",
        "YOLOv11": "yolov11.pt",
        "YOLOv12": "yolov12.pt"
    }
    selected_model_name = st.sidebar.selectbox("Select YOLO Model", list(model_options.keys()))
    model_path = model_options[selected_model_name]

    @st.cache_resource
    def load_yolo_model(path):
        return YOLO(path)

    model = load_yolo_model(model_path)
    st.success(f"✅ YOLO Model `{model_path}` loaded successfully.")

    def draw_boxes(image, results, model):
        annotated_img = image.copy()
        names = model.names
        if results and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                fish_name = names.get(cls_idx, "unknown")
                label = f"{fish_name}: {conf:.2f}"
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return annotated_img

    if uploaded_file and st.button("🔍 Detect Dry Fish"):
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        with st.spinner("Running detection..."):
            results = model(image_np)
            result_image = draw_boxes(image_np, results[0], model)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("Detection Result")
                st.image(result_image, use_column_width=True)

            count = len(results[0].boxes)
            st.success(f"✅ Detected {count} Dry Fish instance(s)." if count > 0 else "No Dry Fish detected.")

            # ============ EigenCAM XAI Visualization ============
            st.subheader("📊 EigenCAM Visualization")
            img_resized = cv2.resize(image_np, (640, 640))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_norm = np.float32(img_resized) / 255.0

            target_layers = [model.model.model[-2]]
            cam = EigenCAM(model.model, target_layers)
            grayscale_cam = cam(input_tensor=img_tensor, eigen_smooth=True)[0, :, :]
            cam_image = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

            st.image(cam_image, caption="EigenCAM Attention Map", use_column_width=True)
            combined = np.hstack((cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), cam_image))
            st.image(combined, caption="Original + CAM", use_column_width=True)

# ===================== Classification Task =====================
elif task_option.startswith("Classification"):
    @st.cache_resource
    def load_classification_model():
        num_classes = 11
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        model.load_state_dict(torch.load("mobilenet_v2.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        model.eval()
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        return model

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class_names = [
        "Corica soborna(কাচকি মাছ)", "Jamuna ailia(কাজরী মাছ)", "Clupeidae(চাপিলা মাছ)", "Shrimp(চিংড়ি মাছ)", "Chepa(চ্যাপা মাছ)",
        "Chela(চ্যালা মাছ)", "Swamp barb(পুঁটি মাছ)", "Silond catfish(ফ্যাসা মাছ)", "Pale Carplet(মলা মাছ)", "Bombay Duck(লইট্যা মাছ)", "Four-finger threadfin(লাইক্ষা মাছ)"
    ]

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        original_np = np.array(image).astype(np.float32) / 255.0
        transformed = transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

        model = load_classification_model()
        with torch.no_grad():
            outputs = model(transformed)
            predicted_class = outputs.argmax().item()

        st.sidebar.markdown(
            f"""
            <div style='border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; text-align: center; background-color: lightgray; color: black;'>
                <h3 style='color: black;'>Prediction</h3>
                <p style='font-size: 18px; font-weight: bold; color: #4CAF50;'>{class_names[predicted_class]}</p>
                <p style='font-size: 14px; color: black;'>(Class ID: {predicted_class})</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        target_layers = [model.features[-1]]
        target = [ClassifierOutputTarget(predicted_class)]

        gradcam = GradCAM(model=model, target_layers=target_layers)
        gradcam_result = show_cam_on_image(original_np, cv2.resize(gradcam(input_tensor=transformed, targets=target)[0], (original_np.shape[1], original_np.shape[0])), use_rgb=True)

        gradcampp = GradCAMPlusPlus(model=model, target_layers=target_layers)
        gradcampp_result = show_cam_on_image(original_np, cv2.resize(gradcampp(input_tensor=transformed, targets=target)[0], (original_np.shape[1], original_np.shape[0])), use_rgb=True)

        eigencam = EigenCAM(model=model, target_layers=target_layers)
        eigencam_result = show_cam_on_image(original_np, cv2.resize(eigencam(input_tensor=transformed, targets=target)[0], (original_np.shape[1], original_np.shape[0])), use_rgb=True)

        st.markdown("<h3 style='text-align: center;'>Visualization Results</h3>", unsafe_allow_html=True)
        cols = st.columns(4)
        imgs = [np.array(image), gradcam_result, gradcampp_result, eigencam_result]
        captions = [
            "**Original Image**",
            "**Grad-CAM**: Highlights important regions by computing the gradient of the class score with respect to the feature maps.",
            "**Grad-CAM++**: Improved localization by weighting gradients more effectively.",
            "**Eigen-CAM**: Uses PCA on activations to visualize focus without needing gradients."
        ]

        for i, col in enumerate(cols):
            with col:
                st.image(cv2.resize(imgs[i], (400, 400)), use_container_width=False)
                st.markdown(f"<div style='text-align: center; font-size: 16px; color: black;'>{captions[i]}</div>", unsafe_allow_html=True)

# ===================== Footer =====================
st.markdown(
    """
    <hr style='border: 1px solid black;'>
    <div style='text-align: center; font-size: 14px; color: black;'>
        © 2025 Dry Fish XAI System | Developed by Md Rifat Ahmmad Rashid, Associate Professor, EWU Bangladesh
    </div>
    """,
    unsafe_allow_html=True,
)
