import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ultralytics import YOLO
import os

st.set_page_config(page_title="Dry Fish Classification & Detection", layout="wide")

# Styling
st.markdown("""
<style>
body {background-color: black; color: white;}
.stApp {background-color: lightcyan; color: black;}
</style>
""", unsafe_allow_html=True)

# ----- Tabs -----
tab1, tab2 = st.tabs(["📌 Classification", "📍 Detection"])

# ======== Classification Tab ========
with tab1:
    st.markdown("<h2 style='text-align: center;'>Dry Fish Classification with XAI</h2>", unsafe_allow_html=True)

    @st.cache_resource
    def load_classification_model():
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = torch.nn.Linear(model.last_channel, 11)
        model.load_state_dict(torch.load("mobilenet_v2.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        model.eval()
        return model.to("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class_names = [
        "Corica soborna(কাচকি মাছ)", "Jamuna ailia(কাজরী মাছ)", "Clupeidae(চাপিলা মাছ)", "Shrimp(চিংড়ি মাছ)",
        "Chepa(চ্যাপা মাছ)", "Chela(চ্যালা মাছ)", "Swamp barb(পুঁটি মাছ)", "Silond catfish(ফ্যাসা মাছ)",
        "Pale Carplet(মলা মাছ)", "Bombay Duck(লইট্যা মাছ)", "Four-finger threadfin(লাইক্ষা মাছ)"
    ]

    uploaded_file = st.sidebar.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"], key="clf")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        original_image_np = np.array(image).astype(np.float32) / 255.0
        transformed_image = transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        model = load_classification_model()

        with torch.no_grad():
            outputs = model(transformed_image)
            predicted_class = outputs.argmax().item()

        st.sidebar.success(f"Predicted: {class_names[predicted_class]} (ID: {predicted_class})")

        target_layers = [model.features[-1]]
        target = [ClassifierOutputTarget(predicted_class)]

        gradcam = GradCAM(model, target_layers)
        gradcam_result = show_cam_on_image(original_image_np,
                                           cv2.resize(gradcam(transformed_image, target)[0],
                                                      (original_image_np.shape[1], original_image_np.shape[0])),
                                           use_rgb=True)

        gradcam_pp = GradCAMPlusPlus(model, target_layers)
        gradcam_pp_result = show_cam_on_image(original_image_np,
                                              cv2.resize(gradcam_pp(transformed_image, target)[0],
                                                         (original_image_np.shape[1], original_image_np.shape[0])),
                                              use_rgb=True)

        eigen_cam = EigenCAM(model, target_layers)
        eigen_result = show_cam_on_image(original_image_np,
                                         cv2.resize(eigen_cam(transformed_image, target)[0],
                                                    (original_image_np.shape[1], original_image_np.shape[0])),
                                         use_rgb=True)

        cols = st.columns(4)
        imgs = [np.array(image), gradcam_result, gradcam_pp_result, eigen_result]
        captions = [
            "**Original Image**",
            "**Grad-CAM**",
            "**Grad-CAM++**",
            "**Eigen-CAM**"
        ]
        for i, col in enumerate(cols):
            with col:
                st.image(cv2.resize(imgs[i], (400, 400)))
                st.markdown(f"<div style='text-align:center'>{captions[i]}</div>", unsafe_allow_html=True)
    else:
        st.info("Upload an image to classify dry fish.")

# ======== Detection Tab ========
with tab2:
    st.markdown("<h2 style='text-align: center;'>YOLO-based Dry Fish Detection with EigenCAM</h2>", unsafe_allow_html=True)

    model_options = {
        "YOLOv9": "yolov9.pt",
        "YOLOv10": "yolov10.pt",
        "YOLOv11": "yolov11.pt",
        "YOLOv12": "yolov12.pt"
    }
    selected_model = st.sidebar.selectbox("Choose YOLO model", list(model_options.keys()), key="det")
    model_path = model_options[selected_model]

    @st.cache_resource
    def load_yolo(path):
        return YOLO(path)

    model = load_yolo(model_path)
    st.success(f"YOLO model {model_path} loaded.")

    uploaded_det_file = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"], key="det2")

    if uploaded_det_file:
        image = Image.open(uploaded_det_file).convert("RGB")
        image_np = np.array(image)

        if st.button("🔍 Run Detection"):
            with st.spinner("Detecting..."):
                results = model(image_np)
                det_img = image_np.copy()

                if results and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        label = f"Dry Fish: {conf:.2f}"
                        cv2.rectangle(det_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(det_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    st.success(f"{len(results[0].boxes)} dry fish detected.")
                else:
                    st.warning("No dry fish detected.")

                st.image(det_img, caption="Detection Result", use_column_width=True)

                st.subheader("📊 EigenCAM on YOLO")
                img_resized = cv2.resize(image_np, (640, 640))
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                norm_img = np.float32(img_resized) / 255.0

                cam = EigenCAM(model.model, [model.model.model[-2]])
                grayscale_cam = cam(input_tensor=img_tensor, eigen_smooth=True)[0]
                cam_img = show_cam_on_image(norm_img, grayscale_cam, use_rgb=True)

                st.image(cam_img, caption="EigenCAM Visualization", use_column_width=True)
    else:
        st.info("Upload an image to detect dry fish.")

# Footer
st.markdown("""
<hr>
<div style='text-align: center; font-size: 14px;'>
© 2025 Dry Fish AI | Developed by Md Rifat Ahmmad Rashid, Associate Professor, EWU Bangladesh
</div>
""", unsafe_allow_html=True)
