import streamlit as st
import torch
torch.classes.__path__ = []
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ultralytics import YOLO
import matplotlib.pyplot as plt
import sys
sys.path.append("./YOLO-V8-CAM")  

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

    # Set background color for right side to light gray and text color to black
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

    @st.cache_resource
    def load_model():
        num_classes = 11  # Updated with the number of dry fish classes
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # Modify the classifier
        model.load_state_dict(torch.load("mobilenet_v2.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        model.eval()
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        return model

    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Class labels
    class_names = [
        "Corica soborna(কাচকি মাছ)", "Jamuna ailia(কাজরী মাছ)", "Clupeidae(চাপিলা মাছ)", "Shrimp(চিংড়ি মাছ)", "Chepa(চ্যাপা মাছ)",
        "Chela(চ্যালা মাছ)", "Swamp barb(পুঁটি মাছ)", "Silond catfish(ফ্যাসা মাছ)", "Pale Carplet(মলা মাছ)", "Bombay Duck(লইট্যা মাছ)", "Four-finger threadfin(লাইক্ষা মাছ)"
    ]

    st.markdown(
    """
    <div style='text-align: center; font-size: 32px; font-weight: bold;'>📊 Explainable AI for Dry Fish Classification</div>
    <div style='text-align: center; font-size: 20px; font-weight: normal; margin-top: -10px;'>Using Grad-CAM, Grad-CAM++, and Eigen-CAM</div>
    """,
    unsafe_allow_html=True,
    )

    st.sidebar.header("Upload Your Image")
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        original_image_np = np.array(image).astype(np.float32) / 255.0
        transformed_image = transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

        model = load_model()
        
        with torch.no_grad():
            outputs = model(transformed_image)
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
        gradcam_heatmap = gradcam(input_tensor=transformed_image, targets=target)[0]
        gradcam_result = show_cam_on_image(original_image_np, cv2.resize(gradcam_heatmap, (original_image_np.shape[1], original_image_np.shape[0])), use_rgb=True)

        gradcam_plus_plus = GradCAMPlusPlus(model=model, target_layers=target_layers)
        gradcam_plus_plus_heatmap = gradcam_plus_plus(input_tensor=transformed_image, targets=target)[0]
        gradcam_plus_plus_result = show_cam_on_image(original_image_np, cv2.resize(gradcam_plus_plus_heatmap, (original_image_np.shape[1], original_image_np.shape[0])), use_rgb=True)

        eigen_cam = EigenCAM(model=model, target_layers=target_layers)
        eigen_cam_heatmap = eigen_cam(input_tensor=transformed_image, targets=target)[0]
        eigen_cam_result = show_cam_on_image(original_image_np, cv2.resize(eigen_cam_heatmap, (original_image_np.shape[1], original_image_np.shape[0])), use_rgb=True)

        st.markdown(
            """
            <div style='text-align: center; font-size: 20px; font-weight: bold; margin-top: 30px;'>
                Visualization Results
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns(4, gap="medium")
        grid_images = [np.array(image), gradcam_result, gradcam_plus_plus_result, eigen_cam_result]
        captions = [
            "**Original Image**",
            "**Grad-CAM**: Highlights important regions by computing the gradient of the class score with respect to the feature maps.",
            "**Grad-CAM++**: An improved version of Grad-CAM that provides better localization by weighting the gradients differently.",
            "**Eigen-CAM**: Utilizes principal component analysis on the feature maps to identify significant regions without relying on gradients."
        ]   
    
        for i, col in enumerate(cols):
            with col:
                st.image(cv2.resize(grid_images[i], (400, 400)), use_container_width=False)
                st.markdown(f"<div style='text-align: center; font-size: 18px; font-weight: bold; color: black;'>{captions[i]}</div>", unsafe_allow_html=True)
    else:
        st.info("Please upload an image to proceed.")

# ----------------------------- DETECTION -----------------------------
elif mode == "Detection":
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
    st.header("📦 Dry Fish Detection using YOLOv Models + EigenCAM")

    @st.cache_resource
    def load_model():
        try:
            from ultralytics import YOLO
            model = YOLO("yolov10.pt")
            st.success("✅ Model `yolov10.pt` loaded successfully.")
            return model
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            return None

    model = load_model()
    if model is None:
        st.stop()

    class_names = [
        "Corica soborna", "Jamuna ailia", "Clupeidae", "Shrimp", "Chepa",
        "Chela", "Swamp barb", "Silond catfish", "Pale Carplet", "Bombay Duck", "Four-finger threadfi"
    ]
    def draw_boxes(image, results):
        annotated_img = image.copy()
        class_names = model.names  # ✅ Pull correct labels from model

        if results and results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                label = f"{class_names[class_id]}: {conf:.2f}"  # ✅ Show actual class

                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return annotated_img


# Image upload section
    st.subheader("📷 Upload an Image to Detect Dry Fish")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")  # PIL image
        image_np = np.array(image).astype(np.uint8)  # Ensure correct format

        if st.button("🔍 Detect Dry Fish"):
            with st.spinner("Processing..."):
                try:
                    results = model(image, imgsz=(600, 800), conf=0.5)
                    result_image = draw_boxes(image_np, results[0])

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        st.image(image, use_column_width=True)
                    with col2:
                        st.subheader("Detection Result")
                        st.image(result_image, use_column_width=True)

                    count = len(results[0].boxes)
                    if count > 0:
                        st.success(f"Detected {count} Dry Fish instance(s).")
                    else:
                        st.info("No Dry Fish detected.")
                except Exception as e:
                    st.error(f"Error during detection: {e}")


                # ============ 🔍 EigenCAM XAI Visualization ============

    @st.cache_resource
    def load_raw_yolo_model():
        model = YOLO("yolo10n.pt")  # Adjust path if needed
        model.cpu()
        return model

    raw_model = load_raw_yolo_model()

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.resize(image, (640, 640))
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_input = np.float32(rgb_img) / 255.0

        # Select target layer
        target_layers = [raw_model.model.model[-2]]  # Last conv layer

        # Run EigenCAM
        with st.spinner("Generating EigenCAM..."):
            cam = EigenCAM(raw_model, target_layers, task='od')
            grayscale_cam = cam(rgb_img)[0, :, :]
            cam_image = show_cam_on_image(img_input, grayscale_cam, use_rgb=True)

        # Show outputs
        st.subheader("Original Image")
        st.image(rgb_img, channels="RGB")

        st.subheader("EigenCAM Overlay")
        st.image(cam_image, channels="RGB")

        st.subheader("Side-by-Side View")
        combined = np.hstack((rgb_img, cam_image))
        st.image(combined, channels="RGB")

# About section
    with st.expander("About this App"):
        st.write("""
    ### Dry Fish Detection App (Image Upload + XAI)
    This app uses YOLOv Models trained for detecting dry fish from images.

    #### Features:
    - Upload an image for dry fish detection
    - Bounding boxes with confidence scores
    - XAI visualization using EigenCAM

    #### How it works:
    The model processes the uploaded image and detects regions containing dry fish. CAMs show model focus areas.

    #### Use cases:
    - Quality control in seafood processing
    - Marine life classification
    - Research and monitoring in fisheries
    """)
