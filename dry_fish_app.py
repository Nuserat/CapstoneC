import streamlit as st
import torch
torch.classes.__path__ = []
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

# Ensure set_page_config is the first Streamlit command
st.set_page_config(page_title="Dry Fish Classification and XAI", layout="wide")

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

# Load the pretrained MobileNetV2 model
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

# Header Section
st.markdown(
    """
    <div style='text-align: center; font-size: 36px; font-weight: bold;'>Dry Fish Classification and Explainable AI</div>
    <hr style='border: 1px solid black;'>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='text-align: center; font-size: 32px; font-weight: bold;'>Explainable AI for Dry Fish Classification</div>
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
    #captions = ["**Original Image**", "**Grad-CAM**", "**Grad-CAM++**", "**Eigen-CAM**"]
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

# Footer Section
st.markdown(
    """
    <hr style='border: 1px solid black;'>
    <div style='text-align: center; font-size: 16px; color: black;'>
        © 2025 Dry Fish Classification System | Developed by Md Rifat Ahmmad Rashid, Associate Professor, EWU Bangladesh
    </div>
    """,
    unsafe_allow_html=True,
)
