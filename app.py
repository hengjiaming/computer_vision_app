import os
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_v2_s

from utilities.camera_utils import extract_camera_details

# Predefined focal lengths for devices
device_focal_lengths = {
    "iPhone 15": 1480,
    "iPhone 14": 1450,
    "Samsung Galaxy S23": 1350,
    "Google Pixel 7": 1420,
    "Other": None,  # Placeholder for unspecified devices
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Using device: {device}")

# Initialize session state variables
if "scaling_factor" not in st.session_state:
    st.session_state["scaling_factor"] = None

if "depth_estimated" not in st.session_state:
    # Tracks depth estimation completion
    st.session_state["depth_estimated"] = False

if "nutrition_estimated" not in st.session_state:
    # Tracks nutrition estimation access
    st.session_state["nutrition_estimated"] = False


class EfficientNetBase(nn.Module):
    def __init__(self):
        super(EfficientNetBase, self).__init__()
        self.base_model = efficientnet_v2_s(weights=None)
        self.base_model.classifier = nn.Identity()  # Remove the classification head

    def forward(self, x):
        return self.base_model(x)

# Define the multi-task model class


class MultiTaskModel(nn.Module):
    def __init__(self, base_model):
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model
        in_features = 1280  # Assuming EfficientNet feature output size

        # Define separate branches for each task
        self.protein_branch = self._create_branch(in_features)
        self.fat_branch = self._create_branch(in_features)
        self.carbs_branch = self._create_branch(in_features)
        self.mass_branch = self._create_branch(in_features)

    def _create_branch(self, in_features):
        return nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.base_model(x)
        protein = self.protein_branch(x)
        fat = self.fat_branch(x)
        carbs = self.carbs_branch(x)
        mass = self.mass_branch(x)
        return {
            'protein': protein,
            'fat': fat,
            'carbs': carbs,
            'mass': mass
        }


@st.cache_resource
def load_depth_model():
    model = torch.hub.load(
        'yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    model = model.to(device).float()
    return model.to(device)


@st.cache_resource
def load_custom_model(pt_file_path):
    base_model = EfficientNetBase()
    model = MultiTaskModel(base_model)

    # Load the saved state_dict
    state_dict = torch.load(pt_file_path, map_location=device)
    model.load_state_dict(state_dict)  # Load weights into the model

    # Set the model to evaluation mode
    model.eval()
    model = model.to(device)
    return model


# Load custom model for nutritional estimation or preprocessing
pt_file_path = "/content/drive/MyDrive/Colab_Notebooks/computer_vision_app/models/final_multi_task_model.pth"
custom_model = load_custom_model(pt_file_path)

# App title and description
st.title("Nutritional and Depth Estimate App")
st.markdown("""
Welcome to the Nutritional and Depth Estimate App! 🎉  
Upload a photo of your meal, and our AI-powered model will provide nutritional estimates and depth estimation.
""")

# Sidebar for navigation or additional info
st.sidebar.title("About")
st.sidebar.info(
    "This app uses computer vision to analyze meals, estimate their nutritional values, and provide depth estimation."
)
# Initialize session state variables
if "depth_estimated" not in st.session_state:
    # To track depth estimation completion
    st.session_state["depth_estimated"] = False

if "nutrition_estimated" not in st.session_state:
    # To track nutrition estimation access
    st.session_state["nutrition_estimated"] = False

# File upload section
st.subheader("Upload a Meal Photo")
uploaded_file = st.file_uploader(
    "Upload an image file (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    unaltered_image = Image.open(uploaded_file)
    image = Image.open(uploaded_file).convert("RGB")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Depth Estimation Section
    st.subheader("Depth Estimation")
    camera_details = extract_camera_details(unaltered_image)
    focal_length = None
    width, height = image.size  # Default to the actual uploaded image dimensions
    if not st.session_state["depth_estimated"]:
        if st.button("Estimate Depth"):
            # Perform depth estimation
            st.write("Processing depth estimation... Please wait.")

            st.info(f"Image Size: {width} x {height} pixels")

            # Handle focal length extraction or fallback
            if camera_details and "Focal Length" in camera_details:
                focal_length = camera_details["Focal Length"]
                st.success(f"Extracted focal length: {focal_length} pixels")
            else:
                st.warning(
                    "Camera details (like focal length) could not be extracted from this image. "
                    "Please select your device from the dropdown below."
                )
                selected_device = st.selectbox(
                    "Select your device:", options=list(device_focal_lengths.keys())
                )
                focal_length = device_focal_lengths[selected_device]
                if focal_length is None:
                    st.warning(
                        "Proceeding without focal length may affect accuracy.")
                else:
                    st.success(f"Using focal length for {
                               selected_device}: {focal_length} pixels")

            depth_model = load_depth_model()
            with torch.no_grad():
                input_dict = {'input': input_tensor}
                pred_depth, confidence, output_dict = depth_model.inference(
                    input_dict)

            # Post-process depth map
            sensor_width = 7.6  # Example value for iPhone 13 Pro
            focal_length = focal_length / sensor_width * width if focal_length else 1.0
            canonical_to_real_scale = focal_length / 1000.0 if focal_length else 1.0
            pred_depth = pred_depth * canonical_to_real_scale
            pred_depth = torch.clamp(pred_depth, 0, 300)
            depth_map = pred_depth.squeeze().cpu().numpy()

            # Display center depth value
            center_x, center_y = depth_map.shape[1] // 2, depth_map.shape[0] // 2
            center_depth = depth_map[center_y, center_x]
            scaling_factor = center_depth / 0.4
            st.session_state["scaling_factor"] = scaling_factor
            st.write(f"Depth at center ({center_x}, {center_y}): {
                     center_depth:.2f} meters. Scaling Factor (Depth of Center / 0.4): {scaling_factor:.2f}")

            # Save depth estimation as completed
            st.session_state["depth_estimated"] = True  # Update session state

            # Display depth map visualization
            depth_map_vis = (depth_map - depth_map.min()) / \
                (depth_map.max() - depth_map.min())
            st.image(depth_map_vis, caption="Depth Map (Normalized)",
                     use_container_width=True, clamp=True)

    # Nutritional Estimation Section (Locked behind depth estimation)
    if st.session_state["depth_estimated"]:
        st.subheader("Nutritional Estimation")
        if st.button("Estimate Nutrition"):
            # Perform nutritional estimation
            st.write("Processing nutritional estimation... Please wait.")
            scaler_directory = "/content/drive/MyDrive/Colab_Notebooks/nutri_estimate/nutrition_model/scalers"

            custom_model = load_custom_model(pt_file_path)

            # Scale target columns
            scalers = {}
            scaler_files = [
                "calorie_scaler.save", "mass_scaler.save", "carb_scaler.save",
                "protein_scaler.save", "fat_scaler.save", "total_calories_scaler.save",
                "total_mass_scaler.save", "total_carb_scaler.save",
                "total_protein_scaler.save", "total_fat_scaler.save"
            ]

            for scaler_file in scaler_files:
                scaler_name = scaler_file.replace("_scaler.save", "")
                scaler_path = os.path.join(scaler_directory, scaler_file)
                scalers[scaler_name] = joblib.load(scaler_path)
            # Save scalers for inverse transformation later

            with torch.no_grad():
                outputs = custom_model(input_tensor)

            results = {}
            for key in outputs:
                # Adjust key naming to match the scalers dictionary (carbs don't match carb)
                scaler_key = f"total_{
                    key[:-1]}" if key == "carbs" else f"total_{key}"
                if scaler_key in scalers:
                    predicted_value = scalers[scaler_key].inverse_transform(
                        outputs[key].cpu().numpy())[0][0]
                    results[key] = max(0, predicted_value)
                else:
                    print(f"Warning: Scaler for key '{
                          scaler_key}' not found. Skipping inverse transformation.")
                    # use raw output as fallback(clamped)
                    results[key] = max(0, outputs[key].cpu().numpy()[0][0])

            # Display the predictions
            st.subheader("Nutritional Estimation Results")
            calories = results["protein"] * 4 + \
                results["carbs"] * 4 + results["fat"] * 9
            results["calories"] = calories
            for key, value in results.items():
                value = value * st.session_state["scaling_factor"]/3
                if key == "calories":
                    st.write(f"{key.capitalize()}: {value:.2f} calories")
                elif key != "mass":
                    st.write(f"{key.capitalize()}: {value:.2f}g")

else:
    st.info("Please upload an image to begin.")
