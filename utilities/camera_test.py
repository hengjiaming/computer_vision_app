import os
from io import BytesIO

# import joblib
import numpy as np
import pandas as pd
import requests
# import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision.transforms as transforms
from PIL import Image
from PIL.ExifTags import TAGS
# from sklearn.preprocessing import MinMaxScaler
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_v2_s


# def extract_camera_details(image_path):
#     try:
#         # Open the image file
#         image = Image.open(image_path)

#         # Get EXIF data
#         exif_data = image._getexif()
#         if not exif_data:
#             return {"error": "No EXIF data found in the image."}

#         # Map EXIF tags to their names
#         exif_tags = {TAGS.get(tag, tag): value for tag,
#                      value in exif_data.items()}

#         # Extract relevant details
#         details = {
#             "Image Size": image.size,
#             "Focal Length": exif_tags.get("FocalLength"),
#             "Camera Model": exif_tags.get("Model"),
#             "Camera Make": exif_tags.get("Make"),
#             "Exposure Time": exif_tags.get("ExposureTime"),
#             "ISO Speed": exif_tags.get("ISOSpeedRatings"),
#         }

#         # Format focal length if it's a tuple
#         if details["Focal Length"] and isinstance(details["Focal Length"], tuple):
#             details["Focal Length"] = float(details["Focal Length"][0]) / float(
#                 details["Focal Length"][1]
#             )
#         width, height = details["Image Size"][0], details["Image Size"][1]
#         print(width, height)
#         return details
#     except AttributeError as e:
#         return {"error": f"Image does not contain EXIF data: {e}"}
#     except Exception as e:
#         return {"error": f"Error extracting camera details: {e}"}


# Example usage
# camera_details = extract_camera_details(image_link)
# print(camera_details)

image_link = "../images/has_exif.jpeg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


pt_file_path = "../models/final_multi_task_model.pth"


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


custom_model = load_custom_model(pt_file_path)


image = Image.open("../images/has_exif.jpeg").convert("RGB")

preprocess = transforms.Compose([
    transforms.ToTensor(),  # Converts [H, W, C] -> [C, H, W]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])  # Standard normalization
])

input_tensor = preprocess(image).unsqueeze(0).to(device)
with torch.no_grad():
    nutritional_output = custom_model(input_tensor)
