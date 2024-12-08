from io import BytesIO

import requests
import streamlit as st
from PIL import Image

from utilities.camera_utils import extract_camera_details



# Predefined focal lengths for devices
device_focal_lengths = {
    "iPhone 15": 1480,
    "iPhone 14": 1450,
    "Samsung Galaxy S23": 1350,
    "Google Pixel 7": 1420,
    "Other": None,  # Placeholder for unspecified devices
}

# App title and description
st.title("Nutritional Estimate App")
st.markdown("""
Welcome to the Nutritional Estimate App! 🎉  
Upload a photo of your meal, and our AI-powered model will provide nutritional estimates in seconds. 
            \n

Please ensure that the dish is **centered** in the image.

""")

# Sidebar for navigation or additional info
st.sidebar.title("About")
st.sidebar.info(
    "This app uses computer vision to analyze meals and estimate their nutritional values."
)

# File upload section
st.subheader("Upload a Meal Photo")
uploaded_file = st.file_uploader(
    "Upload an image file (JPEG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Call the utility to extract camera details
    camera_details = extract_camera_details(image)
    focal_length = None

    # Check if camera details were extracted
    if not camera_details or "Focal Length" not in camera_details:
        st.warning(
            "Camera details (like focal length) could not be extracted from this image. "
            "Please select your device from the dropdown below."
        )

        # Let the user select their device
        selected_device = st.selectbox(
            "Select your device:", options=list(device_focal_lengths.keys())
        )

        # Get the focal length from the selected device
        focal_length = device_focal_lengths[selected_device]

        if focal_length is None:
            st.warning(
                "You selected 'Other'. Please proceed with caution, as accuracy may be affected."
            )
        else:
            st.success(f"Using focal length for {
                       selected_device}: {focal_length} pixels")
    else:
        # Use extracted focal length
        focal_length = camera_details["Focal Length"]
        st.success(
            f"Camera details extracted successfully! Focal Length: {
                focal_length} pixels"
        )

    # Display camera details in the console
    print(camera_details)

    # Nutritional Analysis Section
    st.subheader("Nutritional Analysis")
    if st.button("Analyze Image"):
        st.write("Processing... Please wait.")

        # Placeholder API call function to your model
        api_url = "https://api-endpoint.com/predict"  # Replace with your API URL
        files = {"file": uploaded_file.getvalue()}

        try:
            response = requests.post(api_url, files=files)
            response.raise_for_status()
            result = response.json()

            # Display results
            st.success("Analysis Complete!")
            st.image(BytesIO(response.content),
                     caption="Analysis Result", use_column_width=True)

            # Display nutritional information
            st.subheader("Nutritional Estimates")
            for key, value in result["nutrition"].items():
                st.write(f"**{key.capitalize()}**: {value}")

        except requests.exceptions.RequestException as e:
            st.error("There was an error processing the image.")
            st.error(f"Error: {e}")

else:
    st.info("Please upload an image to begin.")
