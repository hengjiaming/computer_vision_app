import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# app title and description
st.title("Nutritional Estimate App")
st.markdown("""
Welcome to the Nutritional Estimate App! 🎉  
Upload a photo of your meal, and our AI-powered model will provide nutritional estimates in seconds.  
""")

# sidebar for navigation or additional info
st.sidebar.title("About")
st.sidebar.info("This app uses computer vision to analyze meals and estimate their nutritional values.")

# file upload section
st.subheader("Upload a Meal Photo")
uploaded_file = st.file_uploader("Upload an image file (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # call API
    st.subheader("Nutritional Analysis")
    if st.button("Analyze Image"):
        st.write("Processing... Please wait.")
        
        # placeholder API call function to our model
        api_url = "https://api-endpoint.com/predict"  # replace when it is up
        files = {"file": uploaded_file.getvalue()}
        
        try:
            response = requests.post(api_url, files=files)
            response.raise_for_status()
            result = response.json()

            # Display results
            st.success("Analysis Complete!")
            st.image(BytesIO(response.content), caption="Analysis Result", use_column_width=True)

            # Display nutritional information
            st.subheader("Nutritional Estimates")
            for key, value in result["nutrition"].items():
                st.write(f"**{key.capitalize()}**: {value}")
        
        except requests.exceptions.RequestException as e:
            st.error("There was an error processing the image.")
            st.error(f"Error: {e}")

else:
    st.info("Please upload an image to begin.")