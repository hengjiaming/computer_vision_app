# Nutritional Estimate App

This project allows users to upload meal images and get nutritional estimates using a Streamlit-based UI and our Computer Vision system.

---

## Features

-   **Upload Meal Photos**: Users can upload images of their meals.
-   **AI Analysis**: The app sends the uploaded image to a backend model for nutritional analysis.
-   **Nutritional Estimates**: Displays the analyzed image and provides a breakdown of key nutritional values.

---

## Setup Instructions

Follow these steps to set up the project on your local machine.

[Do note that your setup must be GPU enabled (e.g. cuda)]

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone <repository_url>
cd <repository_directory>
```

---

### 2. Set Up a Virtual Environment

Create a Python virtual environment to manage dependencies:

```bash
python -m venv .venv
```

Activate the virtual environment:

-   **Windows**:
    ```bash
    .venv\Scripts\activate
    ```
-   **macOS/Linux**:
    ```bash
    source .venv/bin/activate
    ```

Ensure your device is GPU enabled (e.g. cuda).

---

### 3. Install Dependencies

Install all required Python packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
pip install openmim
mim install mmengine
mim install mmcv
```

---

### 4. Run the Streamlit App

Launch the Streamlit app:

```bash
streamlit run app.py
```

-   This will open the app in your default web browser.
-   If it doesn’t open automatically, navigate to the URL shown in the terminal (e.g. `http://localhost:8501`).

---

---

## About

This app is built using:

-   [Streamlit](https://streamlit.io/) for the UI.
-   Python 3.8+ for development.
-   Backend API to handle AI-based nutritional analysis.
