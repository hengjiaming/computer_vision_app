# Nutritional Estimate App

Welcome to the **Nutritional Estimate App**!  
This project allows users to upload meal images and get nutritional estimates using a Streamlit-based UI and an AI-powered backend.

---

## Features
- **Upload Meal Photos**: Users can upload images of their meals.
- **AI Analysis**: The app sends the uploaded image to a backend model for nutritional analysis.
- **Nutritional Estimates**: Displays the analyzed image and provides a breakdown of key nutritional values.

---

## Setup Instructions

Follow these steps to set up the project on your local machine.

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
- **Windows**:
  ```bash
  .venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```

---

### 3. Install Dependencies
Install all required Python packages from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

### 4. Run the Streamlit App
Launch the Streamlit app:
```bash
streamlit run app.py
```

- This will open the app in your default web browser.
- If it doesn’t open automatically, navigate to the URL shown in the terminal (usually `http://localhost:8501`).

---

### 5. Workflow for Development
1. Make sure your virtual environment is activated before making changes:
   ```bash
   source .venv/bin/activate
   ```
2. After adding or updating any dependencies, update the `requirements.txt` file:
   ```bash
   pip freeze > requirements.txt
   ```
3. Commit your changes to Git:
   ```bash
   git add requirements.txt
   git commit -m "Update requirements.txt with new dependencies"
   git push origin main
   ```

---

## Troubleshooting
- **Virtual Environment Issues**: If you encounter problems with the virtual environment, delete the `.venv` folder and recreate it.
- **Streamlit Errors**: Ensure that Streamlit is correctly installed and the Python version matches the project requirements (e.g., Python 3.8+).

---

## Contributing
1. Fork the repository.
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a Pull Request.

---

## About
This app is built using:
- [Streamlit](https://streamlit.io/) for the UI.
- Python 3.8+ for development.
- Backend API to handle AI-based nutritional analysis.