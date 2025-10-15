# app.py (Phase 2 Version with Educator Dashboard)

import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Wound Care Toolkit",
    page_icon="🎓",
    layout="wide"
)

# --- CUSTOM STYLING (CSS) ---
st.markdown("""
<style>
    .main-container { max-width: 900px; margin: auto; padding: 2rem; border: 2px solid #e0e0e0; border-radius: 15px; background-color: #f9f9f9; }
    .result-card { background-color: #e7f3ff; border: 1px solid #cce0ff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1); margin-top: 1rem; }
    h1, h2, h3 { color: #0d47a1; }
    .st-expander { border-color: #cce0ff !important; }
</style>
""", unsafe_allow_html=True)

# --- PATH AND MODEL CONFIGURATION ---
workspace_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(workspace_path, "models")
demo_images_path = os.path.join(workspace_path, "demo_images")

MODEL_INFO = {
    "Pressure Ulcer Staging": {
        "path": os.path.join(models_path, "pressure_ulcer_model.keras"),
        "class_names": ['SDTI', 'Stage_I', 'Stage_II', 'Stage_III', 'Stage_IV', 'Unstageable'],
        "description": "This model classifies pressure ulcers into their respective stages."
    },
    "DFU Classification": {
        "path": os.path.join(models_path, "dfu_model.keras"),
        "class_names": ['healthy', 'ulcer'],
        "description": "This model analyzes an image patch to determine if it contains an ulcer or healthy skin."
    }
}

# --- MODEL LOADING FUNCTION ---
@st.cache_resource
def load_model(model_name):
    model_path = MODEL_INFO[model_name]["path"]
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

# --- MAIN APP LAYOUT ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

view_mode = st.radio(
    "Select Your View",
    ("Student View", "Educator View"),
    horizontal=True,
    label_visibility="collapsed"
)

# --- Add a spinner while the selected model loads for the first time ---
with st.spinner("Initializing AI model, please be patient... This may take a moment on first load."):
    if view_mode == "Student View":
        model = load_model(st.session_state.get('analysis_type', 'Pressure Ulcer Staging'))
    else:
        model = load_model("Pressure Ulcer Staging") # Educator view uses one model

# -------------------------------------------------------------------
# --- STUDENT VIEW ---
# -------------------------------------------------------------------
if view_mode == "Student View":
    st.title("AI Wound Analysis Tool 🩺")
    st.write("")

    st.subheader("1. Select the Analysis Type")
    # Note: we use session_state to help the spinner load the right model
    analysis_type = st.selectbox("Choose which AI model to use:", list(MODEL_INFO.keys()), key='analysis_type')
    st.info(MODEL_INFO[analysis_type]['description'], icon="ℹ️")

    st.subheader("2. Upload Your Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key=f"student_{analysis_type}")

    if uploaded_file is not None:
        # ... [The rest of the Student View code is identical to the previous version] ...
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        if model is not None:
            # Analysis logic...
            col1, col2 = st.columns(2)
            # ...
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# --- EDUCATOR VIEW ---
# -------------------------------------------------------------------
elif view_mode == "Educator View":
    st.title("Educator Dashboard 📊")
    # ... [The rest of the Educator View code is identical to the previous version] ...

    st.markdown('</div>', unsafe_allow_html=True)
    st.write("This dashboard simulates a classroom activity to demonstrate how an instructor could track student performance using the AI tool.")

    if st.button("Simulate Class Submissions", key="demo_button"):
        with st.spinner("Generating demo data..."):
            student_names = ["Alex", "Ben", "Chloe", "David", "Eva", "Frank", "Grace", "Henry"]
            demo_files = [f for f in os.listdir(demo_images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            if not demo_files:
                st.error("No images found in the 'demo_images' folder. Please add some before running the simulation.")
                st.stop()

            model = load_model("Pressure Ulcer Staging") # Use one model for the demo
            if model is None:
                st.error("Pressure Ulcer Staging model not found. Cannot run simulation.")
                st.stop()
            
            results = []
            for student in student_names:
                image_name = random.choice(demo_files)
                true_label = image_name.split('_')[0]
                
                image = Image.open(os.path.join(demo_images_path, image_name)).convert('RGB')
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = model.predict(img_array)
                scores = tf.nn.softmax(predictions[0])
                class_names = MODEL_INFO["Pressure Ulcer Staging"]["class_names"]
                predicted_class = class_names[np.argmax(scores)]
                
                results.append({
                    "Student": student,
                    "Image Tested": image_name,
                    "True Stage": true_label,
                    "AI Prediction": predicted_class,
                    "Correct": true_label == predicted_class
                })
            
            # Store results in session state to persist them
            st.session_state.demo_results = pd.DataFrame(results)

    if 'demo_results' in st.session_state:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        results_df = st.session_state.demo_results
        
        st.subheader("Simulated Class Results")
        
        # Display key metrics
        accuracy = (results_df["Correct"].sum() / len(results_df)) * 100
        st.metric("Overall Class Accuracy", f"{accuracy:.2f}%")
        
        # Display the full results table
        st.dataframe(results_df)

        # Chart 1: Accuracy per student
        st.subheader("Performance by Student")
        student_accuracy = results_df.groupby('Student')['Correct'].mean().reset_index()
        student_accuracy['Correct'] = student_accuracy['Correct'] * 100
        fig1 = px.bar(student_accuracy, x='Student', y='Correct', title='Student Accuracy (%)', text_auto='.2f', color_discrete_sequence=['#66bb6a'])
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Most common errors
        st.subheader("Analysis of Misclassifications")
        error_df = results_df[~results_df['Correct']]
        if not error_df.empty:
            error_counts = error_df.groupby(['True Stage', 'AI Prediction']).size().reset_index(name='counts')
            fig2 = px.bar(error_counts, x='True Stage', y='counts', color='AI Prediction', title='Common Misclassification Types', barmode='group')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.success("No misclassifications in this simulation!")
            
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Close the main container