#!/usr/bin/env python
# coding: utf-8



from deepface import DeepFace
import matplotlib.pyplot as plt

import os
import cv2
import time
import pandas as pd
from deepface import DeepFace
import streamlit as st
from PIL import Image
import numpy as np

# Available DeepFace models
models = ["VGG-Face", "Facenet", "OpenFace", "DeepID", "Dlib", "ArcFace"]

# Streamlit UI
st.title("Face Analysis with DeepFace")

# Mode selection
mode = st.radio("Choose Input Mode", ["Upload Single Image", "Process Images from Directory"])

# Model selection dropdown
selected_model = st.selectbox("Choose a DeepFace model:", models)

# Function to analyze an image
def analyze_image(image_path):
    """Analyzes an image and returns DeepFace results."""
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'emotion'])
        return result[0]  # Return first face's analysis
    except Exception as e:
        st.error(f"Error analyzing {image_path}: {e}")
        return None

# If user wants to upload a single image
if mode == "Upload Single Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        temp_image_path = "temp.jpg"
        cv2.imwrite(temp_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Run analysis
        start_time = time.time()
        result = analyze_image(temp_image_path)
        end_time = time.time()

        if result:
            # Display results
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.subheader("Results:")
            st.write(f"**Age:** {result['age']} years")
            st.write(f"**Gender:** {result['dominant_gender']}")
            st.write(f"**Emotion:** {result['dominant_emotion']}")
            st.write(f"**Execution Time:** {round(end_time - start_time, 2)} seconds")

            # Save result
            df = pd.DataFrame([{
                "Model": selected_model,
                "Image": uploaded_file.name,
                "Age": result['age'],
                "Gender": result['dominant_gender'],
                "Emotion": result['dominant_emotion'],
                "Time (seconds)": round(end_time - start_time, 2)
            }])
            df.to_csv("deepface_results.csv", mode='a', header=not os.path.exists("deepface_results.csv"), index=False)
            st.success("Results saved!")

# If user wants to process an entire directory
elif mode == "Process Images from Directory":
    image_dir = st.text_input("Enter full directory path (e.g., C:\\Users\\Username\\Downloads\\models)")

    if st.button("Analyze Images"):
        if os.path.isdir(image_dir):
            image_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".jpg", ".png", ".jpeg"))]
            
            if len(image_files) == 0:
                st.warning("No images found in the directory.")
            else:
                results = []
                for image_path in image_files:
                    st.write(f"Processing: {os.path.basename(image_path)}")
                    start_time = time.time()
                    result = analyze_image(image_path)
                    end_time = time.time()

                    if result:
                        results.append({
                            "Model": selected_model,
                            "Image": os.path.basename(image_path),
                            "Age": result['age'],
                            "Gender": result['dominant_gender'],
                            "Emotion": result['dominant_emotion'],
                            "Time (seconds)": round(end_time - start_time, 2)
                        })
                
                if results:
                    # Convert results to DataFrame and save to CSV
                    df = pd.DataFrame(results)
                    df.to_csv("deepface_results.csv", mode='a', header=not os.path.exists("deepface_results.csv"), index=False)
                    st.success(f"Analysis completed! {len(results)} images processed and results saved.")
        else:
            st.error("Invalid directory path. Please check and try again.")




