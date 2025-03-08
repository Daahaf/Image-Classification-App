#!/usr/bin/env python
# coding: utf-8

# In[5]:




# In[7]:


import numpy as np
import pandas as pd
from deepface import DeepFace
import matplotlib.pyplot as plt

# Load an image and analyze it
image_path = r"C:\Users\fahaa\Downloads\Ryan Gosling.jpg"
result = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'emotion'], enforce_detection=False)

print(result)


# In[18]:


import numpy as np
import pandas as pd
from deepface import DeepFace
import matplotlib.pyplot as plt

# Load an image and analyze it
image_path = r"C:\Users\fahaa\Downloads\Ryan Gosling.jpg"
result = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'emotion'])

print(result)


# In[12]:


import cv2


# In[14]:


# Load the image using OpenCV
# Load the image using OpenCV
image_path = r"C:\Users\fahaa\Downloads\Ryan Gosling.jpg"
image = cv2.imread(image_path)

image = cv2.resize(image, (200, 200))

# Convert the image from BGR to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



# Display the image using matplotlib
plt.imshow(image_rgb)
plt.axis('off')  # Hide axes for better display
plt.show()

# Analyze the image with DeepFace
result = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'emotion'], enforce_detection=False)

print(result)





# In[20]:


import numpy as np
import pandas as pd
from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2
import os

# Directory containing images
image_dir = r"C:\Users\fahaa\Downloads\models"

# List to hold results
results = []

# Loop through each image in the directory
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Check if the image is loaded correctly
    if image is None:
        print(f"Error: Image {image_name} not loaded correctly.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

   # Analyze the image with DeepFace using the default model
try:
    result = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'emotion'])
    
    if isinstance(result, list) and len(result) > 0:  # Ensure the result is valid
        results.append(result[0])  # Append the first result dictionary
        print(f"Results for {image_name}: {result[0]}")
    else:
        print(f"Warning: No valid face detected in {image_name}")

except Exception as e:
    print(f"Error analyzing {image_name}: {e}")

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
print(results_df)


# In[ ]:

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
    image_dir = st.text_input("Enter full directory path (e.g., C:\\Users\\fahaa\\Downloads\\models)")

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




