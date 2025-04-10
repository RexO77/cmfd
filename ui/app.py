import streamlit as st
import os
import tempfile
import cv2
import numpy as np
import time
import sys

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.infer import predict, detect_forgery
from utils.mac_utils import get_mac_info

# Set page configuration
st.set_page_config(
    page_title="Copy-Move Forgery Detection",
    page_icon="🔍",
    layout="wide"
)

# Display app title and description
st.title("Copy-Move Forgery Detection")
st.markdown("""
This application uses a Vision Transformer and Siamese Network to detect copy-move forgery in images.
Upload an image to detect if it has been manipulated through copy-move operations.
""")

# Display system information
with st.expander("System Information"):
    mac_info = get_mac_info()
    for key, value in mac_info.items():
        st.write(f"**{key}:** {value}")

# Sidebar options
st.sidebar.header("Detection Options")
detection_mode = st.sidebar.radio("Detection Mode", ["Single Image", "Compare Two Images"])
model_path = st.sidebar.text_input("Model Path", value="outputs/checkpoints/best_model.pt")

# Advanced options
with st.sidebar.expander("Advanced Options"):
    patch_size = st.slider("Patch Size", min_value=32, max_value=128, value=64, step=16)
    stride = st.slider("Stride", min_value=16, max_value=64, value=32, step=8)
    threshold = st.slider("Detection Threshold", min_value=0.1, max_value=0.9, value=0.7, step=0.05)

# Main content
if detection_mode == "Single Image":
    st.header("Single Image Forgery Detection")
    st.write("Upload an image to check for copy-move forgery.")
    
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_image.getvalue())
            image_path = tmp_file.name
        
        # Display the image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Process when button is clicked
        if st.button("Detect Forgery"):
            with st.spinner("Analyzing image for forgery..."):
                # Record start time
                start_time = time.time()
                
                # Run forgery detection
                result = detect_forgery(
                    image_path=image_path,
                    model_path=model_path,
                    patch_size=patch_size,
                    stride=stride,
                    threshold=threshold
                )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Display results
                st.subheader("Results")
                
                # Create columns for better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Forgery Probability", f"{result['forgery_probability']:.2%}")
                    st.metric("Suspicious Regions", len(result['suspicious_pairs']))
                    st.metric("Processing Time", f"{processing_time:.2f} seconds")
                    
                    if result['forgery_detected']:
                        st.error("⚠️ Copy-Move Forgery Detected!")
                    else:
                        st.success("✅ No Forgery Detected")
                
                with col2:
                    if os.path.exists(result['heatmap_path']):
                        st.image(result['heatmap_path'], caption="Forgery Heatmap", use_column_width=True)
                    else:
                        st.error("Heatmap visualization failed to generate")
                
                # Remove temporary file
                os.unlink(image_path)
                
else:  # Compare Two Images
    st.header("Compare Two Image Regions")
    st.write("Upload two images to check their similarity.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        img1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
        if img1:
            st.image(img1, caption="Image 1", use_column_width=True)
    
    with col2:
        img2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])
        if img2:
            st.image(img2, caption="Image 2", use_column_width=True)
    
    if img1 and img2:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp1:
            tmp1.write(img1.getvalue())
            path1 = tmp1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
            tmp2.write(img2.getvalue())
            path2 = tmp2.name
        
        # Process when button is clicked
        if st.button("Compare Images"):
            with st.spinner("Analyzing similarity..."):
                # Record start time
                start_time = time.time()
                
                # Calculate similarity
                similarity = predict(path1, path2, model_path)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Display results
                st.subheader("Results")
                
                # Create metrics display
                st.metric("Similarity Score", f"{similarity:.2%}")
                st.metric("Processing Time", f"{processing_time:.2f} seconds")
                
                # Interpretation
                if similarity > 0.7:
                    st.error("⚠️ High Similarity Detected - Potential Forgery!")
                elif similarity > 0.5:
                    st.warning("⚠️ Moderate Similarity - Possible Forgery")
                else:
                    st.success("✅ Low Similarity - Likely Different Regions")
                
                # Create a gauge chart for similarity
                import plotly.graph_objects as go
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = similarity,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Similarity Score"},
                    gauge = {
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "green"},
                            {'range': [0.5, 0.7], 'color': "orange"},
                            {'range': [0.7, 1], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.7
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Remove temporary files
                os.unlink(path1)
                os.unlink(path2)

# Footer
st.markdown("---")
st.markdown("Copy-Move Forgery Detection using Vision Transformer and Siamese Network | © 2025")
