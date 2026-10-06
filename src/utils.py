import os
import re
import cv2
import streamlit as st
from PIL import Image
import numpy as np

def sanitize_filename(filename):
    """Sanitize student ID or filename to prevent directory traversal and invalid characters."""
    return re.sub(r'[^\w\-]', '_', str(filename).strip())

def load_css(css_file_path):
    """Inject custom CSS into Streamlit application."""
    if os.path.exists(css_file_path):
        with open(css_file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def bgr_to_rgb(image_array):
    """Convert OpenCV BGR image array to RGB for Streamlit display."""
    if image_array is None:
        return None
    return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

def pil_to_bgr(pil_image):
    """Convert PIL Image to OpenCV BGR numpy array."""
    if pil_image is None:
        return None
    rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def format_confidence_badge(confidence_pct):
    """Return colored HTML badge string for confidence score."""
    if confidence_pct >= 85:
        color = "#10B981"
    elif confidence_pct >= 70:
        color = "#3B82F6"
    else:
        color = "#F59E0B"
    return f"<span style='color: {color}; font-weight: 700;'>{confidence_pct:.1f}%</span>"
