import numpy as np
import cv2
import os
import streamlit as st
from scipy.spatial.distance import cosine, euclidean
from src.database import get_all_embeddings, save_face_embedding

# Global cache for pre-loaded embeddings to eliminate DB latency during webcam loop
_EMBEDDINGS_CACHE = {}
_CACHE_MODEL = None

def get_registered_embeddings(model_name="Facenet512", force_reload=False):
    """Retrieve registered student embeddings from cache or database."""
    global _EMBEDDINGS_CACHE, _CACHE_MODEL
    if force_reload or _CACHE_MODEL != model_name or not _EMBEDDINGS_CACHE:
        _EMBEDDINGS_CACHE = get_all_embeddings(model_name=model_name)
        _CACHE_MODEL = model_name
    return _EMBEDDINGS_CACHE

def calculate_distance(emb1, emb2, metric="cosine"):
    """Calculate vector distance between two face embeddings."""
    u = np.array(emb1, dtype=np.float64)
    v = np.array(emb2, dtype=np.float64)
    
    if metric == "cosine":
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 1.0
        return float(1.0 - np.dot(u, v) / (norm_u * norm_v))
    elif metric == "euclidean_l2":
        u_norm = u / (np.linalg.norm(u) + 1e-10)
        v_norm = v / (np.linalg.norm(v) + 1e-10)
        return float(np.linalg.norm(u_norm - v_norm))
    else:  # euclidean
        return float(np.linalg.norm(u - v))

def extract_face_embedding(img_input, model_name="Facenet512", detector_backend="opencv"):
    """
    Extract 512D or N-D face embedding vector using DeepFace.
    `img_input` can be a file path (str) or a BGR frame (np.ndarray).
    Returns embedding vector as a python list, or None if extraction fails.
    """
    try:
        from deepface import DeepFace
        # Convert BGR frame to RGB if numpy array
        if isinstance(img_input, np.ndarray):
            if len(img_input.shape) == 3 and img_input.shape[2] == 3:
                img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img_input
        else:
            img_rgb = img_input

        # DeepFace represent call
        results = DeepFace.represent(
            img_path=img_rgb,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True
        )

        if results and len(results) > 0:
            return results[0]["embedding"]
        return None
    except Exception as e:
        print(f"DeepFace Embedding Extraction Error: {e}")
        return None

def match_face_embedding(live_embedding, registered_embeddings, threshold=0.45, metric="cosine"):
    """
    Compare live face embedding against registered embeddings.
    Returns: (matched_student_id, min_distance, confidence_pct, is_recognized)
    """
    if not live_embedding or not registered_embeddings:
        return None, 1.0, 0.0, False

    best_student_id = None
    min_distance = float("inf")

    # Compare against every student's saved sample embeddings
    for student_id, samples in registered_embeddings.items():
        for sample_emb in samples:
            dist = calculate_distance(live_embedding, sample_emb, metric=metric)
            if dist < min_distance:
                min_distance = dist
                best_student_id = student_id

    # Check if best distance satisfies threshold
    if min_distance <= threshold:
        # Calculate intuitive confidence percentage score
        # When dist = 0 -> confidence = 100%
        # When dist = threshold -> confidence = 60%
        confidence_pct = max(0.0, min(100.0, (1.0 - (min_distance / (threshold * 1.5))) * 100))
        return best_student_id, min_distance, round(confidence_pct, 1), True
    else:
        confidence_pct = max(0.0, (1.0 - min_distance) * 100)
        return None, min_distance, round(confidence_pct, 1), False

def process_face_samples_for_student(student_id, sample_images_paths_or_frames, model_name="Facenet512"):
    """
    Generate and save embeddings for multiple face samples of a student.
    Returns: (successful_count, total_count)
    """
    success_count = 0
    total = len(sample_images_paths_or_frames)

    for item in sample_images_paths_or_frames:
        emb = extract_face_embedding(item, model_name=model_name)
        if emb:
            save_face_embedding(student_id, emb, model_name=model_name)
            success_count += 1

    # Reload cache after adding new student
    get_registered_embeddings(model_name=model_name, force_reload=True)
    return success_count, total
