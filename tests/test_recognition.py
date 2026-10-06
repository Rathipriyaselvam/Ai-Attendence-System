import pytest
import numpy as np

from src.face_recognition import (
    calculate_distance,
    match_face_embedding
)

def test_calculate_distance():
    # Identical vectors should have distance 0.0
    vec1 = [1.0, 0.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0, 0.0]
    dist_cosine = calculate_distance(vec1, vec2, metric="cosine")
    assert abs(dist_cosine - 0.0) < 1e-5

    # Orthogonal vectors should have cosine distance 1.0
    vec3 = [0.0, 1.0, 0.0, 0.0]
    dist_ortho = calculate_distance(vec1, vec3, metric="cosine")
    assert abs(dist_ortho - 1.0) < 1e-5

def test_match_face_embedding_known():
    registered = {
        "STUDENT_A": [[1.0, 0.0, 0.0, 0.0]],
        "STUDENT_B": [[0.0, 1.0, 0.0, 0.0]]
    }

    # Test exact match for STUDENT_A
    live_emb = [0.98, 0.02, 0.0, 0.0]
    matched_id, dist, conf, is_known = match_face_embedding(live_emb, registered, threshold=0.45, metric="cosine")

    assert is_known is True
    assert matched_id == "STUDENT_A"
    assert conf > 80.0

def test_match_face_embedding_unknown():
    registered = {
        "STUDENT_A": [[1.0, 0.0, 0.0, 0.0]]
    }

    # Vector far away from STUDENT_A
    live_emb = [0.0, 0.0, 0.0, 1.0]
    matched_id, dist, conf, is_known = match_face_embedding(live_emb, registered, threshold=0.45, metric="cosine")

    assert is_known is False
    assert matched_id is None
