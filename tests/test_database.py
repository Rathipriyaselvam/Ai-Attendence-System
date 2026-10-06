import pytest
import os
import sqlite3
import numpy as np

from src.database import (
    init_db,
    add_student,
    get_student,
    get_all_students,
    update_student,
    delete_student,
    save_face_embedding,
    get_embeddings_count
)

@pytest.fixture(autouse=True)
def setup_test_db():
    """Setup clean database schema before tests."""
    init_db()
    yield

def test_add_and_get_student():
    sid = "TEST_CS_01"
    name = "Test Student 1"
    dept = "Computer Science & Engineering"
    year = "3rd Year"
    sec = "Section A"
    email = "test1@example.com"

    # Clean prior test data if any
    delete_student(sid)

    success, msg = add_student(sid, name, dept, year, sec, email)
    assert success is True

    student = get_student(sid)
    assert student is not None
    assert student["student_id"] == sid
    assert student["name"] == name
    assert student["department"] == dept

def test_prevent_duplicate_student_id():
    sid = "TEST_CS_DUP"
    delete_student(sid)

    success1, _ = add_student(sid, "Student Original", "Computer Science", "1st Year", "Section A")
    assert success1 is True

    success2, msg = add_student(sid, "Student Duplicate", "Computer Science", "1st Year", "Section A")
    assert success2 is False
    assert "already exists" in msg.lower()

def test_update_student_profile():
    sid = "TEST_CS_UPD"
    delete_student(sid)

    add_student(sid, "Initial Name", "Computer Science", "1st Year", "Section A")
    upd_success, _ = update_student(sid, "Updated Name", "Information Technology", "2nd Year", "Section B", "upd@example.com")
    assert upd_success is True

    updated_st = get_student(sid)
    assert updated_st["name"] == "Updated Name"
    assert updated_st["department"] == "Information Technology"

def test_embedding_storage_and_cascade_deletion():
    sid = "TEST_EMB_01"
    delete_student(sid)

    add_student(sid, "Embedding Test Student", "Computer Science", "1st Year", "Section A")

    fake_emb = np.random.randn(512).tolist()
    save_face_embedding(sid, fake_emb, model_name="Facenet512")

    count = get_embeddings_count(sid)
    assert count == 1

    # Delete student and verify cascade deletion
    del_success, _ = delete_student(sid)
    assert del_success is True
    assert get_student(sid) is None
    assert get_embeddings_count(sid) == 0
