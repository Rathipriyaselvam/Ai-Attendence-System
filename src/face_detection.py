import cv2
import numpy as np

# Load Haar Cascade Classifier for fast offline OpenCV face detection
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def detect_faces_opencv(frame, min_neighbors=5, scale_factor=1.1):
    """
    Detect faces using OpenCV Haar Cascade Classifier.
    Returns a list of bounding boxes [(x, y, w, h), ...].
    """
    if frame is None:
        return []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(60, 60)
    )
    
    return list(faces)

def validate_single_face(frame):
    """
    Validate that exactly one face is present in the captured frame.
    Returns: (is_valid: bool, faces: list, message: str, cropped_face: np.ndarray)
    """
    if frame is None:
        return False, [], "No frame captured from webcam.", None
    
    faces = detect_faces_opencv(frame)
    num_faces = len(faces)
    
    if num_faces == 0:
        return False, [], "No face detected. Please align your face inside the camera view.", None
    elif num_faces > 1:
        return False, faces, f"Multiple faces detected ({num_faces}). Ensure only ONE face is in frame.", None
    
    x, y, w, h = faces[0]
    # Add a slight margin around the face box
    h_margin = int(h * 0.15)
    w_margin = int(w * 0.15)
    
    img_h, img_w = frame.shape[:2]
    y1 = max(0, y - h_margin)
    y2 = min(img_h, y + h + h_margin)
    x1 = max(0, x - w_margin)
    x2 = min(img_w, x + w + w_margin)
    
    cropped_face = frame[y1:y2, x1:x2]
    return True, faces, "Face validated successfully.", cropped_face

def draw_recognition_box(frame, bbox, name, student_id, confidence, is_known=True, status="Present"):
    """
    Draw an aesthetically pleasing bounding box and label overlay on frame.
    Modifies frame in-place.
    """
    x, y, w, h = bbox
    
    # Colors (BGR)
    if is_known:
        color = (0, 200, 83)      # Vibrant Green
        bg_color = (0, 150, 60)
    else:
        color = (40, 40, 230)     # Vibrant Red
        bg_color = (30, 30, 180)
        
    # Draw face box corners
    thickness = 2
    corner_len = int(min(w, h) * 0.2)
    
    # Outer rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Header tag box above or inside face
    header_h = 45
    header_y1 = max(0, y - header_h)
    header_y2 = y
    
    if header_y1 == 0:
        header_y1 = y
        header_y2 = y + header_h
        
    # Draw semi-transparent header overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, header_y1), (x + w, header_y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    if is_known:
        title_text = f"{name}"
        sub_text = f"ID: {student_id} | {confidence:.0f}%"
    else:
        title_text = "UNKNOWN PERSON"
        sub_text = "Face Not Recognized"
        
    cv2.putText(frame, title_text, (x + 6, header_y1 + 18), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, sub_text, (x + 6, header_y1 + 36), font, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    
    return frame

def resize_frame(frame, target_width=640):
    """Resize frame keeping aspect ratio for fast recognition processing."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
