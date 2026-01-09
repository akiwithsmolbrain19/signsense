import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pandas as pd
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "asl_model.tflite")
TRAIN_CSV_PATH = os.path.join(SCRIPT_DIR, "train.csv")

INPUT_SIZE = 64
CONFIDENCE_THRESHOLD = 0.5

# MediaPipe Indices
LIPS_IDXS0 = np.array([
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ])
LEFT_HAND_IDXS0 = np.arange(468, 489)
RIGHT_HAND_IDXS0 = np.arange(522, 543)
LEFT_POSE_IDXS0 = np.array([502, 504, 506, 508, 510])
RIGHT_POSE_IDXS0 = np.array([503, 505, 507, 509, 511])

LANDMARK_IDXS_LEFT_DOMINANT0 = np.concatenate((LIPS_IDXS0, LEFT_HAND_IDXS0, LEFT_POSE_IDXS0))
LANDMARK_IDXS_RIGHT_DOMINANT0 = np.concatenate((LIPS_IDXS0, RIGHT_HAND_IDXS0, RIGHT_POSE_IDXS0))

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_labels():
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"ERROR: train.csv not found at {TRAIN_CSV_PATH}")
        return []
    try:
        if TRAIN_CSV_PATH.endswith('.zip'):
             train = pd.read_csv(TRAIN_CSV_PATH, compression='zip')
        else:
             train = pd.read_csv(TRAIN_CSV_PATH)
        labels = list(train['sign'].astype('category').cat.categories)
        return labels
    except Exception as e:
        print(f"Error reading labels: {e}")
        return []

def extract_landmarks(results):
    # FIXED: Added .landmark to access the actual list of points
    def get_coords(landmarks_obj, num_points):
        if landmarks_obj:
            return np.array([[res.x, res.y, res.z] for res in landmarks_obj.landmark])
        return np.zeros((num_points, 3))
    
    face = get_coords(results.face_landmarks, 468)
    left_hand = get_coords(results.left_hand_landmarks, 21)
    pose = get_coords(results.pose_landmarks, 33)
    right_hand = get_coords(results.right_hand_landmarks, 21)
    
    # Concatenate all landmarks
    return np.concatenate([face, left_hand, pose, right_hand])

def process_frame_for_model(frame_buffer):
    data = np.array(frame_buffer)
    
    # Interpolate to 64 frames
    if len(data) != INPUT_SIZE:
        x_old = np.linspace(0, len(data) - 1, len(data))
        x_new = np.linspace(0, len(data) - 1, INPUT_SIZE)
        data_resized = np.zeros((INPUT_SIZE, 543, 3))
        for i in range(543):
            for j in range(3):
                data_resized[:, i, j] = np.interp(x_new, x_old, data[:, i, j])
        data = data_resized
    
    # Dominant Hand Logic
    left_hand_sum = np.sum(np.abs(data[:, LEFT_HAND_IDXS0, :]))
    right_hand_sum = np.sum(np.abs(data[:, RIGHT_HAND_IDXS0, :]))
    
    if left_hand_sum > right_hand_sum:
        data = data[:, LANDMARK_IDXS_LEFT_DOMINANT0, :]
    else:
        data = data[:, LANDMARK_IDXS_RIGHT_DOMINANT0, :]
        data[:, :, 0] = -data[:, :, 0] # Mirror

    data = np.nan_to_num(data)
    
    # Prepare Inputs for TFLite (Float32)
    frames = np.expand_dims(data, axis=0).astype(np.float32)
    idxs = np.expand_dims(np.arange(INPUT_SIZE), axis=0).astype(np.float32)
    
    return frames, idxs

# ==========================================
# 3. TFLITE INFERENCE CLASS
# ==========================================
class TFLiteModel:
    def __init__(self, model_path):
        print(f"Loading TFLite model from {model_path}...")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.frames_idx = 0
        self.indices_idx = 1
        
        for i, detail in enumerate(self.input_details):
            if detail['shape'][-1] == 3:
                self.frames_idx = i
            else:
                self.indices_idx = i
        print("✅ TFLite Model loaded successfully!")

    def predict(self, frames, idxs):
        self.interpreter.set_tensor(self.input_details[self.frames_idx]['index'], frames)
        self.interpreter.set_tensor(self.input_details[self.indices_idx]['index'], idxs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    labels = get_labels()
    if not labels: return
    print(f"Loaded {len(labels)} classes.")

    try:
        model = TFLiteModel(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic
    
    frame_buffer = []
    prediction_text = "Waiting..."
    confidence_text = ""
    
    print("\nStarting Webcam... Press 'q' to quit.")
    print("Keep your hand steady for a moment to trigger a prediction.")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # --- Data Collection ---
            landmarks = extract_landmarks(results)
            frame_buffer.append(landmarks)
            
            # Sliding Window (Keep last 64 frames)
            if len(frame_buffer) > INPUT_SIZE:
                frame_buffer.pop(0)
            
            # --- Prediction Logic ---
            # Predict every 5 frames, but only if we have at least 30 frames of data
            if len(frame_buffer) >= 30 and len(frame_buffer) % 5 == 0:
                X_frames, X_idxs = process_frame_for_model(frame_buffer)
                
                preds = model.predict(X_frames, X_idxs)
                pred_idx = np.argmax(preds)
                conf = np.max(preds)
                
                if conf > CONFIDENCE_THRESHOLD:
                    prediction_text = labels[pred_idx]
                    confidence_text = f"{conf:.0%}"
                    color = (0, 255, 0) # Green
                else:
                    # Keep old prediction but dim the text or show '...'
                    # prediction_text = "..." 
                    confidence_text = f"{conf:.0%}"
                    color = (0, 165, 255) # Orange

            # --- Display ---
            cv2.rectangle(image, (0, 0), (640, 60), (0, 0, 0), -1)
            cv2.putText(image, f"Sign: {prediction_text} ({confidence_text})", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('ASL Model TFLite', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()