import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf 
import logging
import json
import pickle
import Levenshtein 
import base64

with open('corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

# num_to_char = tf.keras.layers.StringLookup(
#     vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
# )

# vocab_size = num_to_char.vocabulary_size()


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

height, width, channels  = 46, 140, 1

# Define lip landmark indices (based on MediaPipe FaceMesh)
LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 
                 13, 311, 308, 402, 14, 178, 88, 95]  # Upper & lower lip landmarks

def preprocessframe(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    logger.info("RGB Image Shape: " + str(image_rgb.shape))

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)

    frame_gs = np.zeros((height, width, 1), dtype=np.uint8)

    if results.multi_face_landmarks:
        logger.info("Face Detected")
        for face_landmarks in results.multi_face_landmarks:
            # Extract lip landmark coordinates
            lip_points = np.array([(int(face_landmarks.landmark[i].x * image_rgb.shape[1]), 
                                    int(face_landmarks.landmark[i].y * image_rgb.shape[0])) for i in LIP_LANDMARKS])
            
            # Compute bounding box for lips
            x_min, y_min = np.min(lip_points, axis=0)
            x_max, y_max = np.max(lip_points, axis=0)

            # Expand bounding box slightly
            padding = 10
            extra_width = 30
            x_min = max(x_min - extra_width, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + extra_width, image_rgb.shape[1])
            y_max = min(y_max + padding, image_rgb.shape[0])

            # Crop the lips region
            lips_cropped = image_rgb[y_min:y_max, x_min:x_max]
        
            frame_gs = cv2.cvtColor(lips_cropped, cv2.COLOR_RGB2GRAY)
            frame_gs = cv2.resize(frame_gs, (width, height))
            frame_gs = np.expand_dims(frame_gs, axis = -1)
            
            # mean = np.mean(frame_gs)
            # std = np.std(frame_gs) 
            # frame_gs = (frame_gs - mean) / std
    else:
        logger.info("Face Not Detected")
        return None

    logger.info("Grayscaled Frame Shape: " + str(frame_gs.shape))        
    return frame_gs

# def reset_model_states(model):
#      for layer in model.layers:
#             if isinstance(layer, tf.keras.layers.Bidirectional):
#                 layer.reset_states()


# def get_predictions(model, frames):
#         # Convert frames to a NumPy array (Shape: (25, H, W, C))
#         frames_array = np.stack(frames, axis=0)  
#         input_length = frames_array.shape[0]
#         frames_array = np.expand_dims(frames_array, axis = 0)

#         logits = model.predict(frames_array)
#         decoded = tf.keras.backend.ctc_decode(logits, input_length=[input_length], greedy=True)[0][0].numpy()
#         predicted = tf.strings.reduce_join([num_to_char(word) for word in decoded[0]]).numpy().decode('utf-8')

#         logger.info(f"Model Predicted: {predicted}")

#         return predicted    

def prepare_websocket_message(frame = None, prediction=None):
    message = {}
    if frame is not None: 
        _, buffer = cv2.imencode(".jpg", frame.squeeze())
        frame_base64 = base64.b64encode(buffer).decode("utf-8")
        message = {"frame": frame_base64}
        logger.info("Sending frames using websocket")

    elif prediction:
        message = {"prediction": prediction} 
    return json.dumps(message)
        
def get_corrected_preds(prediction):
    predicted_words = prediction.split()
    corrected_pred = ""
    for word in predicted_words:
        if word in corpus:
            corrected_pred += word + " "
        else:
            word = min(corpus, key = lambda w: Levenshtein.distance(word, w))
            corrected_pred += word + " "

    return corrected_pred.strip()
    

def normalize_frames(frames):
    mean = np.mean(frames)
    std = np.std(frames) 
    normalized_frames = (frames - mean) / std
    return normalized_frames

