import os
import tensorflow as tf
import numpy as np
import cv2
from typing import List
import subprocess

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# print(
#     f"The vocabulary is: {char_to_num.get_vocabulary()} "
#     f"(size ={char_to_num.vocabulary_size()})"
# )

def load_video(path:str) -> List[float]: 

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()

    frames = np.array(frames)

    # Normalize frames
    mean = np.mean(frames)
    std = np.std(frames)
    normalized_frames = (frames - mean) / (std)  # Prevent divide by zero
    return normalized_frames

def load_alignments(path:str) -> List[str]: 
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
   
    result = char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]
    result = tf.expand_dims(result, axis = 0)
    return result

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    # file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('..','Data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('..','Data','align',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)

    return frames, alignments


def is_video_corrupted(path: str) -> bool:
    # FFmpeg command to detect video errors
    command = ["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.stderr:
            print(f"Errors found in video {path}:\n{result.stderr}")
            return True
    except Exception as e:
        print(f"Error running FFmpeg: {e}")
        return True
    return False

def get_corrupted():
    corrupted = []
    for file in os.listdir(os.path.join('..','Data', 's1')):
        #print(file)
        path = os.path.join('..','Data', 's1',f'{file}')
        # print(f"Processing file: {file}")
        if is_video_corrupted(path):
            corrupted.append(file)
        
    return corrupted

