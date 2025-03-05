import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import av
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
mp_face_mesh = mp.solutions.face_mesh

# Define Lip Landmark Indices
LIP_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 
                 13, 311, 308, 402, 14, 178, 88, 95]

# WebRTC Video Processing Class
class VideoProcessor(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    async def recv(self):
        frame = await super().recv()
        img = frame.to_ndarray(format="bgr24")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            logger.error("⚠️ No face detected!")
        else:
            logger.info("✅ Face detected!")

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lip_points = np.array([
                    (int(face_landmarks.landmark[i].x * img.shape[1]), 
                    int(face_landmarks.landmark[i].y * img.shape[0])) for i in LIP_LANDMARKS
                ])

                if len(lip_points) == 0:
                    logger.error("⚠️ Lip landmarks not found!")
                else:
                    logger.info(f"✅ Detected {len(lip_points)} lip landmarks.")

                x_min, y_min = np.min(lip_points, axis=0)
                x_max, y_max = np.max(lip_points, axis=0)

                # Expand width to make it more visible
                extra_width = 30  # Increase width
                extra_height = 20  # Increase height
                x_min = max(x_min - extra_width, 0)
                x_max = min(x_max + extra_width, img.shape[1])
                y_min = max(y_min - extra_height, 0)
                y_max = min(y_max + extra_height, img.shape[0])

                # Draw bounding box on the frame
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

@app.get("/", response_class=HTMLResponse)
async def index():
    return open("index.html").read()

# WebRTC Route
@app.post("/offer")
async def offer(sdp: dict):
    pc = RTCPeerConnection()

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            logger.info("Received video track!")  # Debugging log
            pc.addTrack(VideoProcessor())  # Attach processed video back to WebRTC

    desc = RTCSessionDescription(sdp["sdp"], sdp["type"])
    await pc.setRemoteDescription(desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
