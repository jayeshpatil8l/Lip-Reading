import asyncio
import json
import logging
from fastapi.responses import HTMLResponse
import numpy as np
import multiprocessing
from fastapi import FastAPI, WebSocket
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaStreamTrack
from contextlib import asynccontextmanager
from utils import preprocessframe, prepare_websocket_message, normalize_frames
from model import LipReadingModel
import concurrent.futures
import pickle

# Create a process pool with 2 workers
pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

model = LipReadingModel()

# Graceful Shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # App running

    # Close all WebRTC connections before shutdown
    if pcs:
        logger.info("Shutting down: Closing WebRTC connections...")
        await asyncio.gather(*(pc.close() for pc in pcs))
        pcs.clear()
        logger.info("All WebRTC connections closed.")

# Initialize FastAPI with Lifespan
app = FastAPI(lifespan=lifespan)

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store active peer connections & WebSocket clients
pcs = set()
connected_clients = set()
should_send_frames = False # Flag to control frame sending
y_true = ""
 

# # WebRTC Video Stream Processing
# class LipReadingTrack(VideoStreamTrack):
#     def __init__(self, track):
#         super().__init__()
#         self.track = track
#         self.frames = []  # Store frames for batch processing
#         self.count = 0

#     async def recv(self):
#         global should_send_frames
#         frame = await self.track.recv()  # Receive a video frame from WebRTC

#         if should_send_frames:  # Only process frames if flag is enabled
#             img = frame.to_ndarray(format="bgr24")  # Convert frame to NumPy array
#             self.frames.append(img)
#             self.count += 1
#             logger.info(f"Frame {self.count} received.")

#             if len(self.frames) == 25:  # Process batch of 25 frames
#                 text_prediction = self.process_frames(self.frames)
#                 self.frames = []  # Reset frame buffer
#                 self.count = 0
#                 logger.info(f"Prediction: {text_prediction}")
#                 await self.send_prediction(text_prediction)

#         return frame  # Return frame (WebRTC expects this)

#     def process_frames(self, frames):
#         """ Simulated lip-reading model inference. """
#         frames_array = np.stack(frames, axis=0)  # Shape: (25, H, W, C)
#         return "example prediction"  # Replace with actual model inference

#     async def send_prediction(self, prediction):
#         """ Send predicted text to WebSocket clients. """
#         for websocket in connected_clients:
#             await websocket.send_text(prediction)

# Serve HTML Page
@app.get("/", response_class=HTMLResponse)
async def index():
    return open("index1.html").read()

@app.get("/alignments")
async def get_alignments():
    try:
        with open("alignments.pkl", "rb") as file:
            alignments = pickle.load(file)
            logger.info('alignments loaded!')
        return alignments
    except Exception as e:
        return {"error": str(e)}

# WebSocket Route
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global should_send_frames
    # global y_true
    await websocket.accept()
    connected_clients.add(websocket)
    

    try:
        while True:
            data = await websocket.receive_text()
            parsed_data = json.loads(data)
            message = parsed_data.get("message")  # Extract message
            alignment = parsed_data.get("alignment")  # Extract alignment

            if message == "start":
                should_send_frames = True
                model.reset_model_states()
                globals()['y_true']  = alignment
                logger.info("Frame sending started.")
            elif message == "stop":
                should_send_frames = False  
                logger.info("Frame sending stopped.")
               

    except:
        connected_clients.remove(websocket)

from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer

# config = RTCConfiguration([
#     RTCIceServer(urls=["stun:stun.l.google.com:19302"])  # Use Google STUN server
# ])

# pc = RTCPeerConnection(config)

# # Prefer VP8 or H.264 and remove "video/rtx"
# for transceiver in pc.getTransceivers():
#     if transceiver.kind == "video":
#         transceiver.setCodecPreferences([
#             codec for codec in transceiver.sender.getCapabilities().codecs
#             if codec.mimeType in ["video/VP8", "video/H264"]  # Keep only these codecs
#         ])

config = RTCConfiguration([
    RTCIceServer(urls=["stun:stun.l.google.com:19302"])  # Use Google STUN server
    ])

# WebRTC Signaling Route
@app.post("/offer")
async def offer(request: dict):
    
    pc = RTCPeerConnection(config)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(LipReadingTrack(track))  # Attach video processing track

    for transceiver in pc.getTransceivers():
        if transceiver.kind == "video":
            transceiver.setCodecPreferences([
                codec for codec in transceiver.sender.getCapabilities().codecs
                if codec.mimeType in ["video/VP8", "video/H264"]  # Keep only these codecs
            ])

    pcs.add(pc)
    
    sdp = RTCSessionDescription(sdp=request["sdp"], type=request["type"])
    await pc.setRemoteDescription(sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


from collections import deque

class LipReadingTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_queue = deque()  # Use deque for efficient frame storage
        self.processing = False  # Flag to check if processing is happening
        self.count = 1

    async def recv(self):
        global y_true
        frame = await self.track.recv() 
        if self.count == 1:
            logger.info("Started!")
            self.count -= 1

        if should_send_frames: 
            img = frame.to_ndarray(format="bgr24")  
            preprocessed_frame = preprocessframe(img)
            if preprocessed_frame is not None:
                self.frame_queue.append(preprocessed_frame)  # Store frame in the queue
                print(f"Frame received. Queue size: {len(self.frame_queue)}")
            
                message = prepare_websocket_message(frame = preprocessed_frame)
                for websocket in connected_clients:
                    await websocket.send_text(message)
                    logger.info("Frame Sent!")

            # Start processing if not already running
            if not self.processing and len(self.frame_queue) >= 25:
                await self.process_queue()

        if not should_send_frames and len(self.frame_queue) > 0: 
            await self.process_queue()
            print(f"Queue size: {len(self.frame_queue)}")
            print(y_true)
            accuracy = model.get_accuracy(y_true)
            for websocket in connected_clients:
                await websocket.send_json(accuracy)
            y_true = ""

        return frame # Return the frame (unused in WebRTC processing)

    async def process_queue(self):
        self.processing = True  # Mark processing as active
        length = len(self.frame_queue)
        # global y_true
        
        while len(self.frame_queue) > 0:  # Process as long as frames exist
            num_frames = min(len(self.frame_queue), 25)  # Take max 25 frames or remaining frames
            frames_batch = [self.frame_queue.popleft() for _ in range(num_frames)]
            frames = normalize_frames(frames_batch)
            
            # text_prediction = model.predict(frames_batch)  # Run model
            text_prediction = await asyncio.get_event_loop().run_in_executor(pool, model.predict, frames ) # running the model inference in another process to acheive multiprocessing
            await self.send_prediction(text_prediction)  # Send prediction

        # if self.frame_queue:  # If frames remain, process them
        #     frames_batch = list(self.frame_queue)  # Use all remaining frames
        #     self.frame_queue.clear()  # Empty queue
        #     text_prediction = get_predictions(frames_batch)  # Run model
        #     await self.send_prediction(text_prediction)
        
        self.processing = False  # Mark processing as finished
        # if not should_send_frames:
        #     accuracy = model.get_accuracy(y_true)
        #     for websocket in connected_clients:
        #         await websocket.send_json(accuracy)
        #     y_true = ""

        
    async def send_prediction(self, prediction):
        # Broadcast prediction to all WebSocket clients
        message = prepare_websocket_message(prediction = prediction)
        for websocket in connected_clients:
            await websocket.send_text(message)