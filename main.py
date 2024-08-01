# Import the modules we need

print("Loading OpenCV...")
import cv2
print("Loading Ultralytics YOLOv8...")
from ultralytics import YOLO
print("Importing NumPy...")
import numpy as np
print("Importing argparse...")
import argparse
print("Importing AsyncIO...")
import asyncio
print("Importing JSON...")
import json
print("Importing Logging...")
import logging
print("Importing OS Tools...")
import os
print("Importing SSL Tools...")
import ssl
print("Importing UUID Tools...")
import uuid
print("Loading GPU Accelerated Torch...")
import torch
print("Loading AIORTC Modules...")
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame

ROOT = os.path.dirname(__file__)

# Allows us to log our RTCPeerConnection
logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

print("Starting Model Inference...")
# Load the local PyTorch model
model = YOLO('best.pt')
debugresults = []
class VideoTransformTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, track, transform):
        global debugresults
        super().__init__()
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()
        w = frame.width
        h = frame.height
        img = frame.to_ndarray(format="bgr24") # Convert the raw data to inferrable N-Dimensional array
        results = model(source=img, conf=0.25, iou=0.5)

        for detection in results: # Per Frame
            debugresults.append(detection)
            maskplatforms = np.zeros((h, w), dtype=np.uint8)
            masktracks = np.zeros((h, w), dtype=np.uint8)
            
            mspd = [0,0]
            mstd = [0,0]
            
            for i in range(0,detection.boxes.cls.size()[0]): # Per Detection in Frame
                if(detection.boxes.cls[i].item() == 0.0):
                    print("Platform Detected")
                    ccp = detection.boxes.conf[i].item()
                    if(ccp > mspd[1]):
                        mspd[0] = i 
                        mspd[1] = ccp
                        
                if(detection.boxes.cls[i].item() == 1.0):
                    print("Track Detected")
                    cct = detection.boxes.conf[i].item()
                    if(cct > mstd[1]):
                        mstd[0] = i
                        mstd[1] = cct
                
            if(mspd != [0,0]):
                points = np.array(detection.masks[mspd[0]].xy[0],dtype='int32')
                cv2.fillPoly(maskplatforms, [points], 255)

            if(mstd != [0,0]):
                points = np.array(detection.masks[mstd[0]].xy[0],dtype='int32')
                cv2.fillPoly(masktracks, [points], 255)

            if (mspd != [0,0] and mstd != [0,0]):

                # Combine our masks together
                maskplatforms = cv2.threshold(maskplatforms, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                masktracks = cv2.threshold(masktracks, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                img[maskplatforms == 255] = (255, 255, 0) # Will display as yellow
                img[masktracks == 255] = (255, 0, 255) # Will display as blue
                
                detected = '' # Right or left of the camera?
                
                # If the track is to the right of the platform...
                if(detection.masks[mspd[0]].xyn[0][0][0] < detection.masks[mstd[0]].xyn[0][0][0]): 
                    detected = 'R' # We'll set 'detected' to 'R'

                # If the track is to the left of the platform...   
                else:
                    detected = 'L' # We'll set 'detected' to 'L'
                
                if dc != None: # as long as the data channel is set up...
                    if detected == 'L':
                        dc.send("playLeft") # Send the playLeft message to the datachannel
                        
                    if detected == 'R':
                        dc.send("playRight") # Or send the playRight message to the datachannel
                
            # Convert our N-d array back to image format with all the masking added and return as a new frame.
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame


# These functions allow us to serve elements of the media server as URL paths...

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

# For MP3s, we need to interpret as binary and send that way:
async def leftFile(request):
    filepath = os.path.join(ROOT, "LeftDetection.mp3")
    with open(filepath, "rb") as file:
        content = file.read()
    return web.Response(body=content, content_type='audio/mpeg')

async def rightFile(request):
    filepath = os.path.join(ROOT, "RightDetection.mp3")
    with open(filepath, "rb") as file:
        content = file.read()
    return web.Response(body=content, content_type='audio/mpeg')


# This is where we set up peer connections between the host and clients
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    
    pcs.add(pc)

    # Some logging methods
    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    recorder = MediaBlackhole()

    # When the data channel is detected, we'll send a message to the JS console
    @pc.on("datachannel")
    def on_datachannel(channel):
        global dc # We also want to make sure datachannel is global...
        # ...so the VideoStreamTrack class can send audio info
        
        dc = channel
        channel.send("setGlobal")
        
    # Event logging methods...
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)
        
        if track.kind == "video":

            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track),
                    transform=params["video_transform"]
                )
            )

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)

    # Handling the 'offer'
    await pc.setRemoteDescription(offer)

    # Send Answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Return our peer connection
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )



async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

# Set up arguments for running this script.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", default="cert.pem", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", default="key.pem", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=443, help="Port for HTTP server (default: 8080)"
    )
    args = parser.parse_args()


    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    # This links to our earlier serving methods...
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/left",leftFile)
    app.router.add_get("/right",rightFile)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)

    web.run_app(
        app, access_log=None, host="0.0.0.0", port=443, ssl_context=ssl_context
    )
