# Import the modules we need

print("Importing OpenCV.")
import cv2
print("Loaded OpenCV.")

print("Importing Roboflow Inference...")
from inference.models.utils import get_roboflow_model
print("Loaded Roboflow Inference...")

print("Loading other modules...")
import numpy as np
import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame
print("Loaded other modules.")



ROOT = os.path.dirname(__file__)

# Allows us to log our RTCPeerConnection
logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

# Cache the model from Roboflow Servers

model_name = "railway-instancesegmentation"
model_version = "7"
dc = None

model = get_roboflow_model(
    model_id="{}/{}".format(model_name, model_version),
    api_key="G1hyD71ORZ16FcNt6cKL"
)

# We define a class that will take our frames and infer them on the recv() method
class VideoTransformTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, track, transform):
        super().__init__()
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()
        w = frame.width
        h = frame.height
        img = frame.to_ndarray(format="bgr24") # Convert the raw data to inferrable N-Dimensional array

        # Infer the model        
        result = model.infer(
            image=img,
            confidence=0.25,
            iou_threshold=0.5
        )

        # Grab response object
        response = result[0]
        
        image_width, image_height = img.shape[1], img.shape[0]

        # Create individual masks for tracks and platforms in the same format
        maskplatforms = np.zeros((image_height, image_width), dtype=np.uint8)
        masktracks = np.zeros((image_height, image_width), dtype=np.uint8)

        # Initialise for 'Most Significant Platform/Track Detection'
        mspd = [0,0]
        mstd = [0,0]

        # Dig through the response data
        for prediction in response.predictions:

            
            if(prediction.class_name == "platform"):
                # CCP =  Current Confidence, Platform.
                ccp = prediction.confidence 
                if(ccp > mspd[1]): # If it's a more significant detection than the others in the response object...
                    mspd[0] = prediction 
                    mspd[1] = ccp
                    # ...we'll set it to be the one we display

            # 
            if(prediction.class_name == "track"):
                # CCP =  Current Confidence, Track
                cct = prediction.confidence
                if(cct > mstd[1]): # If it's a more significant detection than the others in the response object...
                    mstd[0] = prediction
                    mstd[1] = cct
                    # ...we'll set it to be the one we display

        
        if(mspd != [0,0]): # If we actually detected platforms in this frame...   
            points = mspd[0].points # Get the most significant track's polygon points
            
            # Draw the shape on our platform mask
            pts = np.array([[point.x, point.y] for point in points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(maskplatforms, [pts], 255)
            
        if(mstd != [0,0]): # If we actually detected platforms in this frame...  
            points = mstd[0].points # Get the most significant track's polygon points

            # Draw the shape on our track mask
            pts = np.array([[point.x, point.y] for point in points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(masktracks, [pts], 255)

        # As long as we have BOTH a platform and a track detected
        # (we need both for the audio)
        if (mspd != [0,0] and mstd != [0,0]):

            # Combine our masks together
            maskplatforms = cv2.threshold(maskplatforms, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            masktracks = cv2.threshold(masktracks, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            img[maskplatforms == 255] = (255, 255, 0) # Will display as yellow
            img[masktracks == 255] = (255, 0, 255) # Will display as blue

            detected = '' # Right or left of the camera?

            # If the track is to the right of the platform...
            if(mspd[0].points[0].x < mstd[0].points[0].x): 
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
