var dataChannelLog = document.getElementById('data-channel'),
    iceConnectionLog = document.getElementById('ice-connection-state'),
    iceGatheringLog = document.getElementById('ice-gathering-state'),
    signalingLog = document.getElementById('signaling-state');

var pc = null;

var dc = null, dcInterval = null;

function createPeerConnection() {
    var config = {
        
    };

    pc = new RTCPeerConnection(config);

    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video')
            document.getElementById('video').srcObject = evt.streams[0];
        else
            document.getElementById('audio').srcObject = evt.streams[0];
    });

    return pc;
}


function negotiate() {
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {

        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        var offer = pc.localDescription;
        var codec;

        codec = "H264";
		
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                video_transform: "1024x1024",
				
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        return response.json();
    }).then((answer) => {
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        alert(e);
    });
}
error = "";

function start() {
    pc = createPeerConnection();
	
    const constraints = {
        audio: false,
        video: false
    };
	
	dc = pc.createDataChannel('InferenceResult');
        dc.addEventListener('close', () => {
            clearInterval(dcInterval);
            
        });
		
        dc.addEventListener('open', () => {
			console.log("Channel opened")
        });
		
		dc.addEventListener('message', (evt) => {
			if(evt.data === 'setGlobal'){
				console.log("channel is global");
			}
			
			if(evt.data === 'playLeft'){
				document.getElementById("detectLeft").play()
			}

			if(evt.data === 'playRight'){
				document.getElementById("detectRight").play()
			}
				
		});

	
	navigator.mediaDevices.enumerateDevices()
		.then((devices) => {
			
			const videoDevices = devices.filter(device => device.kind === 'videoinput');
			if (videoDevices.length === 0) {
				throw new Error('No video input devices found');
			}
		
            const lastVideoDeviceId = videoDevices[videoDevices.length - 1].deviceId;
			
			const videoConstraints = {}
			videoConstraints.facingMode = {ideal:"environment"}
			videoConstraints.deviceId = { ideal: lastVideoDeviceId}
			
			const resolution = "1024x1024"
			const dimensions = resolution.split('x');
			videoConstraints.height = {ideal:640}
			videoConstraints.frameRate = {exact:10}
			videoConstraints.width = {ideal: 640}
			
			
			constraints.video = Object.keys(videoConstraints).length ? videoConstraints : true;
			
			document.getElementById('media').style.display = 'block';
			return navigator.mediaDevices.getUserMedia(constraints);
		})
		.then((stream) => {
			stream.getTracks().forEach((track) => {
				pc.addTrack(track, stream);
			});
			return negotiate();
		})
		.catch((err) => {
            alert('Could not acquire media: ' + err);
			error = err
		});
		
}

function stop() {
    document.getElementById('stop').style.display = 'none';

    if (dc) {
        dc.close();
    }

    if (pc.getTransceivers) {
        pc.getTransceivers().forEach((transceiver) => {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });
    }

    pc.getSenders().forEach((sender) => {
        sender.track.stop();
    });

    setTimeout(() => {
        pc.close();
    }, 500);
}

function sdpFilterCodec(kind, codec, realSdp) {
    var allowed = []
    var rtxRegex = new RegExp('a=fmtp:(\\d+) apt=(\\d+)\r$');
    var codecRegex = new RegExp('a=rtpmap:([0-9]+) ' + escapeRegExp(codec))
    var videoRegex = new RegExp('(m=' + kind + ' .*?)( ([0-9]+))*\\s*$')

    var lines = realSdp.split('\n');

    var isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var match = lines[i].match(codecRegex);
            if (match) {
                allowed.push(parseInt(match[1]));
            }

            match = lines[i].match(rtxRegex);
            if (match && allowed.includes(parseInt(match[2]))) {
                allowed.push(parseInt(match[1]));
            }
        }
    }

    var skipRegex = 'a=(fmtp|rtcp-fb|rtpmap):([0-9]+)';
    var sdp = '';

    isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var skipMatch = lines[i].match(skipRegex);
            if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
                continue;
            } else if (lines[i].match(videoRegex)) {
                sdp += lines[i].replace(videoRegex, '$1 ' + allowed.join(' ')) + '\n';
            } else {
                sdp += lines[i] + '\n';
            }
        } else {
            sdp += lines[i] + '\n';
        }
    }

    return sdp;
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}


function getDeviceList(){
	result = new Array();
	navigator.mediaDevices.enumerateDevices()
    .then((devices) => {
      devices.forEach((device) => {
		  if (device.kind == 'videoinput'){
			  result.push(device.deviceId)
		  }
        
      })
	})
	.then((devices) => {
		return r[devices.length-1].deviceId;
	})
    .catch((err) => {
    });
}

start();