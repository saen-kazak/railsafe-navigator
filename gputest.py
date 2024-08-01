import onnxruntime as ort

def check_gpu():
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        print("GPU is available for ONNX Runtime.")
    else:
        print("GPU is not available for ONNX Runtime, using CPU instead.")

check_gpu()
