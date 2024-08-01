import torch
from ultralytics import YOLO
import torchvision.models as models

# Example: Load a pre-trained model
model = YOLO('best.pt')
model.eval()
# Dummy input for the model
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "model.onnx", export_params=True, opset_version=11, do_constant_folding=True, input_names=['input'], output_names=['output'])
