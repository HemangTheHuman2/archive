import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import numpy as np

# Load model once
print("Loading BiRefNet...")
model = AutoModelForImageSegmentation.from_pretrained(
    'ZhengPeng7/BiRefNet', 
    trust_remote_code=True
)
model.eval()
if torch.cuda.is_available():
    model = model.cuda().half()
    print("Using GPU with half precision")
else:
    print("Using CPU")

def process_image(input_image):
    """Process image with BiRefNet"""
    # Preprocess
    image_size = (1024, 1024)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    original_size = input_image.size
    input_tensor = transform(input_image).unsqueeze(0)
    
    # Move to device
    device = next(model.parameters()).device
    if device.type == 'cuda':
        input_tensor = input_tensor.half()
    
    input_tensor = input_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    # Process mask
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(original_size)
    
    # Create masked image
    result = input_image.copy()
    result.putalpha(mask)
    
    return mask, result

# Create Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=[
        gr.Image(type="pil", label="Segmentation Mask"),
        gr.Image(type="pil", label="Extracted Object")
    ],
    title="BiRefNet - Object Segmentation",
    description="Upload an image to extract the main object using BiRefNet",
    examples=[
        ["https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg"]
    ]
)

if __name__ == "__main__":
    iface.launch(share=True)  # Set share=True for temporary public link