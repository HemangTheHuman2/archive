import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import requests
from io import BytesIO
import numpy as np
import cv2

def load_birefnet():
    """Load BiRefNet model from HuggingFace"""
    try:
        from transformers import AutoModelForImageSegmentation
        print("Loading BiRefNet from HuggingFace...")
        model = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet', 
            trust_remote_code=True
        )
        return model
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("\nTrying alternative method...")
        return load_birefnet_alternative()

def load_birefnet_alternative():
    """Alternative loading method from GitHub"""
    try:
        # Clone the repository first
        import subprocess
        import sys
        import os
        
        print("Cloning BiRefNet repository...")
        subprocess.run(['git', 'clone', 'https://github.com/ZhengPeng7/BiRefNet.git'], 
                      check=True, capture_output=True)
        
        # Add to Python path
        sys.path.append('./BiRefNet')
        
        from models.birefnet import BiRefNet
        print("Loading model weights from HuggingFace...")
        model = BiRefNet.from_pretrained('ZhengPeng7/BiRefNet')
        return model
    except Exception as e:
        print(f"Error with alternative method: {e}")
        return None

def download_test_image():
    """Download a test image if no local image is provided"""
    print("Downloading test image...")
    url = "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def preprocess_image(image, image_size=(1024, 1024)):
    """Preprocess image for BiRefNet"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, str):
        # If input is a file path
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        # If input is numpy array (OpenCV)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor, original_size, image

def extract_object(model, image_path=None, image=None):
    """Extract object using BiRefNet"""
    # Load or use provided image
    if image_path:
        image = Image.open(image_path).convert('RGB')
    elif image is None:
        image = download_test_image()
    
    # Preprocess
    input_tensor, original_size, pil_image = preprocess_image(image)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    model.eval()
    
    if device.type == 'cuda':
        model = model.half()  # Use half precision for GPU
        input_tensor = input_tensor.half()
    
    input_tensor = input_tensor.to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    # Process output
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    
    # Resize mask to original image size
    mask = pred_pil.resize(original_size)
    
    # Create masked image (RGBA)
    rgba_image = pil_image.copy()
    rgba_image.putalpha(mask)
    
    return rgba_image, mask, pil_image

def display_results(original, mask, result):
    """Display results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    axes[2].imshow(result)
    axes[2].set_title('Extracted Object')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_results(result, mask, output_dir='./outputs'):
    """Save results to files"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    result.save(f"{output_dir}/extracted_object.png")
    mask.save(f"{output_dir}/mask.png")
    print(f"Results saved to {output_dir}/")

def simple_test():
    """Simple test function"""
    print("="*60)
    print("BiRefNet Simple Test Script")
    print("="*60)
    
    # Load model
    model = load_birefnet()
    if model is None:
        print("Failed to load model. Please check the installation.")
        return
    
    # Test with a sample image
    print("\nTesting with sample image...")
    
    # Option 1: Use local image
    # image_path = 'path/to/your/image.jpg'  # Uncomment and set your image path
    
    # Option 2: Use downloaded test image
    image_path = None
    
    try:
        result, mask, original = extract_object(model, image_path)
        
        # Display results
        display_results(original, mask, result)
        
        # Save results
        save_results(result, mask)
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

def webcam_test():
    """Test with webcam feed"""
    print("Starting webcam test...")
    print("Press 'q' to quit, 's' to save current frame")
    
    model = load_birefnet()
    if model is None:
        return
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        try:
            # Extract object
            result, mask, _ = extract_object(model, image=pil_image)
            
            # Convert result back to OpenCV format
            result_np = np.array(result)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGBA2BGRA)
            
            # Display
            cv2.imshow('BiRefNet - Webcam', result_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('webcam_capture.png', result_bgr)
                print("Frame saved as 'webcam_capture.png'")
                
        except Exception as e:
            print(f"Error: {e}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run simple test
    simple_test()
    
    # Uncomment to test with webcam
    # webcam_test()