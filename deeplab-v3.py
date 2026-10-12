# pip install torch torchvision opencv-python

import torch
import torchvision.transforms as T
import cv2
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101

model = deeplabv3_resnet101(pretrained=True).eval()

img = cv2.imread("test-img.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

input_tensor = transform(img_rgb).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)["out"][0]

mask = output.argmax(0).byte().cpu().numpy()

# Class 0 = background, 7 = road (approx for COCO-style)
road_mask = (mask == 7).astype(np.uint8) * 255

cv2.imshow("Road Mask", road_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
