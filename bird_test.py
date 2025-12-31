from PIL import Image
import time
import torch
import requests
from io import BytesIO
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2", use_fast=True)
model = AutoModelForImageClassification.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")


start = time.time()
# For demonstration, we'll use an image from a URL.
# If you have a local file, replace this with:
# image = Image.open('path/to/your/image.jpg')
image_url = "https://www.natureswaybirds.com/cdn/shop/articles/eastern_bluebird_750x.png?v=1713204293"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Convert image to RGB format to ensure it has 3 channels
image = image.convert("RGB")

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Make a prediction
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class probabilities
logits = outputs.logits
probabilities = torch.softmax(logits, dim=-1)

# Get the top 5 predicted classes and their probabilities
top5_prob, top5_indices = torch.topk(probabilities, 5)
end = time.time()
length = end - start

print("It took", length, "seconds!")
print("Top 5 Predictions:")
for i in range(top5_prob.size(1)):
    label = model.config.id2label[top5_indices[0, i].item()]
    score = top5_prob[0, i].item()
    print(f"  Label: {label}, Score: {score:.4f}")
