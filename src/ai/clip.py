import torch
from PIL import Image
from rich import print
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")


image = Image.open(R"E:\pictoria-server\demo\9c34d98c7242c2b174fa0f7617f1d736.jpg")

texts = ["high-quality art", "low-quality art"]
# important: we pass `padding=max_length` since the model was trained with this
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
for prob, label in zip(probs[0], texts, strict=False):
    print(f"{label}: {prob:.2f}")
