import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, CLIPForImageClassification

dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPForImageClassification.from_pretrained("openai/clip-vit-large-patch14")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
print(model.config.id2label)
# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
