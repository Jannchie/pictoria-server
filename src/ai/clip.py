from functools import cache
from pathlib import Path

import torch
from PIL import Image
from rich import print
from transformers import AutoModel, AutoProcessor

device = "cuda"


@cache
def get_clip_model() -> AutoModel:
    return AutoModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        device_map=device,
    )


@cache
def get_processor() -> AutoProcessor:
    return AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")


def calculate_image_features(image: Image.Image | Path | str) -> torch.Tensor:
    if isinstance(image, Path | str):
        image = Image.open(image)
    model = get_clip_model()
    processor = get_processor()
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        return model.get_image_features(pixel_values=inputs.pixel_values)


if __name__ == "__main__":
    model = get_clip_model()
    image = Image.open(R"E:\pictoria\server\demo\9c34d98c7242c2b174fa0f7617f1d736.jpg")

    texts = ["high-quality art", "low-quality art"]
    # important: we pass `padding=max_length` since the model was trained with this
    inputs = get_processor()(text=texts, images=image, return_tensors="pt", padding=True)
    print(inputs.pixel_values.shape)
    intputs = inputs.to(device)

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs.pixel_values)
        text_features = model.get_text_features(input_ids=inputs.input_ids)
        print(image_features.shape)
        print(text_features.shape)
        print(model(**inputs))

    # with torch.no_grad():
    #     outputs = model(**inputs)

    # outputs = model(**inputs)
    # logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    # probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    # for prob, label in zip(probs[0], texts, strict=False):
    #     print(f"{label}: {prob:.2f}")
