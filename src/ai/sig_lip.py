import pandas as pd
import torch
from PIL import Image
from rich import print
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to("cuda")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

image = Image.open(R"E:\pictoria-server\demo\收藏量1409画师wez作品80360838_p0.jpg")

styles = [
    "pixel",
    "realistic",
    "cartoon",
    "comic",
    "anime",
    "webtoon",
    "watercolor",
    "photography",
    "sketch",
    "retro",
]
targets = ["animal", "girl", "boy", "man", "woman", "loli", "shota", "monster", "robot", "landscape", "cityscape"]


texts = [f"A {style} illustration of a {target}" for style in styles for target in targets]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
print(logits_per_image)
probs_sigmoid = torch.sigmoid(logits_per_image)  # these are the probabilities
probs_softmax = torch.softmax(logits_per_image, dim=1)  # these are the probabilities

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

dataframe = pd.DataFrame(
    {
        "text": texts,
        "softmax": probs_softmax[0].cpu().numpy(),
        "sigmoid": probs_sigmoid[0].cpu().numpy(),
    },
)

# format softmax and sigmoid probabilities to 2 decimal places
dataframe["softmax"] = dataframe["softmax"].apply(lambda x: round(x, 2))
dataframe["sigmoid"] = dataframe["sigmoid"].apply(lambda x: round(x, 2))
# show all rows
pd.set_option("display.max_rows", None)
print(dataframe)

# for prob_softmax, prob_sigmoid, label in zip(probs_softmax[0], probs_sigmoid[0], texts):
#     print(f"{label}: {prob_softmax:.2f} (softmax), {prob_sigmoid:.2f} (sigmoid)")
#     print(f"{label}: {prob_softmax:.2f} (softmax), {prob_sigmoid:.2f} (sigmoid)")
#     print(f"{label}: {prob_softmax:.2f} (softmax), {prob_sigmoid:.2f} (sigmoid)")
