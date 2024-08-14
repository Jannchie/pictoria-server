import base64
import json
from io import BytesIO

from diffusers.utils.loading_utils import load_image
from openai import OpenAI
from rich import get_console
from wdtagger import Tagger

console = get_console()


class BaseAnnotator:
    def annotate_image(self, image_path):
        raise NotImplementedError(f"Annotating the image at: {image_path}")


class WDTaggerAnnotator(BaseAnnotator):
    def __init__(self):
        self.tagger = Tagger()

    def annotate_image(self, image_path):
        image = load_image(image_path)
        tagger_resp = self.tagger.tag(image)
        return tagger_resp.all_tags_string


class OpenAIImageAnnotator(BaseAnnotator):
    MODEL = "gpt-4o-mini"

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def load_and_process_image(self, image_path):
        return load_image(image_path)

    def get_img_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")  # Change format as needed
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def calculate_cost(self, response):
        usage = response.usage
        price_per_token = {
            "gpt-4o-mini": {
                "prompt_tokens": 0.15 / 1_000_000,
                "response_tokens": 0.6 / 1_000_000,
            },
            "gpt-4o": {
                "prompt_tokens": 5 / 1_000_000,
                "response_tokens": 15 / 1_000_000,
            },
        }
        return (
            usage.prompt_tokens * price_per_token[self.MODEL]["prompt_tokens"]
            + usage.completion_tokens * price_per_token[self.MODEL]["response_tokens"]
        )

    def annotate_image(self, image_path) -> dict:
        image = self.load_and_process_image(image_path)
        base64_image = self.get_img_base64(image)

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Create a caption for the image "
                                "using natural language. Your tags and caption need to be as specific as possible and "
                                "should be distinguishable from other images. Not only should you include the content "
                                "and elements, but you should also cover aspects such as composition, art style, type, "
                                "color, structure, etc. I will use your output to label training data. The response "
                                "should be json format, with the key 'caption'."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
        )

        content = response.choices[0].message.content
        return json.loads(content)


if __name__ == "__main__":
    api_key = ""
    image_path = R"demo/1.jpg"
    annotator = OpenAIImageAnnotator(api_key)
    annotations = annotator.annotate_image(image_path)
    console.log(annotations)
