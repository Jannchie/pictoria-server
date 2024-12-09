import base64
from io import BytesIO
from pathlib import Path

import PIL.Image
from diffusers.utils.loading_utils import load_image
from openai import OpenAI
from rich import get_console
from wdtagger import Tagger

from utils import timer

console = get_console()


class BaseAnnotator:
    def annotate_image(self, image_path: Path) -> str:
        msg = f"Annotating the image at: {image_path}"
        raise NotImplementedError(msg)


class WDTaggerAnnotator(BaseAnnotator):
    def __init__(self) -> None:
        self.tagger = Tagger()

    def annotate_image(self, image_path: Path) -> str:
        image = load_image(image_path.as_posix())
        tagger_resp = self.tagger.tag(image)
        return tagger_resp.all_tags_string


class OpenAIImageAnnotator(BaseAnnotator):
    MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

    def load_and_process_image(self, image_path: Path) -> PIL.Image.Image:
        return load_image(image_path.as_posix())

    def get_img_base64(self, image: PIL.Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")  # Change format as needed
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @timer
    def annotate_image(self, image_path: Path) -> str:
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
                                "Create a caption for the image. "
                                "You must use simple natural language. Your caption need to be as specific as possible"
                                "and should be distinguishable from other images. You should include all the content "
                                "and elements in the image."
                                "The caption should shorter than 60 words."
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
                },
            ],
            max_tokens=200,
        )
        if response.choices[0].message.content is None:
            return ""
        return response.choices[0].message.content


if __name__ == "__main__":
    api_key = ""
    image_path = R"demo/1.jpg"
    annotator = OpenAIImageAnnotator(api_key)
    annotations = annotator.annotate_image(image_path)
    console.log(annotations)
