import concurrent.futures
import json
import pathlib

from openai import OpenAI
from pydantic import BaseModel
from rich.progress import track

with pathlib.Path("data/tag_group_gt_100.json").open("r", encoding="utf-8") as file:
    tag_data = json.load(file)
tags = set()
for tag_group in tag_data:
    for tag in tag_data[tag_group]:
        if tag:
            tags.add(tag.replace("_", " "))

en_data = {tag: tag for tag in tags}

with pathlib.Path("data/tag.en.json").open("w", encoding="utf-8") as file:
    json.dump(en_data, file, ensure_ascii=False)

client = OpenAI()


class TranslateItem(BaseModel):
    source_tag: str
    target_tag: str


class TranslateResult(BaseModel):
    items: list[TranslateItem]
    source_language: str
    target_language: str


tag_list = list(tags)
chunk_size = 16


zh_data = {}


def translate_chunk(chunk: list[str]) -> dict[str, str]:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "将这些标签翻译成中文。这些标签可能包含某些游戏、漫画、动画等作品的人物、组织、实体名称，请尽量识别并使用正式翻译名称",  # noqa: E501, RUF001
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "\n".join(chunk)}],
            },
        ],
        response_format=TranslateResult,
    )
    resp: TranslateResult = completion.choices[0].message.parsed
    return {item.source_tag: item.target_tag for item in resp.items}


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(translate_chunk, tag_list[i : i + chunk_size]) for i in range(0, len(tag_list), chunk_size)
    ]
    for future in track(concurrent.futures.as_completed(futures), total=len(futures)):
        zh_data.update(future.result())

with pathlib.Path("data/tag.zh-Hans.json").open("w", encoding="utf-8") as file:
    json.dump(zh_data, file, ensure_ascii=False)
