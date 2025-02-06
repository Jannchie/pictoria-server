import concurrent.futures
import os
from datetime import datetime
from logging import getLogger
from pathlib import Path
from threading import Thread
from types import TracebackType

import httpx
from pydantic import BaseModel, HttpUrl

logger = getLogger("danbooru")


class Variant(BaseModel):
    type: str
    url: HttpUrl
    width: int
    height: int
    file_ext: str


class MediaAsset(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    md5: str | None = None
    file_ext: str
    file_size: int
    image_width: int
    image_height: int
    duration: int | float | None = None
    status: str
    file_key: str | None = None
    is_public: bool
    pixel_hash: str
    variants: list[Variant] | None = None


class DanbooruPost(BaseModel):
    id: int
    created_at: datetime
    uploader_id: int
    score: int
    source: str | None = None
    md5: str | None = None
    last_comment_bumped_at: datetime | None = None
    rating: str
    image_width: int
    image_height: int
    tag_string: str
    fav_count: int
    file_ext: str
    last_noted_at: datetime | None = None
    parent_id: int | None = None
    has_children: bool
    approver_id: int | None = None
    tag_count_general: int
    tag_count_artist: int
    tag_count_character: int
    tag_count_copyright: int
    file_size: int
    up_score: int
    down_score: int
    is_pending: bool
    is_flagged: bool
    is_deleted: bool
    tag_count: int
    updated_at: datetime
    is_banned: bool
    pixiv_id: int | None = None
    last_commented_at: datetime | None = None
    has_active_children: bool
    bit_flags: int
    tag_count_meta: int
    has_large: bool
    has_visible_children: bool
    media_asset: MediaAsset
    tag_string_general: str
    tag_string_character: str
    tag_string_copyright: str
    tag_string_artist: str
    tag_string_meta: str
    file_url: HttpUrl | None = None
    large_file_url: HttpUrl | None = None
    preview_file_url: HttpUrl | None = None


class DanbooruClient:
    def __init__(self, api_key: str, user_id: str, base_url: str = "https://danbooru.donmai.us") -> None:
        self.api_key: str = api_key
        self.user_id: str = user_id
        self.base_url: str = base_url

    def get_post(self, post_id: int) -> DanbooruPost:
        url: str = f"{self.base_url}/posts/{post_id}.json"
        response = httpx.get(url, params={"api_key": self.api_key, "login": self.user_id})
        response.raise_for_status()
        return DanbooruPost(**response.json())

    def get_posts(  # noqa: PLR0913
        self,
        key: str | None = None,
        value: str | int | None = None,
        tags: str | None = None,
        limit: int = 10,
        before_id: int | None = None,
        only: str | list[str] | None = None,
    ) -> list[DanbooruPost]:
        url: str = f"{self.base_url}/posts.json"
        only_str: str | None = ",".join(only) if isinstance(only, list) else only
        all_posts: list[dict] = []

        while True:
            current_limit: int = min(limit - len(all_posts), 200)
            if current_limit <= 0:
                break

            params: dict = {
                f"{key}": value if key else None,
                "limit": current_limit,
                "api_key": self.api_key,
                "login": self.user_id,
                "page": f"b{before_id}" if before_id else None,
                "tags": tags,
                "only": only_str,
            }
            params = {k: v for k, v in params.items() if v is not None}

            response = httpx.get(url, params=params)
            logger.debug(response.url)
            response.raise_for_status()

            posts: list[dict] = response.json()
            if not posts:
                break

            all_posts.extend(posts)
            before_id = min(post["id"] for post in posts)
            if len(all_posts) >= limit:
                break
        res = []
        for post in all_posts:
            try:
                res.append(DanbooruPost(**post))
            except Exception:
                logger.exception(post)
                logger.exception("Failed to parse posts")
        return res

    def download_image(self, post: DanbooruPost, target_dir: str, retries: int = 3) -> None:
        if post.file_url is None:
            return
        url = str(post.file_url)
        post_id: int = post.id
        for attempt in range(retries):
            try:
                logger.debug("Downloading post %s, attempt %d", post.id, attempt + 1)
                ext = post.file_ext
                file_path = Path(target_dir) / f"{post_id}.{ext}"
                if file_path.exists():
                    logger.debug("File %s already exists, skipping", file_path)
                    return
                with httpx.stream("GET", url) as response:
                    response.raise_for_status()
                    with file_path.open("wb") as f:
                        for chunk in response.iter_bytes():
                            f.write(chunk)
            except Exception:
                logger.error("Failed to download post %s on attempt %d", post_id, attempt + 1)
            else:
                return
        logger.exception("All attempts to download post %s have failed", post_id)

    def download_by_id(self, post_id: int, target_dir: str) -> None:
        post: dict = self.get_post(post_id)
        if not post:
            logger.warning("Post %s not found", post_id)
            return
        Thread(target=self.download_image, args=(post, target_dir)).start()

    def download_posts(self, posts: list[DanbooruPost], target_dir: os.PathLike, n_worker: int = 16) -> None:
        target_dir = Path(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)
        logger.info("Download started!")
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_worker) as executor:
            for post in posts:
                logger.debug("Downloading post %s", post.id)
                executor.submit(self.download_image, post, target_dir)
        logger.info("Download completed!")


class Downloader:
    def __init__(self, n_workers: int = 4) -> None:
        self.n_workers: int = n_workers
        self.executor: concurrent.futures.ThreadPoolExecutor | None = None

    def __enter__(self) -> "Downloader":
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)

    def download(self, url: str, target_path: os.PathLike) -> concurrent.futures.Future:
        target_path: Path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        return self.executor.submit(self._download_single, url, target_path)

    def _download_single(self, url: str, target_path: Path) -> None:
        try:
            response = httpx.get(url)
            response.raise_for_status()
            with target_path.open("wb") as file:
                file.write(response.content)
            logger.debug("Downloaded {url}")
        except Exception:
            logger.exception("Failed to download {url}")
