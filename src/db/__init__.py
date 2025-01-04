import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec
from pydantic import BaseModel

import shared
from ai.clip import calculate_image_features
from models import Post

INIT_SQL = """CREATE VIRTUAL TABLE IF NOT EXISTS post_vecs USING vec0(
  post_id INTEGER PRIMARY KEY,
  embedding FLOAT[768])"""


def get_vec_db():
    db = sqlite3.connect(shared.vec_path)
    db.enable_load_extension(True)  # noqa: FBT003
    sqlite_vec.load(db)
    db.enable_load_extension(False)  # noqa: FBT003
    return db


def init_vec_db():
    db = get_vec_db()
    cursor = db.cursor()
    shared.logger.info("Initializing post_vecs table")
    cursor.execute(INIT_SQL)
    db.commit()
    cursor.close()


def insert_img_vec(post_id: int, image_path: Path):
    features = calculate_image_features(image_path)
    features_np = features.cpu().numpy()
    upsert_query = """INSERT INTO post_vecs (post_id, embedding) VALUES (:post_id, :embedding)"""
    db = get_vec_db()
    cursor = db.cursor()
    cursor.execute(upsert_query, {"post_id": post_id, "embedding": features_np})
    db.commit()
    cursor.close()
    return features_np


def get_img_vec(post: Post):
    post_id = post.id
    query = """SELECT embedding FROM post_vecs WHERE post_id = :post_id"""
    db = get_vec_db()
    cursor = db.cursor()
    cursor.execute(query, {"post_id": post_id})
    result = cursor.fetchone()
    if result is None:
        return insert_img_vec(post_id, post.absolute_path)
    cursor.close()
    return result[0]


class SimilarImageResult(BaseModel):
    post_id: int
    distance: float


def find_similar_posts(vec: np.ndarray, *, limit: int = 10) -> list[SimilarImageResult]:
    query = """
        SELECT post_id, distance
        FROM post_vecs
        WHERE embedding MATCH :embedding and k = :limit;
        """
    db = get_vec_db()

    cursor = db.cursor()
    cursor.execute(query, {"embedding": vec, "limit": limit + 1})
    result = cursor.fetchall()
    return [SimilarImageResult(post_id=row[0], distance=row[1]) for row in result[1:]]
