import os
import json
from pathlib import Path
from io import BytesIO

import httpx
import numpy as np
from PIL import Image
import torch
import open_clip
import chromadb
from chromadb import Settings
from dotenv import load_dotenv

# Load .env
load_dotenv()

DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"D:\Exhibit"))
JSON_ROOT = DATA_ROOT / "json_extracted"
IMG_ROOT = DATA_ROOT / "image_extracted"

CATEGORY_LIST = [
    "painting_json",
    "craft_json",
    "sculpture_json"
]

CATEGORY_MAP = {
    "painting_json": "TL_01. 2D_02.회화(Json)",
    "craft_json":    "TL_01. 2D_04.공예(Json)",
    "sculpture_json": "TL_01. 2D_06.조각(Json)",
}

# -----------------------------
# Chroma 클라이언트
# -----------------------------
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False),
)

collection = client.get_or_create_collection(
    name="curator_image_clip",
    metadata={"hnsw:space": "cosine"},
)

# -----------------------------
# CLIP 모델 불러오기
# -----------------------------
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME,
    pretrained=CLIP_PRETRAINED
)
model = model.to(device)
model.eval()

@torch.no_grad()
def embed_image(img: Image.Image) -> np.ndarray:
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    feats = model.encode_image(img_tensor)
    feats /= feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]


# -----------------------------
# 이미지 파일 찾기
# -----------------------------
def find_image_path(art_id: str) -> str | None:
    exts = ["jpg", "jpeg", "png", "JPG", "PNG"]
    for ext in exts:
        files = list(IMG_ROOT.rglob(f"{art_id}*.{ext}"))
        if files:
            rel = files[0].relative_to(IMG_ROOT)
            return f"/image_extracted/{rel.as_posix()}"
    return None


# -----------------------------
# 카드(JSON)에서 메타 데이터 추출
# -----------------------------
def extract_meta(card: dict) -> dict:
    desc = card.get("Description") or {}
    info = card.get("Data_Info") or {}
    photo = card.get("Photo_Info") or {}

    title = desc.get("ArtTitle_kor") or card.get("title") or info.get("ImageFileName")
    artist = desc.get("ArtistName_kor") or card.get("artist")
    klass = desc.get("Class_kor") or card.get("class")
    material = desc.get("Material_kor") or card.get("material")
    year = desc.get("Date") or photo.get("PhotoDate")

    return {
        "title": title or "",
        "artist": artist or "",
        "class": klass or "",
        "material": material or "",
        "year": year or "",
    }


# -----------------------------
# 메인 인덱싱 루프
# -----------------------------
def main():
    print("🔍 Building CLIP image index...")
    added = 0

    for category in CATEGORY_LIST:
        real_folder = CATEGORY_MAP.get(category)
        target_dir = JSON_ROOT / real_folder

        if not target_dir.exists():
            print(f"⚠️  Missing folder: {target_dir}")
            continue

        json_files = list(target_dir.glob("*.json"))
        print(f"📁 {category} → {len(json_files)} JSON found")

        for js in json_files:
            art_id = js.stem

            # 이미지 경로 찾기
            image_path = find_image_path(art_id)
            if not image_path:
                print(f"   ⚠️ No image for {art_id}, skip")
                continue

            try:
                with open(js, "r", encoding="utf-8") as f:
                    card = json.load(f)
            except Exception as e:
                print(f"   ⚠️ JSON read fail {art_id}: {e}")
                continue

            meta = extract_meta(card)
            meta["category"] = category
            meta["image_path"] = image_path
            meta["id"] = art_id

            # 이미지 로드
            full_img_path = IMG_ROOT / Path(image_path.replace("/image_extracted/", ""))
            try:
                img = Image.open(full_img_path).convert("RGB")
            except Exception as e:
                print(f"   ⚠️ Image open fail {art_id}: {e}")
                continue

            # 임베딩 생성
            try:
                vec = embed_image(img)
            except Exception as e:
                print(f"   ⚠️ Embedding fail {art_id}: {e}")
                continue

            # DB 저장
            collection.add(
                ids=[art_id],
                embeddings=[vec.tolist()],
                metadatas=[meta]
            )

            added += 1
            print(f"   ➕ Added {art_id} ({category})")

    print(f"\n🎉 Done! Indexed {added} items in curator_image_clip")

if __name__ == "__main__":
    main()
