# build_aicaption.py (배치 캡션 버전)

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
import google.generativeai as genai

# ──────────────────────────────
# 0. 환경설정 (app.py와 동일하게)
# ──────────────────────────────
load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY가 없습니다.")

genai.configure(api_key=API_KEY)

# 이미지도 되는 모델로 선택 (1.5-flash, 2.0-flash 등)
VISION_MODEL_NAME = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")
vision_model = genai.GenerativeModel(VISION_MODEL_NAME)

DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"D:\Exhibit"))
IMG_ROOT = DATA_ROOT / "image_extracted"
JSON_ROOT = DATA_ROOT / "json_extracted"
IMAGE_INDEX_PATH = DATA_ROOT / "image_index.json"

CATEGORY_MAP: Dict[str, str] = {
    "painting_json": "TL_01. 2D_02.회화(Json)",
    "craft_json": "TL_01. 2D_04.공예(Json)",
    "sculpture_json": "TL_01. 2D_06.조각(Json)",
}

with IMAGE_INDEX_PATH.open("r", encoding="utf-8") as f:
    IMAGE_INDEX = json.load(f)

print(f"[build_aicaption] total images in index: {len(IMAGE_INDEX)}")

# 한 번에 몇 장씩 보낼지 (너무 크면 rate limit / 응답 실패 위험↑)
BATCH_SIZE = 8
# 호출 사이에 살짝 쉬고 싶으면 (옵션)
SLEEP_BETWEEN_CALLS = 0.0  # 초 단위. 필요하면 0.5 ~ 1.0 정도로 조정.


# ──────────────────────────────
# 1. art_id → JSON 경로 찾기
# ──────────────────────────────
def find_json_for_art_id(art_id: str) -> Path | None:
    """
    id에 해당하는 json_extracted 내부 JSON 파일 경로를 찾는다.
    (painting / craft / sculpture 전부 탐색)
    """
    for _, real_name in CATEGORY_MAP.items():
        p = JSON_ROOT / real_name / f"{art_id}.json"
        if p.exists():
            return p
    return None


# ──────────────────────────────
# 2. 단일 이미지 캡션 (배치 실패 시 fallback 용)
# ──────────────────────────────
def generate_caption_for_image_single(img_bytes: bytes) -> str:
    """
    이미지 1장에 대해 캡션 1개 생성.
    (배치 호출이 실패했을 때 per-image fallback 으로만 사용)
    """
    prompt = (
        "이 이미지는 미술 작품의 사진입니다. "
        "화면에 보이는 주요 대상과 장면을 한국어로 아주 짧게 설명해 주세요. "
        "예시: '나뭇가지에 앉아 있는 작은 참새', '연못가에 서 있는 두 사람의 뒷모습', "
        "'바닷가 해변과 파도', '도시 야경과 불빛'. "
        "작품 해설처럼 길게 쓰지 말고, 핵심 대상/장면만 1문장으로 30자 이내로 적어 주세요. "
        "문장 끝에 마침표는 붙이지 않아도 됩니다."
    )
    try:
        resp = vision_model.generate_content(
            [
                prompt,
                {"mime_type": "image/jpeg", "data": img_bytes},
            ]
        )
        text = (resp.text or "").strip()
        if len(text) > 60:
            text = text[:60]
        return text
    except Exception as e:
        print(f"[build_aicaption] single caption generation failed: {e}")
        return ""


# ──────────────────────────────
# 3. 배치 캡션 생성
# ──────────────────────────────
def generate_captions_for_batch(
    batch_items: List[Tuple[str, Path]]
) -> Dict[str, str]:
    """
    여러 장의 이미지를 한 번에 Gemini에 보내서
    { art_id: caption } 딕셔너리로 받는다.

    batch_items: [(art_id, img_path), ...]
    """
    if not batch_items:
        return {}

    prompt = (
        "You are a captioner for museum artworks. "
        "For each image I give you, you will see a line 'ID: <art_id>' followed by the artwork image. "
        "For each image, describe the main visible subject/scene in Korean, "
        "as a single short phrase (not a long explanation), within 30 Korean characters. "
        "Do NOT include periods at the end.\n\n"
        "Output MUST be a pure JSON array with objects like:\n"
        '[{"id": "kart_2d000001-C-8-81-1", "caption": "나뭇가지에 앉아 있는 작은 참새"},\n'
        ' {"id": "kart_2d000002-C-8-81-2", "caption": "연못가에 서 있는 두 사람의 뒷모습"}]\n\n'
        "Do not add any extra keys. Do not add comments or explanations outside JSON.\n"
    )

    contents: List[object] = [prompt]

    # 이미지 파트 붙이기
    for art_id, img_path in batch_items:
        with img_path.open("rb") as f:
            img_bytes = f.read()
        contents.append(f"ID: {art_id}")
        contents.append({"mime_type": "image/jpeg", "data": img_bytes})

    try:
        resp = vision_model.generate_content(contents)
        text = (resp.text or "").strip()
        # 모델이 JSON만 내놨다고 가정하고 파싱 시도
        data = json.loads(text)
        result: Dict[str, str] = {}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                art_id = str(item.get("id") or "").strip()
                caption = str(item.get("caption") or "").strip()
                if not art_id or not caption:
                    continue
                # 너무 길면 잘라주기 (안전용)
                if len(caption) > 60:
                    caption = caption[:60]
                result[art_id] = caption
        return result
    except Exception as e:
        print(f"[build_aicaption] batch caption generation failed, fallback to single: {e}")
        # 배치 전체가 실패하면, 여기서 빈 dict를 리턴하고
        # main() 쪽에서 per-image로 다시 시도하게 한다.
        return {}


# ──────────────────────────────
# 4. 메인 루프: 이미지 → 캡션 → JSON 업데이트 (배치)
# ──────────────────────────────
def main():
    updated_count = 0
    skipped_caption_exist = 0
    missing_json = 0
    missing_img = 0
    batch_calls = 0
    batch_single_fallback = 0

    # 현재 배치에 쌓여 있는 작업: (art_id, img_path, json_path)
    current_batch: List[Tuple[str, Path, Path]] = []

    def process_batch(batch: List[Tuple[str, Path, Path]]):
        nonlocal updated_count, batch_calls, batch_single_fallback

        if not batch:
            return

        batch_calls += 1
        # (art_id, img_path)만 떼서 caption 요청
        id_img_pairs = [(art_id, img_path) for art_id, img_path, _ in batch]

        captions = generate_captions_for_batch(id_img_pairs)

        # 배치 호출이 완전히 실패한 경우 -> 전부 single로 fallback
        if not captions:
            batch_single_fallback += len(batch)
            for art_id, img_path, json_path in batch:
                with img_path.open("rb") as f:
                    img_bytes = f.read()
                caption = generate_caption_for_image_single(img_bytes)
                if not caption:
                    continue
                try:
                    with json_path.open("r", encoding="utf-8") as f:
                        card = json.load(f)
                except Exception as e:
                    print(f"[build_aicaption] JSON reload error (fallback) for {json_path}: {e}")
                    continue

                card["vision_caption_ko"] = caption
                try:
                    with json_path.open("w", encoding="utf-8") as f:
                        json.dump(card, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[build_aicaption] JSON write error (fallback) for {json_path}: {e}")
                    continue
                updated_count += 1
            return

        # 배치 호출이 어느 정도 성공한 경우 -> id별 caption 사용
        for art_id, img_path, json_path in batch:
            caption = captions.get(art_id)
            if not caption:
                # 특정 id만 빠졌으면, 그 id만 single 호출로 보충
                batch_single_fallback += 1
                with img_path.open("rb") as f:
                    img_bytes = f.read()
                caption = generate_caption_for_image_single(img_bytes)
                if not caption:
                    continue

            # JSON 로드 & 저장
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    card = json.load(f)
            except Exception as e:
                print(f"[build_aicaption] JSON reload error for {json_path}: {e}")
                continue

            card["vision_caption_ko"] = caption

            try:
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(card, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[build_aicaption] JSON write error for {json_path}: {e}")
                continue

            updated_count += 1

        if SLEEP_BETWEEN_CALLS > 0:
            time.sleep(SLEEP_BETWEEN_CALLS)

    # ─────────────────────
    # 메인 loop
    # ─────────────────────
    for art_id, rel_img in IMAGE_INDEX.items():
        img_path = IMG_ROOT / rel_img

        if not img_path.exists():
            print(f"[build_aicaption] MISSING IMG: {img_path}")
            missing_img += 1
            continue

        json_path = find_json_for_art_id(art_id)
        if not json_path:
            print(f"[build_aicaption] MISSING JSON for id={art_id}")
            missing_json += 1
            continue

        # JSON 로드해서 caption 유무 확인
        try:
            with json_path.open("r", encoding="utf-8") as f:
                card = json.load(f)
        except Exception as e:
            print(f"[build_aicaption] JSON load error for {json_path}: {e}")
            missing_json += 1
            continue

        if card.get("vision_caption_ko"):
            skipped_caption_exist += 1
            continue

        print(f"[build_aicaption] queue for caption id={art_id}")
        current_batch.append((art_id, img_path, json_path))

        # 배치가 가득 찼으면 처리
        if len(current_batch) >= BATCH_SIZE:
            print(f"[build_aicaption] processing batch of {len(current_batch)} items...")
            process_batch(current_batch)
            current_batch = []

    # 마지막 남은 것 처리
    if current_batch:
        print(f"[build_aicaption] processing last batch of {len(current_batch)} items...")
        process_batch(current_batch)

    print("─────────────────────────────────────")
    print(f"캡션 새로 추가한 작품 수: {updated_count}")
    print(f"이미 캡션 있어서 건너뛴 수: {skipped_caption_exist}")
    print(f"이미지는 있는데 JSON 없는 수: {missing_json}")
    print(f"index에는 있는데 이미지 없는 수: {missing_img}")
    print(f"배치 호출 수: {batch_calls}")
    print(f"배치 실패/누락으로 single fallback 사용한 개수: {batch_single_fallback}")
    print("─────────────────────────────────────")


if __name__ == "__main__":
    main()
