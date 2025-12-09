# build_image_index.py
import json
from pathlib import Path

# 👉 app.py랑 맞춤
DATA_ROOT = Path(r"D:\Exhibit")
IMG_ROOT = DATA_ROOT / "image_extracted"

# 사용할 확장자
EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def main():
    index = {}  # prefix(id) -> "TS_01. 2D_02.회화_1/kart_2d000496-C-8-81-1.jpg"

    # image_extracted 전체 순회
    for path in IMG_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in EXTS:
            continue

        prefix = path.stem  # kart_2d000496-C-8-81-1
        rel = path.relative_to(IMG_ROOT).as_posix()  # TS_01. 2D_02.회화_1/...

        # 같은 prefix가 여러 번 나와도 먼저 것 유지 (원하면 리스트로 바꿔도 됨)
        if prefix not in index:
            index[prefix] = rel

    out_path = DATA_ROOT / "image_index.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"총 {len(index)}개 prefix 인덱싱 완료")
    print("저장 경로:", out_path)

if __name__ == "__main__":
    main()
