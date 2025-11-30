// node-server/server.js
const express = require("express");
const path = require("path");
const fs = require("fs");
const glob = require("glob");
const cors = require("cors");

const app = express();
const PORT = 8080;

// 🔥 실제 경로
const TRAIN_LABEL_ROOT =
  "E:/207.디지털 K-Art 데이터/01-1.정식개방데이터/Training/02.라벨링데이터";
const VAL_LABEL_ROOT =
  "E:/207.디지털 K-Art 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터";

const TRAIN_IMG_ROOT =
  "E:/207.디지털 K-Art 데이터/01-1.정식개방데이터/Training/01.원천데이터";
const VAL_IMG_ROOT =
  "E:/207.디지털 K-Art 데이터/01-1.정식개방데이터/Validation/01.원천데이터";

app.use(cors());

/** id → json 경로 / 이미지 경로 */
const jsonIndex = {};
const imageIndex = {};

/** id → { id, title, artist, category }  (한 번 계산한 메타 캐시) */
const metaCache = {};

// JSON 인덱싱
function indexJson(root) {
  const pattern = path.join(root, "**/*.json").replace(/\\/g, "/");
  console.log("📂 JSON 스캔:", pattern);
  const files = glob.sync(pattern, { nodir: true });
  files.forEach((file) => {
    const stem = path.basename(file, ".json");
    if (!jsonIndex[stem]) {
      jsonIndex[stem] = file;
    }
  });
}

// 이미지 인덱싱
function indexImages(root) {
  const pattern = path.join(root, "**/*.+(jpg|jpeg|JPG|JPEG)").replace(
    /\\/g,
    "/"
  );
  console.log("🖼 이미지 스캔:", pattern);
  const files = glob.sync(pattern, { nodir: true });
  files.forEach((file) => {
    const stem = path.basename(file).replace(/\.(jpg|jpeg|JPG|JPEG)$/, "");
    if (!imageIndex[stem]) {
      imageIndex[stem] = file;
    }
  });
}

// 카테고리 추출 (공예 / 회화 / 조각 → craft / painting / sculpture)
function deriveSimpleCategory(json) {
  const obj = json.Object_Info || {};
  const main = (obj.MainCategory || "").toLowerCase();
  const mid = (obj.MiddleCategory || "").toLowerCase();
  const sub = (obj.SubCategory || "").toLowerCase();
  const all = `${main} ${mid} ${sub}`;

  if (all.includes("공예") || all.includes("craft")) return "craft";
  if (all.includes("조각") || all.includes("sculpture")) return "sculpture";
  // 나머지는 전부 회화 계열로 취급
  return "painting";
}

// id → 메타정보(title, artist, category) 가져오기 (캐시 사용)
function getMetaForId(id) {
  if (metaCache[id]) return metaCache[id];

  const filePath = jsonIndex[id];
  if (!filePath) return null;

  try {
    const raw = fs.readFileSync(filePath, "utf-8");
    const json = JSON.parse(raw);

    const desc = json.Description || {};
    const title =
      desc.ArtTitle_kor ||
      desc.ArtTitle_eng ||
      desc.ArtTitle ||
      json.title ||
      "제목 없음";
    const artist =
      desc.ArtistName_kor ||
      desc.ArtistName_eng ||
      desc.ArtistName ||
      json.artist ||
      "작가 미상";

    const category = deriveSimpleCategory(json);

    const meta = { id, title, artist, category };
    metaCache[id] = meta;
    return meta;
  } catch (e) {
    console.error("getMetaForId 오류:", id, e.message);
    return null;
  }
}

// Fisher–Yates 셔플
function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ───────────────────────────────
// 서버 시작 시 인덱싱
// ───────────────────────────────
console.log("=== 인덱싱 시작 ===");
indexJson(TRAIN_LABEL_ROOT);
indexJson(VAL_LABEL_ROOT);
indexImages(TRAIN_IMG_ROOT);
indexImages(VAL_IMG_ROOT);
console.log("✅ JSON 개수   :", Object.keys(jsonIndex).length);
console.log("✅ 이미지 개수 :", Object.keys(imageIndex).length);
console.log("=== 인덱싱 완료 ===");

// ───────────────────────────────
// 0) 작품 목록 API: /json_list
//    예) /json_list?category=craft&limit=20
//    - category: craft | painting | sculpture (없으면 전체)
//    - limit:    최대 개수 (기본 60)
//    결과는 항상 랜덤 순서
// ───────────────────────────────
app.get("/json_list", (req, res) => {
  const limit = parseInt(req.query.limit, 10) || 60;
  const requestedCategory = req.query.category; // craft / painting / sculpture / undefined

  const allIds = Object.keys(jsonIndex);
  if (allIds.length === 0) {
    return res.json({ total: 0, items: [] });
  }

  const shuffledIds = shuffle(allIds);
  const items = [];

  for (const id of shuffledIds) {
    const meta = getMetaForId(id);
    if (!meta) continue;

    if (requestedCategory && meta.category !== requestedCategory) {
      continue;
    }

    items.push(meta);
    if (items.length >= limit) break;
  }

  // total은 여기서는 "이번에 반환한 개수"로만 사용
  res.json({ total: items.length, items });
});

// ───────────────────────────────
// 1) JSON 제공: /json/:id
// ───────────────────────────────
app.get("/json/:id", (req, res) => {
  const id = req.params.id;
  const jsonPath = jsonIndex[id];
  if (!jsonPath) {
    return res.status(404).json({ error: "JSON not found", id });
  }
  res.sendFile(path.resolve(jsonPath));
});

// ───────────────────────────────
// 2) 이미지 파일 제공: /image/:id
// ───────────────────────────────
app.get("/image/:id", (req, res) => {
  const id = req.params.id;
  const imgPath = imageIndex[id];
  if (!imgPath) {
    return res.status(404).json({ error: "Image not found", id });
  }
  res.sendFile(path.resolve(imgPath));
});

// ───────────────────────────────
// 3) 이미지 URL만 알려주는 API: /find_image/:id
// ───────────────────────────────
app.get("/find_image/:id", (req, res) => {
  const id = req.params.id;
  if (!imageIndex[id]) {
    return res.status(404).json({ error: "Image not found", id });
  }
  return res.json({ url: `/image/${id}` });
});

// 헬스체크
app.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    jsonCount: Object.keys(jsonIndex).length,
    imageCount: Object.keys(imageIndex).length,
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Node data server running at http://localhost:${PORT}`);
});
