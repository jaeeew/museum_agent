from flask import Flask, render_template_string, request, send_from_directory, url_for
import pandas as pd
from pathlib import Path
import os
import json

# ✅ Gemini SDK
from google import genai  # pip install google-genai

app = Flask(__name__)

# ======================
#  데이터 로드
# ======================
DATA_PATH = Path(r"E:\207.디지털 K-Art 데이터\01-1.정식개방데이터\k_art_metadata.csv")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df = df[df["img_path"].notna()].reset_index(drop=True)
df["idx"] = df.index  # 각 작품 고유 인덱스

# ======================
#  Gemini 클라이언트
# ======================
gemini_client = genai.Client()  # GEMINI_API_KEY 환경변수 사용


def generate_gemini_description(row):
    """짧은 큐레이터 설명"""
    title = row.get("title_kor") or row.get("title_eng") or "제목 없음"
    artist = row.get("artist_kor") or row.get("artist_eng") or "미상"
    period = row.get("main_category") or "-"
    art_class = row.get("class_kor") or row.get("class_eng") or "-"
    material = row.get("material_kor") or row.get("material_eng") or "-"
    location = row.get("location_kor") or row.get("location_eng") or "-"

    prompt = f"""
너는 한국 미술 전문 큐레이터야.
아래 작품 정보를 보고 일반 관람객에게 5~8문장 정도로 설명해 줘.

- 작품 제목: {title}
- 작가: {artist}
- 시대/연대: {period}
- 분류(장르): {art_class}
- 재질: {material}
- 소장처: {location}

설명할 때는 다음을 지켜 줘:
1. 첫 문장은 이 작품의 인상을 한 문장으로 요약해 줘.
2. 너무 학술적이지 말고, 누구나 이해할 수 있는 쉬운 표현을 사용해.
3. 이 작품의 미술사적/문화적 의미나 특징을 2~3가지 짚어 줘.
4. 마지막 문장은 "이 작품을 볼 때 ○○을(를) 함께 떠올려 보세요." 형태의 감상 팁으로 끝내 줘.
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def generate_gemini_narration(row):
    """
    Immersive 모드용 단계별 도슨트 내레이션 생성.
    4~6개의 단계로 나눠서 한두 문장씩 설명하도록 요청.
    """
    title = row.get("title_kor") or row.get("title_eng") or "제목 없음"
    artist = row.get("artist_kor") or row.get("artist_eng") or "미상"
    period = row.get("main_category") or "-"
    art_class = row.get("class_kor") or row.get("class_eng") or "-"
    material = row.get("material_kor") or row.get("material_eng") or "-"
    location = row.get("location_kor") or row.get("location_eng") or "-"

    prompt = f"""
너는 한국 미술관의 전문 도슨트야.
아래 작품을 관람객과 함께 감상한다고 생각하고, 4~6단계로 나누어 '투어 내레이션'을 만들어 줘.

각 단계는 1~2문장 정도로 해 줘.
번호는 쓰지 말고, 각 단계마다 줄바꿈만 해서 구분해 줘.

특히 다음과 같은 표현을 적절히 섞어 줘:
- "왼쪽 아래를 한 번 보세요..."
- "이제 시선을 화면 중앙으로 옮겨 볼까요?"
- "오른쪽 부분을 보면..."
- "이제 한 걸음 물러나 전체를 바라보면..."

작품 정보:
- 작품 제목: {title}
- 작가: {artist}
- 시대/연대: {period}
- 분류(장르): {art_class}
- 재질: {material}
- 소장처: {location}
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    text = response.text.strip()
    lines = [line.strip(" -•\n") for line in text.splitlines() if line.strip()]
    if len(lines) > 6:
        lines = lines[:6]
    return lines


# ======================
#  HTML 템플릿
# ======================

LIST_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>K-Art AI 큐레이터</title>
    <style>
        body { font-family: 'Noto Sans KR', sans-serif; background-color: #fafafa; margin: 40px; }
        .art { display: flex; margin-bottom: 40px; background: white; border-radius: 12px;
               box-shadow: 0 0 10px rgba(0,0,0,0.05); padding: 20px; align-items: flex-start; }
        .art img { width: 260px; height: auto; border-radius: 8px; margin-right: 20px; object-fit: contain; }
        .info h2 { margin-top: 0; }
        .search { margin-bottom: 30px; }
        a { text-decoration: none; color: #0044aa; }
        a:hover { text-decoration: underline; }
        .btn-detail { display:inline-block; margin-top:10px; padding:6px 10px; border-radius:6px;
                      background:#f0f4ff; font-size:0.9rem; }
    </style>
</head>
<body>
    <h1>🎨 K-Art AI 큐레이터</h1>
    <form class="search" method="get" action="/">
        <input type="text" name="q" placeholder="작품명·작가명·재질 검색" value="{{q}}" size="40">
        <input type="submit" value="검색">
        <a href="{{ url_for('home') }}">전체보기</a>
    </form>

    {% for _, row in items.iterrows() %}
        <div class="art">
            {% if row['img_path'] %}
                <img src="{{ url_for('serve_image', idx=row['idx']) }}" alt="이미지">
            {% endif %}
            <div class="info">
                <h2><a href="{{ url_for('detail', idx=row['idx']) }}">{{ row['title_kor'] or row['title_eng'] or '제목 없음' }}</a></h2>
                <p><b>작가:</b> {{ row['artist_kor'] or row['artist_eng'] or '정보 없음' }}</p>
                <p><b>분류:</b> {{ row['class_kor'] or '-' }}</p>
                <p><b>시대:</b> {{ row['main_category'] or '-' }}</p>
                <p><b>재질:</b> {{ row['material_kor'] or '-' }}</p>
                <p><b>소장처:</b> {{ row['location_kor'] or '-' }}</p>
                <a class="btn-detail" href="{{ url_for('detail', idx=row['idx']) }}">🧠 AI 설명 보기</a>
            </div>
        </div>
    {% endfor %}
</body>
</html>
"""

DETAIL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{ title }} - K-Art AI 큐레이터</title>
    <style>
        body { font-family: 'Noto Sans KR', sans-serif; background-color: #fafafa; margin: 40px; }
        .container {
            max-width: 1000px;
            margin: 40px auto;
            background:white;
            padding:30px;
            border-radius:12px;
            box-shadow:0 0 12px rgba(0,0,0,0.07);
        }

        /* ✅ 이미지 프레임: 이 안에서만 과감하게 줌 */
        .image-frame {
            width: 100%;
            max-height: 600px;
            overflow: hidden;
            border-radius: 12px;
            margin-bottom: 20px;
            position: relative;
            background: #000;
        }

        #art-image {
            width: 100%;
            height: auto;
            border-radius: 0;
            transition: transform 2.8s ease, transform-origin 2.8s ease;
            transform-origin: 50% 50%;
        }

        a { text-decoration:none; color:#0044aa; }
        a:hover { text-decoration:underline; }
        .meta p { margin: 3px 0; }
        .desc {
            background:#f6f7ff;
            padding:15px 20px;
            border-radius:10px;
            white-space:pre-line;
            margin-top: 15px;
        }
        .tts-buttons {
            margin-top: 10px;
        }
        .tts-buttons button {
            margin-right: 8px;
            padding:6px 10px;
            border-radius:6px;
            border:none;
            cursor:pointer;
            background:#ffe9f0;
            font-weight:bold;
        }
        .tts-buttons button:hover {
            background:#ffd6e3;
        }
    </style>
</head>
<body>
    <a href="{{ url_for('home') }}">← 목록으로 돌아가기</a>
    <div class="container">
        <h1>{{ title }}</h1>
        {% if img_url %}
            <div class="image-frame">
                <img id="art-image" src="{{ img_url }}" alt="이미지">
            </div>
        {% endif %}

        <div class="meta">
            <p><b>작가:</b> {{ artist }}</p>
            <p><b>분류:</b> {{ art_class }}</p>
            <p><b>시대:</b> {{ period }}</p>
            <p><b>재질:</b> {{ material }}</p>
            <p><b>소장처:</b> {{ location }}</p>
        </div>

        <h3>🧠 AI 큐레이터 설명 (Gemini)</h3>

        <!-- 🔊 기본 설명 TTS + Immersive 투어 버튼 -->
        <div class="tts-buttons">
            <button onclick="speakDesc()">🔊 설명 듣기</button>
            <button onclick="stopDesc()">⏹ 멈추기</button>
            <button onclick="startTour()">🎧 작품 속으로 들어가기</button>
            <button onclick="stopTour()">⏹ 투어 멈추기</button>
        </div>

        <!-- 설명 텍스트 -->
        <div id="desc-text" class="desc">
            {{ description }}
        </div>
    </div>

    <!-- narration 데이터 (JS에서 사용) -->
    <script>
        const tourNarration = {{ narration | safe }};
    </script>

    <!-- 🧠 음성 읽기 + Immersive 투어 (Web Speech API) -->
    <script>
        let docentVoice = null;

        function pickKoreanVoice() {
            const voices = window.speechSynthesis.getVoices();
            if (!voices || voices.length === 0) return null;

            const koVoices = voices.filter(v => v.lang && v.lang.startsWith('ko'));
            if (koVoices.length === 0) return null;

            const preferredKeywords = ["natural", "neural", "online", "cloud", "han", "heami", "sunhi", "Google", "Microsoft"];
            for (const v of koVoices) {
                const nameLower = (v.name || "").toLowerCase();
                if (preferredKeywords.some(k => nameLower.includes(k.toLowerCase()))) {
                    return v;
                }
            }
            return koVoices[0];
        }

        if ('speechSynthesis' in window) {
            window.speechSynthesis.onvoiceschanged = () => {
                docentVoice = pickKoreanVoice();
                console.log("선택된 한국어 음성:", docentVoice ? docentVoice.name : "기본 음성");
            };
        }

        function makeUtter(text) {
            const utter = new SpeechSynthesisUtterance(text);
            utter.lang = 'ko-KR';
            utter.rate = 0.9;
            utter.pitch = 0.95;
            if (docentVoice) utter.voice = docentVoice;
            return utter;
        }

        function speakDesc() {
            if (!('speechSynthesis' in window)) {
                alert('이 브라우저는 음성 읽기를 지원하지 않습니다.');
                return;
            }
            const text = document.getElementById('desc-text').innerText.trim();
            if (!text) return;

            window.speechSynthesis.cancel();
            const utter = makeUtter(text);
            window.speechSynthesis.speak(utter);
        }

        function stopDesc() {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
            }
        }

        // ===== Immersive 투어 =====
        let tourIndex = 0;

        function updateImageForStep(step) {
            const img = document.getElementById('art-image');
            if (!img) return;

            let scale = 1.0;
            let originX = "50%";
            let originY = "50%";

            if (step === 0) {
                // 전체 첫인상
                scale = 1.15;
                originX = "50%"; originY = "50%";
            } else if (step === 1) {
                // 왼쪽 아래 크게
                scale = 2.0;
                originX = "20%"; originY = "80%";
            } else if (step === 2) {
                // 중앙 강하게
                scale = 2.2;
                originX = "50%"; originY = "40%";
            } else if (step === 3) {
                // 오른쪽 강조
                scale = 2.0;
                originX = "80%"; originY = "50%";
            } else if (step === 4) {
                // 다시 전체 쪽으로
                scale = 1.1;
                originX = "50%"; originY = "50%";
            } else {
                // 투어 종료: 원래대로
                scale = 1.0;
                originX = "50%"; originY = "50%";
            }

            img.style.transformOrigin = originX + " " + originY;
            img.style.transform = "scale(" + scale + ")";

            const frame = img.parentElement;
            if (frame) {
                const top = frame.getBoundingClientRect().top + window.scrollY;
                window.scrollTo({ top: top - 40, behavior: 'smooth' });
            }
        }

        function playTourStep() {
            if (!('speechSynthesis' in window)) return;
            if (!tourNarration || tourNarration.length === 0) return;
            if (tourIndex >= tourNarration.length) {
                updateImageForStep(999);
                return;
            }

            const text = tourNarration[tourIndex];
            if (!text) return;

            window.speechSynthesis.cancel();
            const utter = makeUtter(text);

            utter.onstart = () => {
                updateImageForStep(tourIndex);
            };
            utter.onend = () => {
                tourIndex++;
                playTourStep();
            };

            window.speechSynthesis.speak(utter);
        }

        function startTour() {
            if (!('speechSynthesis' in window)) {
                alert('이 브라우저는 음성 읽기를 지원하지 않습니다.');
                return;
            }
            tourIndex = 0;
            playTourStep();
        }

        function stopTour() {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
            }
            tourIndex = 0;
            updateImageForStep(999);
        }
    </script>
</body>
</html>
"""


# ======================
#  라우트
# ======================

@app.route("/")
def home():
    q = request.args.get("q", "")
    results = df.copy()

    if q:
        q_str = q.strip()
        mask = (
            results["title_kor"].fillna("").str.contains(q_str, case=False)
            | results["artist_kor"].fillna("").str.contains(q_str, case=False)
            | results["material_kor"].fillna("").str.contains(q_str, case=False)
            | results["location_kor"].fillna("").str.contains(q_str, case=False)
        )
        results = results[mask]

    # 최대 20개만 랜덤으로
    if len(results) > 20:
        results = results.sample(20)

    return render_template_string(LIST_TEMPLATE, items=results, q=q)


@app.route("/image/<int:idx>")
def serve_image(idx):
    row = df.iloc[idx]
    img_path = Path(row["img_path"])
    return send_from_directory(img_path.parent, img_path.name)


@app.route("/detail/<int:idx>")
def detail(idx):
    row = df.iloc[idx]

    title = row.get("title_kor") or row.get("title_eng") or "제목 없음"
    artist = row.get("artist_kor") or row.get("artist_eng") or "미상"
    period = row.get("main_category") or "-"
    art_class = row.get("class_kor") or row.get("class_eng") or "-"
    material = row.get("material_kor") or row.get("material_eng") or "-"
    location = row.get("location_kor") or row.get("location_eng") or "-"

    description = generate_gemini_description(row)
    narration_steps = generate_gemini_narration(row)

    img_url = None
    if row.get("img_path"):
        img_url = url_for("serve_image", idx=idx)

    return render_template_string(
        DETAIL_TEMPLATE,
        title=title,
        artist=artist,
        period=period,
        art_class=art_class,
        material=material,
        location=location,
        img_url=img_url,
        description=description,
        narration=json.dumps(narration_steps, ensure_ascii=False),
    )


if __name__ == "__main__":
    app.run(debug=True)
