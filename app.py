from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from image_ops import adjust_contrast_brightness, augment_image, degrade_resolution

MAX_SIDE = 1600  # 너무 큰 이미지는 이 크기로 줄여서 처리 (느려짐 방지)

st.set_page_config(page_title="Sim Lab", page_icon="🧪", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 12% 18%, rgba(125, 211, 252, 0.24), transparent 36%),
                radial-gradient(circle at 88% 28%, rgba(196, 181, 253, 0.26), transparent 38%),
                radial-gradient(circle at 75% 78%, rgba(167, 243, 208, 0.22), transparent 40%),
                linear-gradient(180deg, #f8faff 0%, #f4f6ff 55%, #f9faff 100%);
            color: #1e293b;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f5f7ff 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }
        .hero-banner {
            background: linear-gradient(115deg, #99f6e4 0%, #93c5fd 48%, #c4b5fd 100%);
            border-radius: 24px;
            padding: 28px 32px;
            margin-bottom: 20px;
            box-shadow: 0 12px 32px rgba(99, 102, 241, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.75);
        }
        .hero-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.72);
            color: #3730a3;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 999px;
            margin-bottom: 10px;
        }
        .hero-title {
            margin: 0;
            font-size: 1.9rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.02em;
        }
        .hero-sub {
            margin: 8px 0 0 0;
            font-size: 0.98rem;
            color: #334155;
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(226, 232, 240, 0.9);
            border-radius: 18px;
            padding: 16px 20px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        [data-testid="stMetricValue"] {
            color: #2563eb;
            font-weight: 700;
        }
        [data-testid="stImage"] img {
            border-radius: 18px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(226, 232, 240, 0.9);
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(99, 102, 241, 0.22);
            background: #ffffff;
            color: #1e293b;
            font-weight: 600;
            padding: 0.55rem 1rem;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.08);
            transition: all 0.2s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: #6366f1;
            background: #eef2ff;
            color: #3730a3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">IMAGE RESOLUTION &amp; AUGMENTATION LAB</div>
        <h1 class="hero-title">🧪 Sim Lab — 해상도 &amp; 데이터 증강 시뮬레이터</h1>
        <p class="hero-sub">해상도 저하에 따른 시각 정보 변화를 관찰하고, 밝기·대비 보정 및 데이터 증강 결과를 나란히 비교합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def sample_image():
    """업로드가 없을 때 쓰는 테스트 이미지 (RGB)."""
    img = np.full((240, 320, 3), 200, np.uint8)
    cv2.line(img, (10, 230), (150, 10), (200, 0, 0), 3)
    cv2.circle(img, (230, 80), 55, (0, 80, 200), -1)
    cv2.putText(img, "Sim Lab", (160, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3)
    for x in range(10, 150, 8):
        cv2.line(img, (x, 160), (x, 230), (60, 60, 60), 2)
    return img


def load_uploaded(file):
    """업로드 파일 → RGB uint8 배열. 실패하면 None."""
    try:
        img = np.array(Image.open(file).convert("RGB"))
    except (UnidentifiedImageError, OSError):
        return None
    if img.size == 0:
        return None
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        f = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return img


def make_comparison_png(orig, res):
    """원본과 변환 결과를 나란히 붙인 비교 이미지를 PNG 바이트로 반환."""
    h = max(orig.shape[0], res.shape[0])
    orig_r = cv2.resize(orig, (orig.shape[1], h), interpolation=cv2.INTER_AREA)
    res_r = cv2.resize(res, (res.shape[1], h), interpolation=cv2.INTER_AREA)
    divider = np.full((h, 12, 3), 240, np.uint8)
    merged = np.hstack([orig_r, divider, res_r])
    buf = BytesIO()
    Image.fromarray(merged).save(buf, format="PNG")
    return buf.getvalue()


def reset_controls():
    st.session_state["scale"] = 1.0
    st.session_state["up_name"] = "NEAREST (계단)"
    st.session_state["alpha"] = 1.0
    st.session_state["beta"] = 0
    st.session_state["flip"] = False
    st.session_state["use_noise"] = False
    st.session_state["sigma"] = 20


# 기본값 초기화
st.session_state.setdefault("scale", 1.0)
st.session_state.setdefault("up_name", "NEAREST (계단)")
st.session_state.setdefault("alpha", 1.0)
st.session_state.setdefault("beta", 0)
st.session_state.setdefault("flip", False)
st.session_state.setdefault("use_noise", False)
st.session_state.setdefault("sigma", 20)

# ------------------------------------------------------------------ 사이드바
with st.sidebar:
    st.subheader("📂 입력 이미지")
    uploaded = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg", "bmp", "webp"])
    st.button("설정 초기화", on_click=reset_controls, width="stretch")

    st.divider()
    st.subheader("1. 해상도 설정")
    scale = st.slider(
        "해상도 비율", 0.05, 1.0, key="scale", step=0.05,
        help="0.2 = 가로·세로 20% 크기로 줄였다가 다시 키움"
    )
    up_name = st.radio(
        "복원 보간법", ["NEAREST (계단)", "LINEAR (뭉개짐)"], key="up_name", horizontal=True
    )
    up = cv2.INTER_NEAREST if up_name.startswith("NEAREST") else cv2.INTER_LINEAR

    st.divider()
    st.subheader("2. 밝기·대비 보정")
    alpha = st.slider("대비 (alpha)", 0.5, 3.0, key="alpha", step=0.1)
    beta = st.slider("밝기 (beta)", -100, 100, key="beta", step=5)

    st.divider()
    st.subheader("3. 데이터 증강")
    flip = st.checkbox("좌우 반전", key="flip")
    use_noise = st.checkbox("가우시안 노이즈", key="use_noise")
    sigma = st.slider("노이즈 강도 (σ)", 1, 80, key="sigma", disabled=not use_noise)

# ------------------------------------------------------------------ 이미지 준비
if uploaded is None:
    original = sample_image()
    st.info("현재 기본 테스트 이미지를 보고 있습니다. 사이드바에서 원하는 이미지를 업로드해 보세요.")
else:
    original = load_uploaded(uploaded)
    if original is None:
        st.error("이미지를 읽을 수 없습니다. 다른 파일을 올려 주세요.")
        st.stop()

# ------------------------------------------------------------------ 변환 (해상도 → 밝기·대비 → 증강)
try:
    result = degrade_resolution(original, scale, up=up)
    result = adjust_contrast_brightness(result, alpha, beta)
    result = augment_image(result, flip=flip, noise_sigma=sigma if use_noise else 0, seed=0)
except Exception as e:
    st.error(f"변환 중 오류: {e}")
    st.stop()

# ------------------------------------------------------------------ 화면
h, w = original.shape[:2]
small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))

m1, m2, m3 = st.columns(3)
m1.metric("원본 크기", f"{w} × {h} px")
m2.metric("축소 크기", f"{small_w} × {small_h} px", f"{int(scale * 100)}%")
m3.metric("명암 표준편차 (대비)", f"{result.std():.1f}", f"{result.std() - original.std():+.1f}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("원본")
    st.image(original, width="stretch")
with col2:
    st.subheader("변환 결과")
    st.image(result, width="stretch")
    st.caption(
        f"{small_w} × {small_h} px 축소 후 복원({up_name.split()[0]}) · "
        f"alpha={alpha:.1f} · beta={beta:+d}"
        + (" · 좌우반전" if flip else "")
        + (f" · 노이즈 σ={sigma}" if use_noise else "")
    )

png_bytes = make_comparison_png(original, result)
st.download_button(
    label="📥 원본 + 변환 결과 비교 이미지 저장 (PNG)",
    data=png_bytes,
    file_name="sim_lab_comparison.png",
    mime="image/png",
)
