import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from image_ops import adjust_contrast_brightness, augment_image, degrade_resolution

MAX_SIDE = 1600  # 너무 큰 이미지는 이 크기로 줄여서 처리 (느려짐 방지)

st.set_page_config(page_title="Sim Lab", page_icon="🧪", layout="wide")
st.title("🧪 Sim Lab — 해상도 & 데이터 증강 시뮬레이터")


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
        img = np.array(Image.open(file).convert("RGB"))  # 흑백·RGBA도 RGB 3채널로 통일
    except (UnidentifiedImageError, OSError):
        return None
    if img.size == 0:
        return None
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        f = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return img


# ------------------------------------------------------------------ 사이드바
with st.sidebar:
    uploaded = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg", "bmp", "webp"])

    st.header("1. 해상도")
    scale = st.slider("해상도 비율", 0.05, 1.0, 1.0, 0.05,
                      help="0.2 = 가로·세로 20% 크기로 줄였다가 다시 키움")
    up_name = st.radio("다시 키울 때 보간법", ["NEAREST (계단)", "LINEAR (뭉개짐)"], horizontal=True)
    up = cv2.INTER_NEAREST if up_name.startswith("NEAREST") else cv2.INTER_LINEAR

    st.header("2. 밝기·대비")
    alpha = st.slider("대비 (alpha)", 0.5, 3.0, 1.0, 0.1)
    beta = st.slider("밝기 (beta)", -100, 100, 0, 5)

    st.header("3. 데이터 증강")
    flip = st.checkbox("좌우 반전")
    use_noise = st.checkbox("가우시안 노이즈")
    sigma = st.slider("노이즈 강도 (σ)", 1, 80, 20, disabled=not use_noise)

# ------------------------------------------------------------------ 이미지 준비
if uploaded is None:
    original = sample_image()
    st.caption("샘플 이미지 사용 중 — 사이드바에서 내 이미지를 올릴 수 있어요.")
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
except Exception as e:  # 예상 못 한 값이 들어와도 앱 전체가 죽지 않게
    st.error(f"변환 중 오류: {e}")
    st.stop()

# ------------------------------------------------------------------ 화면
h, w = original.shape[:2]
small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))

col1, col2 = st.columns(2)
with col1:
    st.subheader("원본")
    st.image(original, width="stretch")
    st.caption(f"{w} × {h} px")
with col2:
    st.subheader("변환 결과")
    st.image(result, width="stretch")
    st.caption(f"{small_w} × {small_h} px로 축소 후 복원 · alpha={alpha} · beta={beta:+d}"
               + (" · 반전" if flip else "") + (f" · 노이즈 σ={sigma}" if use_noise else ""))
