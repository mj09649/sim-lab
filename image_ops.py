"""이미지 처리 함수 모음. 모든 함수는 uint8 이미지(H, W, 3)를 받아 같은 모양으로 돌려준다."""
import cv2
import numpy as np


# ------------------------------------------------------------------ 1. 해상도 저하
def degrade_resolution(img, scale, down=cv2.INTER_AREA, up=cv2.INTER_NEAREST):
    """원본 크기 × scale 로 줄였다가(down) 다시 원래 크기로 키운다(up).

    scale=0.2 → 가로·세로 20% 크기로 축소 후 복원.
    up=cv2.INTER_NEAREST : 픽셀이 네모 블록으로 커짐 → 계단 현상
    up=cv2.INTER_LINEAR  : 주변 픽셀을 섞음 → 부드럽지만 뭉개짐
    """
    if img is None or img.size == 0:
        raise ValueError("빈 이미지입니다.")
    scale = float(np.clip(scale, 0.01, 1.0))  # 0 이하·1 초과 값 방어
    h, w = img.shape[:2]
    small_w = max(1, int(w * scale))  # 아무리 작아도 최소 1픽셀 → 크기 0 에러 방지
    small_h = max(1, int(h * scale))
    small = cv2.resize(img, (small_w, small_h), interpolation=down)  # 주의: (가로, 세로) 순서
    return cv2.resize(small, (w, h), interpolation=up)


# ------------------------------------------------------------------ 2. 밝기·대비
def adjust_contrast_brightness(img, alpha=1.0, beta=0):
    """새 픽셀 = alpha × 원래 픽셀 + beta  (결과는 0~255로 잘라냄)

    alpha > 1 : 대비 증가 (밝은 곳은 더 밝게, 어두운 곳은 더 어둡게)
    beta  > 0 : 전체가 밝아짐, beta < 0 : 어두워짐
    """
    # uint8 상태로 계산하면 255를 넘을 때 0부터 다시 돌아간다(오버플로) → float로 계산 후 자르기
    out = img.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ 3. 데이터 증강
def augment_image(img, flip=False, noise_sigma=0, seed=None):
    """좌우 반전 + 가우시안 노이즈, 두 가지만.

    flip=True       : 좌우 반전
    noise_sigma > 0 : 평균 0, 표준편차 noise_sigma 인 정규분포 노이즈를 더함
    """
    out = img
    noise_sigma = max(0.0, float(noise_sigma))  # 음수 표준편차 방어
    if flip:
        out = cv2.flip(out, 1)  # 1 = 좌우, 0 = 상하
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, noise_sigma, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out
