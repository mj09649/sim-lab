"""함수를 하나씩 테스트한다.

    python test_image_ops.py            # 전부
    python test_image_ops.py degrade    # 해상도 저하만
    python test_image_ops.py contrast   # 밝기·대비만
    python test_image_ops.py augment    # 증강만
    python test_image_ops.py all 사진.jpg   # 내 사진으로

결과 그림은 out/ 폴더에 저장된다.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from image_ops import adjust_contrast_brightness, augment_image, degrade_resolution

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


# ------------------------------------------------------------------ 준비
def imread(path):  # 한글 경로 대응 (결과는 BGR)
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img_bgr):
    cv2.imencode(Path(path).suffix, img_bgr)[1].tofile(str(path))


def make_test_image():
    """계단 현상이 잘 보이도록 대각선·원·글자·줄무늬가 있는 테스트 이미지 (BGR)."""
    img = np.full((240, 320, 3), 200, np.uint8)
    cv2.line(img, (10, 230), (150, 10), (0, 0, 200), 3)
    cv2.circle(img, (230, 80), 55, (200, 80, 0), -1)
    cv2.putText(img, "Sim Lab", (160, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3)
    for x in range(10, 150, 8):  # 촘촘한 세로 줄무늬
        cv2.line(img, (x, 160), (x, 230), (60, 60, 60), 2)
    return img


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def check(cond, msg):
    print(f"  [{'OK ' if cond else 'FAIL'}] {msg}")
    if not cond:
        check.failed += 1
check.failed = 0


# ------------------------------------------------------------------ 테스트
def test_degrade(img):
    print("\n[1] degrade_resolution")
    h, w = img.shape[:2]
    near = degrade_resolution(img, 0.2, up=cv2.INTER_NEAREST)
    lin = degrade_resolution(img, 0.2, up=cv2.INTER_LINEAR)
    check(near.shape == img.shape, f"크기 유지: {near.shape}")
    check(near.dtype == np.uint8, "dtype uint8 유지")
    check(np.array_equal(degrade_resolution(img, 1.0), img), "scale=1.0이면 원본과 동일")
    # NEAREST는 5x5 블록이 같은 값이어야 한다 (0.2배 → 1픽셀이 5픽셀로)
    check(len(np.unique(near[:5, :5].reshape(-1, 3), axis=0)) == 1, "NEAREST: 5×5 블록이 한 색")
    err_n = np.abs(near.astype(int) - img).mean()
    err_l = np.abs(lin.astype(int) - img).mean()
    print(f"  원본과의 평균 오차  NEAREST={err_n:.2f}  LINEAR={err_l:.2f}")
    for s in (0.5, 0.2, 0.1):
        print(f"  scale={s}: 축소 크기 {int(w*s)}×{int(h*s)}, "
              f"평균 오차 {np.abs(degrade_resolution(img, s, up=cv2.INTER_LINEAR).astype(int) - img).mean():.2f}")
    imwrite(OUT / "1_degrade.png", np.hstack([
        label(img, "original"), label(near, "0.2 AREA->NEAREST"), label(lin, "0.2 AREA->LINEAR")]))
    print(f"  → {OUT / '1_degrade.png'}")


def test_contrast(img):
    print("\n[2] adjust_contrast_brightness")
    check(np.array_equal(adjust_contrast_brightness(img, 1.0, 0), img), "alpha=1, beta=0이면 원본과 동일")
    px = np.array([[[10, 100, 250]]], np.uint8)
    check(adjust_contrast_brightness(px, 1.0, 50)[0, 0].tolist() == [60, 150, 255],
          "beta=+50: 10→60, 100→150, 250→255(넘치면 255로 자름)")
    check(adjust_contrast_brightness(px, 1.0, -50)[0, 0].tolist() == [0, 50, 200],
          "beta=-50: 10→0(음수는 0으로 자름)")
    check(adjust_contrast_brightness(px, 2.0, 0)[0, 0].tolist() == [20, 200, 255], "alpha=2: 값 2배")
    print(f"  참고) 그냥 uint8로 더하면: {(px + np.uint8(50))[0,0].tolist()}  ← 250+50이 44로 돌아감(오버플로)")

    blurry = degrade_resolution(img, 0.2, up=cv2.INTER_LINEAR)
    fixed = adjust_contrast_brightness(blurry, 1.5, -60)
    print(f"  명암 표준편차(대비 지표)  원본={img.std():.1f}  흐림={blurry.std():.1f}  보정={fixed.std():.1f}")
    imwrite(OUT / "2_contrast.png", np.hstack([
        label(blurry, "blurry"),
        label(adjust_contrast_brightness(blurry, 1.0, 50), "a=1.0 b=+50"),
        label(fixed, "a=1.5 b=-60")]))
    print(f"  → {OUT / '2_contrast.png'}")


def test_augment(img):
    print("\n[3] augment_image")
    flipped = augment_image(img, flip=True)
    check(np.array_equal(flipped[:, 0], img[:, -1]), "좌우 반전: 첫 열 == 원본 마지막 열")
    check(np.array_equal(augment_image(flipped, flip=True), img), "두 번 반전하면 원본")
    noisy = augment_image(img, noise_sigma=20, seed=0)
    diff = noisy.astype(int) - img
    print(f"  노이즈 σ=20 → 실제 차이 평균={diff.mean():.2f}, 표준편차={diff.std():.2f}")
    check(abs(diff.mean()) < 2, "노이즈 평균 ≈ 0")
    check(np.array_equal(noisy, augment_image(img, noise_sigma=20, seed=0)), "같은 seed → 같은 결과")
    both = augment_image(img, flip=True, noise_sigma=20, seed=0)
    imwrite(OUT / "3_augment.png", np.hstack([
        label(img, "original"), label(flipped, "flip"),
        label(noisy, "noise s=20"), label(both, "flip+noise")]))
    print(f"  → {OUT / '3_augment.png'}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    img = imread(sys.argv[2]) if len(sys.argv) > 2 else make_test_image()
    if img is None:
        sys.exit("이미지를 읽지 못했습니다.")
    tests = {"degrade": test_degrade, "contrast": test_contrast, "augment": test_augment}
    for name, fn in tests.items():
        if which in ("all", name):
            fn(img)
    print(f"\n실패 {check.failed}개" if check.failed else "\n전부 통과")
