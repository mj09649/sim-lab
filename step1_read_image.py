"""1단계: 이미지가 파이썬 안에서 숫자(배열)로 잘 읽히는지 확인하기.

실행:
    python step1_read_image.py                # 테스트 이미지 자동 생성 후 확인
    python step1_read_image.py 내사진.jpg      # 내 이미지로 확인
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path):
    """cv2.imread는 Windows에서 한글 경로(예: C:\\Users\\민지)를 못 읽고 None을 돌려준다.
    파일을 바이트로 읽은 뒤 cv2.imdecode로 디코딩하면 한글 경로도 OK. 결과는 똑같이 BGR."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def make_test_image(path):
    """왼쪽부터 빨강 / 초록 / 파랑 세로 띠. (OpenCV 기준이니 BGR 순서로 칠한다)"""
    img = np.zeros((90, 90, 3), dtype=np.uint8)
    img[:, 0:30] = (0, 0, 255)    # 빨강 = B0 G0 R255
    img[:, 30:60] = (0, 255, 0)   # 초록
    img[:, 60:90] = (255, 0, 0)   # 파랑 = B255 G0 R0
    ok, enc = cv2.imencode(".png", img)
    enc.tofile(str(path))         # 한글 경로 저장도 imwrite 대신 이렇게


here = Path(__file__).parent
if len(sys.argv) > 1:
    path = Path(sys.argv[1])
else:
    path = here / "test_rgb.png"
    make_test_image(path)
    print(f"테스트 이미지 생성: {path}")

# 1) 그냥 cv2.imread — 한글 경로면 None이 나올 수 있다
plain = cv2.imread(str(path))
print(f"\ncv2.imread 결과: {'None (읽기 실패!)' if plain is None else '성공'}")

# 2) 안전하게 읽기
bgr = imread_unicode(path)
if bgr is None:
    sys.exit("이미지를 읽지 못했습니다. 경로/파일 형식을 확인하세요.")

print("\n=== 숫자로 읽혔는지 확인 ===")
print(f"type  : {type(bgr)}")
print(f"shape : {bgr.shape}   (세로, 가로, 채널)")
print(f"dtype : {bgr.dtype}   (0~255 정수)")
print(f"min/max : {bgr.min()} / {bgr.max()}")

h, w = bgr.shape[:2]
pts = {"왼쪽": (h // 2, w // 6), "가운데": (h // 2, w // 2), "오른쪽": (h // 2, w * 5 // 6)}

# 3) BGR -> RGB 변환 (화면에 보여줄 때 필수)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

print("\n=== 같은 픽셀, 채널 순서 비교 ===")
for name, (y, x) in pts.items():
    print(f"{name:4s} (y={y}, x={x})  BGR={tuple(int(v) for v in bgr[y, x])}  "
          f"RGB={tuple(int(v) for v in rgb[y, x])}")

# 4) 변환 안 하고 RGB 라이브러리(PIL/Streamlit/matplotlib)에 넘기면 빨강↔파랑이 뒤바뀐다
out = here / "step1_compare.png"
wrong = bgr   # BGR을 RGB인 척 넘긴 것
compare = np.hstack([rgb, np.full((h, 10, 3), 255, np.uint8), wrong])
from PIL import Image
Image.fromarray(compare).save(out)
print(f"\n비교 이미지 저장: {out}")
print("  왼쪽 = RGB로 변환 후 (정상 색) / 오른쪽 = 변환 없이 (빨강↔파랑 뒤바뀜)")
