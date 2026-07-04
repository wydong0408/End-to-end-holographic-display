import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. 参数设置
# ============================================================

image_path = r"E:/HoloD/Paper/NC/R1/MTF/a3.jpg"

save_dir = r"E:/HoloD/Paper/NC/R1/MTF//grating_profile"
os.makedirs(save_dir, exist_ok=True)

# ROI大小，根据你的条纹patch大小调整
ROI_W = 180
ROI_H = 120

# 自动检测参数
THRESHOLD_RATIO = 0.20
MIN_AREA = 100
MAX_AREA = 100000

# 是否手动指定9个ROI中心
USE_MANUAL_CENTERS = False

# 如果自动检测不准，改成 True，并手动填写中心坐标
MANUAL_CENTERS = [
    # (x, y)
    # (380, 220), (760, 220), (1140, 220),
    # (380, 450), (760, 450), (1140, 450),
    # (380, 680), (760, 680), (1140, 680)
]

# 为了避免亮边框影响剖面，可以在ROI内部再裁掉边缘
INNER_MARGIN_X = 15
INNER_MARGIN_Y = 15

# 剖面是否归一化
NORMALIZE_PROFILE = True


# ============================================================
# 2. 读取图像
# ============================================================

img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

if img_bgr is None:
    raise FileNotFoundError(image_path)

# 只取绿色通道
green = img_bgr[:, :, 1].astype(np.float64)

H, W = green.shape
print("Image size:", W, H)


# ============================================================
# 3. 自动检测九个亮区域中心
# ============================================================

def sort_3x3_targets(targets):
    """
    排序为：
    F1 F2 F3
    F4 F5 F6
    F7 F8 F9
    """
    targets = sorted(targets, key=lambda p: p[1])
    rows = [targets[0:3], targets[3:6], targets[6:9]]

    sorted_targets = []
    for row in rows:
        row = sorted(row, key=lambda p: p[0])
        sorted_targets.extend(row)

    return sorted_targets


def detect_9_targets(gray):
    img = gray.copy()

    # 扣除暗背景
    img = img - np.percentile(img, 2)
    img[img < 0] = 0

    img_norm = img / (img.max() + 1e-12)

    mask = (img_norm > THRESHOLD_RATIO).astype(np.uint8)

    # 形态学闭运算，连接破碎条纹
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    targets = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if MIN_AREA <= area <= MAX_AREA:
            cx, cy = centroids[i]
            targets.append((cx, cy, area))

    targets = sorted(targets, key=lambda p: p[2], reverse=True)[:9]

    if len(targets) == 9:
        targets = sort_3x3_targets(targets)
    else:
        print(f"Warning: detected {len(targets)} targets, not 9.")
        targets = sorted(targets, key=lambda p: (p[1], p[0]))

    return targets, mask


if USE_MANUAL_CENTERS:
    targets = [(float(x), float(y), 1.0) for x, y in MANUAL_CENTERS]
    mask = None
else:
    targets, mask = detect_9_targets(green)

print("\nDetected / used centers:")
for i, (cx, cy, area) in enumerate(targets):
    print(f"F{i+1}: x={cx:.1f}, y={cy:.1f}, area={area:.1f}")


# ============================================================
# 4. 裁剪ROI
# ============================================================

def crop_roi(img, cx, cy, roi_w, roi_h):
    h, w = img.shape

    x0 = int(round(cx - roi_w / 2))
    x1 = x0 + roi_w

    y0 = int(round(cy - roi_h / 2))
    y1 = y0 + roi_h

    roi = np.zeros((roi_h, roi_w), dtype=np.float64)

    xs0 = max(x0, 0)
    xs1 = min(x1, w)
    ys0 = max(y0, 0)
    ys1 = min(y1, h)

    xd0 = xs0 - x0
    yd0 = ys0 - y0

    roi[yd0:yd0 + (ys1 - ys0), xd0:xd0 + (xs1 - xs0)] = img[ys0:ys1, xs0:xs1]

    return roi


rois = []

for cx, cy, area in targets:
    roi = crop_roi(green, cx, cy, ROI_W, ROI_H)
    rois.append(roi)


# ============================================================
# 5. 提取剖面图
# ============================================================

def get_x_profile_from_roi(roi):
    """
    对竖直条纹：
    沿 y 方向平均，得到 x 方向强度曲线。
    """
    h, w = roi.shape

    x0 = INNER_MARGIN_X
    x1 = w - INNER_MARGIN_X
    y0 = INNER_MARGIN_Y
    y1 = h - INNER_MARGIN_Y

    inner = roi[y0:y1, x0:x1]

    # 扣背景
    inner = inner - np.percentile(inner, 2)
    inner[inner < 0] = 0

    # 沿 y 方向平均
    profile = inner.mean(axis=0)

    if NORMALIZE_PROFILE:
        profile = profile - np.min(profile)
        if np.max(profile) > 0:
            profile = profile / np.max(profile)

    x = np.arange(len(profile))

    return x, profile, inner


def michelson_contrast(profile):
    """
    用百分位数而不是绝对最大最小，降低speckle异常点影响。
    """
    Imax = np.percentile(profile, 95)
    Imin = np.percentile(profile, 5)

    C = (Imax - Imin) / (Imax + Imin + 1e-12)

    return C, Imax, Imin


profiles = []
contrasts = []
inner_rois = []

for roi in rois:
    x, profile, inner = get_x_profile_from_roi(roi)
    C, Imax, Imin = michelson_contrast(profile)

    profiles.append((x, profile))
    contrasts.append(C)
    inner_rois.append(inner)


# ============================================================
# 6. 显示九个ROI
# ============================================================

plt.figure(figsize=(9, 8))

for i, roi in enumerate(rois):
    plt.subplot(3, 3, i + 1)
    plt.imshow(roi, cmap="gray")
    plt.title(f"F{i+1}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "roi_9field.png"), dpi=300)


# ============================================================
# 7. 显示九个内部ROI
# ============================================================

plt.figure(figsize=(9, 8))

for i, inner in enumerate(inner_rois):
    plt.subplot(3, 3, i + 1)
    plt.imshow(inner, cmap="gray")
    plt.title(f"F{i+1}, C={contrasts[i]:.3f}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "inner_roi_9field.png"), dpi=300)


# ============================================================
# 8. 画九个剖面图
# ============================================================

plt.figure(figsize=(10, 8))

for i, (x, profile) in enumerate(profiles):
    plt.subplot(3, 3, i + 1)
    plt.plot(x, profile, "b-", linewidth=1.2)
    plt.title(f"F{i+1}, C={contrasts[i]:.3f}")
    plt.xlabel("x pixels")
    plt.ylabel("Intensity")
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "profile_9field.png"), dpi=300)


# ============================================================
# 9. 叠加显示九个剖面
# ============================================================

plt.figure(figsize=(8, 5))

for i, (x, profile) in enumerate(profiles):
    plt.plot(x, profile, label=f"F{i+1}, C={contrasts[i]:.2f}")

plt.xlabel("x pixels")
plt.ylabel("Normalized intensity" if NORMALIZE_PROFILE else "Intensity")
plt.title("X profiles of 9 field ROIs")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "profile_overlay.png"), dpi=300)


# ============================================================
# 10. 保存contrast结果
# ============================================================

txt_path = os.path.join(save_dir, "contrast_results.txt")

with open(txt_path, "w", encoding="utf-8") as f:
    for i, C in enumerate(contrasts):
        line = f"Field {i+1}: Michelson contrast = {C:.4f}"
        print(line)
        f.write(line + "\n")

print("\nSaved results to:", save_dir)

plt.show()