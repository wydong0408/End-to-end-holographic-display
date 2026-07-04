import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import cv2


# ============================================================
# 1. 参数设置
# ============================================================

image_path = r"E:/HoloD/Paper/NC/R1/MTF/a4.png"

save_dir = r"E:/HoloD/Paper/NC/R1/MTF/grating_profile"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# 视场角设置
# -----------------------------
FOV_X_DEG = 16.0     # 虚像水平视场角，单位 degree
FOV_Y_DEG = 8.0      # 虚像垂直视场角，单位 degree；不知道可设为 None

# -----------------------------
# 虚像区域设置
# -----------------------------
# 如果输入图像已经是裁剪后的虚像区域，设为 None
# 如果输入图像是完整 iPhone 照片，手动填虚像区域 bbox:
# VIRTUAL_BBOX = (x0, y0, x1, y1)
# 注意 x1, y1 是右下角坐标，不包含该像素
VIRTUAL_BBOX = None

# 示例：
# VIRTUAL_BBOX = (120, 80, 3920, 2050)

# -----------------------------
# ROI 设置
# -----------------------------
ROI_W = 280       # ROI 宽度，单位 camera pixel
ROI_H = 180       # ROI 高度，单位 camera pixel

# 九个视场位置在虚像区域内的相对位置
GRID_X_FRACS = [0.20, 0.50, 0.80]
GRID_Y_FRACS = [0.20, 0.50, 0.80]

# 如果想手动指定 9 个 ROI 中心，设为 True
USE_MANUAL_CENTERS = False
MANUAL_CENTERS = [
    # (x, y), 坐标是输入图像坐标，不是局部 bbox 坐标
    # (800, 500), (2000, 500), (3200, 500),
    # (800, 1100), (2000, 1100), (3200, 1100),
    # (800, 1700), (2000, 1700), (3200, 1700)
]

# ROI 内部裁剪，避免边缘、亮框、畸变影响
INNER_MARGIN_X = 20
INNER_MARGIN_Y = 20

# -----------------------------
# 背景扣除设置
# -----------------------------
# 对于全息显示，局部背景会影响条纹调制度。
# profile_raw: 不扣背景
# profile_bg: 扣除局部背景
# profile_norm: 仅用于画图
BACKGROUND_PERCENTILE = 2

# -----------------------------
# 对比度计算设置
# -----------------------------
# "percentile": 用 95% 和 5% 分位数，抗 speckle，推荐用于全息图
# "maxmin": 用最大值和最小值，更接近原始 Michelson 定义，但易受异常点影响
CONTRAST_MODE = "percentile"

# 计算哪个对比度作为主要结果：
# "bg": 使用扣背景后的 profile_bg，推荐
# "raw": 使用未扣背景的 profile_raw
MAIN_CONTRAST_PROFILE = "bg"

# -----------------------------
# 频率估计设置
# -----------------------------
ESTIMATE_CPD_BY_FFT = True

# 如果知道输入图案的理论 cpd，可填入用于对比；不知道填 None
KNOWN_CPD = None
# KNOWN_CPD = 5.0

# 绘图 x 轴单位："pixel" 或 "degree"
X_AXIS_UNIT = "degree"

# 是否保存图像
SAVE_FIGURES = True


# ============================================================
# 2. 读取图像
# ============================================================

img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

if img_bgr is None:
    raise FileNotFoundError(image_path)

# 取绿色通道
green = img_bgr[:, :, 1].astype(np.float64)

H_img, W_img = green.shape
print("Input image size:", W_img, H_img)


# ============================================================
# 3. 确定虚像区域与角度采样
# ============================================================

if VIRTUAL_BBOX is None:
    vx0, vy0, vx1, vy1 = 0, 0, W_img, H_img
else:
    vx0, vy0, vx1, vy1 = VIRTUAL_BBOX

N_vir_camera_x = vx1 - vx0
N_vir_camera_y = vy1 - vy0

theta_x_deg_per_px = FOV_X_DEG / N_vir_camera_x

if FOV_Y_DEG is not None:
    theta_y_deg_per_px = FOV_Y_DEG / N_vir_camera_y
else:
    theta_y_deg_per_px = None

print("\nVirtual image region:")
print(f"x0={vx0}, y0={vy0}, x1={vx1}, y1={vy1}")
print(f"N_vir_camera_x = {N_vir_camera_x} px")
print(f"N_vir_camera_y = {N_vir_camera_y} px")
print(f"theta_x = {theta_x_deg_per_px:.8f} deg/pixel")
print(f"Camera sampling Nyquist-x = {0.5 / theta_x_deg_per_px:.2f} cpd")

if theta_y_deg_per_px is not None:
    print(f"theta_y = {theta_y_deg_per_px:.8f} deg/pixel")
    print(f"Camera sampling Nyquist-y = {0.5 / theta_y_deg_per_px:.2f} cpd")

if KNOWN_CPD is not None:
    expected_period_camera_px = 1.0 / (KNOWN_CPD * theta_x_deg_per_px)
    print(f"Expected camera period at {KNOWN_CPD:.2f} cpd = {expected_period_camera_px:.2f} px")


# ============================================================
# 4. 生成九个 ROI 中心
# ============================================================

def generate_9_centers_from_bbox(vx0, vy0, vx1, vy1, x_fracs, y_fracs):
    centers = []

    width = vx1 - vx0
    height = vy1 - vy0

    for fy in y_fracs:
        for fx in x_fracs:
            cx = vx0 + fx * width
            cy = vy0 + fy * height
            centers.append((cx, cy, 1.0))

    return centers


if USE_MANUAL_CENTERS:
    targets = [(float(x), float(y), 1.0) for x, y in MANUAL_CENTERS]
else:
    targets = generate_9_centers_from_bbox(
        vx0, vy0, vx1, vy1,
        GRID_X_FRACS,
        GRID_Y_FRACS
    )

print("\nUsed ROI centers:")
for i, (cx, cy, area) in enumerate(targets):
    print(f"F{i+1}: x={cx:.1f}, y={cy:.1f}")


# ============================================================
# 5. 裁剪 ROI
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
# 6. 提取 x 方向剖面
# ============================================================

def get_x_profile_from_roi(roi):
    """
    针对竖直条纹：
    沿 y 方向平均，得到 x 方向强度曲线。

    返回：
    x_px: 像素坐标
    x_deg: 角度坐标
    profile_norm: 归一化剖面，仅用于画图
    profile_raw: 未扣背景强度剖面，用于 raw contrast
    profile_bg: 扣背景强度剖面，用于 background-corrected contrast
    inner_raw: 内部 ROI 原始图像
    inner_bg: 内部 ROI 扣背景图像
    bg: 扣除的背景值
    """

    h, w = roi.shape

    x0 = INNER_MARGIN_X
    x1 = w - INNER_MARGIN_X
    y0 = INNER_MARGIN_Y
    y1 = h - INNER_MARGIN_Y

    if x1 <= x0 or y1 <= y0:
        raise ValueError("INNER_MARGIN_X or INNER_MARGIN_Y is too large for the ROI size.")

    inner_raw = roi[y0:y1, x0:x1].astype(np.float64)

    # 未扣背景的 profile
    profile_raw = inner_raw.mean(axis=0)

    # 扣除局部背景
    bg = np.percentile(inner_raw, BACKGROUND_PERCENTILE)
    inner_bg = inner_raw - bg
    inner_bg[inner_bg < 0] = 0

    profile_bg = inner_bg.mean(axis=0)

    # 归一化 profile，仅用于画图
    profile_norm = profile_bg.copy()
    profile_norm = profile_norm - np.min(profile_norm)
    max_val = np.max(profile_norm)

    if max_val > 0:
        profile_norm = profile_norm / max_val

    x_px = np.arange(len(profile_norm))
    x_deg = x_px * theta_x_deg_per_px

    return x_px, x_deg, profile_norm, profile_raw, profile_bg, inner_raw, inner_bg, bg


# ============================================================
# 7. Michelson contrast 与 FFT 主频估计
# ============================================================

def michelson_contrast(profile, mode="percentile"):
    """
    计算 Michelson contrast:
        C = (Imax - Imin) / (Imax + Imin)

    mode:
        "percentile": Imax=95 percentile, Imin=5 percentile，抗 speckle
        "maxmin": Imax=max, Imin=min，标准形式但易受异常点影响
    """

    p = np.asarray(profile, dtype=np.float64)

    if p.size == 0:
        return np.nan, np.nan, np.nan

    if not np.any(np.isfinite(p)):
        return np.nan, np.nan, np.nan

    p = p[np.isfinite(p)]

    if mode == "percentile":
        Imax = np.percentile(p, 95)
        Imin = np.percentile(p, 5)

    elif mode == "maxmin":
        Imax = np.max(p)
        Imin = np.min(p)

    else:
        raise ValueError("Unknown CONTRAST_MODE. Use 'percentile' or 'maxmin'.")

    denom = Imax + Imin

    if denom <= 1e-12:
        C = np.nan
    else:
        C = (Imax - Imin) / denom

    return C, Imax, Imin


def estimate_dominant_cpd_fft(profile, theta_x_deg_per_px):
    """
    用 FFT 从 profile 中估计主频，单位 cpd。
    只用于检查 measured cpd 是否接近理论 cpd。
    """

    p = np.asarray(profile, dtype=np.float64)

    if p.size < 4:
        return np.nan, np.nan

    p = p - np.mean(p)

    if np.std(p) < 1e-12:
        return np.nan, np.nan

    # 加 Hann 窗减少边缘泄漏
    win = np.hanning(len(p))
    p_win = p * win

    spec = np.abs(np.fft.rfft(p_win))
    freq_cpp = np.fft.rfftfreq(len(p_win), d=1.0)  # cycles / camera pixel

    if len(spec) < 3:
        return np.nan, np.nan

    # 忽略 DC
    spec[0] = 0

    peak_idx = np.argmax(spec)

    f_cpp = freq_cpp[peak_idx]
    f_cpd = f_cpp / theta_x_deg_per_px

    if f_cpp <= 0:
        period_px = np.nan
    else:
        period_px = 1.0 / f_cpp

    return f_cpd, period_px


# ============================================================
# 8. 计算九个 ROI 的 profile、contrast、cpd
# ============================================================

profiles_norm = []
profiles_raw = []
profiles_bg = []

contrasts_main = []
contrasts_raw = []
contrasts_bg = []

Imax_main_list = []
Imin_main_list = []

Imax_raw_list = []
Imin_raw_list = []

Imax_bg_list = []
Imin_bg_list = []

inner_rois_raw = []
inner_rois_bg = []

bg_values = []

measured_cpds = []
measured_periods_px = []

for roi in rois:
    x_px, x_deg, profile_norm, profile_raw, profile_bg, inner_raw, inner_bg, bg = get_x_profile_from_roi(roi)

    # raw contrast：未扣背景
    C_raw, Imax_raw, Imin_raw = michelson_contrast(profile_raw, mode=CONTRAST_MODE)

    # background-corrected contrast：扣背景
    C_bg, Imax_bg, Imin_bg = michelson_contrast(profile_bg, mode=CONTRAST_MODE)

    if MAIN_CONTRAST_PROFILE == "raw":
        C_main = C_raw
        Imax_main = Imax_raw
        Imin_main = Imin_raw
    elif MAIN_CONTRAST_PROFILE == "bg":
        C_main = C_bg
        Imax_main = Imax_bg
        Imin_main = Imin_bg
    else:
        raise ValueError("MAIN_CONTRAST_PROFILE should be 'raw' or 'bg'.")

    if ESTIMATE_CPD_BY_FFT:
        # 主频估计建议用扣背景但未归一化的 profile
        f_cpd_meas, period_px_meas = estimate_dominant_cpd_fft(profile_bg, theta_x_deg_per_px)
    else:
        f_cpd_meas, period_px_meas = np.nan, np.nan

    profiles_norm.append((x_px, x_deg, profile_norm))
    profiles_raw.append(profile_raw)
    profiles_bg.append(profile_bg)

    contrasts_main.append(C_main)
    contrasts_raw.append(C_raw)
    contrasts_bg.append(C_bg)

    Imax_main_list.append(Imax_main)
    Imin_main_list.append(Imin_main)

    Imax_raw_list.append(Imax_raw)
    Imin_raw_list.append(Imin_raw)

    Imax_bg_list.append(Imax_bg)
    Imin_bg_list.append(Imin_bg)

    inner_rois_raw.append(inner_raw)
    inner_rois_bg.append(inner_bg)

    bg_values.append(bg)

    measured_cpds.append(f_cpd_meas)
    measured_periods_px.append(period_px_meas)


# ============================================================
# 9. 显示虚像区域和 9 个 ROI 位置
# ============================================================

overlay = img_bgr.copy()

cv2.rectangle(
    overlay,
    (int(vx0), int(vy0)),
    (int(vx1), int(vy1)),
    (0, 0, 255),
    3
)

for i, (cx, cy, area) in enumerate(targets):
    x0 = int(round(cx - ROI_W / 2))
    y0 = int(round(cy - ROI_H / 2))
    x1 = x0 + ROI_W
    y1 = y0 + ROI_H

    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)
    cv2.putText(
        overlay,
        f"F{i+1}",
        (x0 + 5, y0 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2
    )

plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Virtual image region and 9 ROI positions")
plt.axis("off")
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(os.path.join(save_dir, "virtual_region_and_9roi.png"), dpi=300)


# ============================================================
# 10. 显示九个原始 ROI
# ============================================================

plt.figure(figsize=(9, 8))

for i, roi in enumerate(rois):
    plt.subplot(3, 3, i + 1)
    plt.imshow(roi, cmap="gray")
    title = f"F{i+1}"
    if ESTIMATE_CPD_BY_FFT:
        title += f"\n{measured_cpds[i]:.2f} cpd"
    plt.title(title)
    plt.axis("off")

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(os.path.join(save_dir, "roi_9field.png"), dpi=300)


# ============================================================
# 11. 显示九个内部 ROI：raw
# ============================================================

plt.figure(figsize=(9, 8))

for i, inner in enumerate(inner_rois_raw):
    plt.subplot(3, 3, i + 1)
    plt.imshow(inner, cmap="gray")

    title = f"F{i+1}, C_raw={contrasts_raw[i]:.3f}"
    if ESTIMATE_CPD_BY_FFT:
        title += f"\n{measured_cpds[i]:.2f} cpd"

    plt.title(title)
    plt.axis("off")

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(os.path.join(save_dir, "inner_roi_raw_9field.png"), dpi=300)


# ============================================================
# 12. 显示九个内部 ROI：background-corrected
# ============================================================

plt.figure(figsize=(9, 8))

for i, inner in enumerate(inner_rois_bg):
    plt.subplot(3, 3, i + 1)
    plt.imshow(inner, cmap="gray")

    title = f"F{i+1}, C_bg={contrasts_bg[i]:.3f}"
    if ESTIMATE_CPD_BY_FFT:
        title += f"\n{measured_cpds[i]:.2f} cpd"

    plt.title(title)
    plt.axis("off")

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(os.path.join(save_dir, "inner_roi_bg_9field.png"), dpi=300)


# ============================================================
# 13. 画九个剖面图：归一化 profile，仅用于显示
# ============================================================

plt.figure(figsize=(11, 8))

for i, (x_px, x_deg, profile_norm) in enumerate(profiles_norm):
    plt.subplot(3, 3, i + 1)

    if X_AXIS_UNIT == "degree":
        x_plot = x_deg
        xlabel = "x angle (degree)"
    else:
        x_plot = x_px
        xlabel = "x (camera pixels)"

    plt.plot(x_plot, profile_norm, "b-", linewidth=1.2)

    title = f"F{i+1}, C={contrasts_main[i]:.3f}"
    if ESTIMATE_CPD_BY_FFT:
        title += f"\n{measured_cpds[i]:.2f} cpd"

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Normalized intensity")
    plt.grid(True)

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(os.path.join(save_dir, "profile_norm_9field.png"), dpi=300)


# ============================================================
# 14. 叠加显示九个归一化剖面
# ============================================================

plt.figure(figsize=(8, 5))

for i, (x_px, x_deg, profile_norm) in enumerate(profiles_norm):
    if X_AXIS_UNIT == "degree":
        x_plot = x_deg
        xlabel = "x angle (degree)"
    else:
        x_plot = x_px
        xlabel = "x (camera pixels)"

    label = f"F{i+1}, C={contrasts_main[i]:.2f}"
    if ESTIMATE_CPD_BY_FFT:
        label += f", {measured_cpds[i]:.1f} cpd"

    plt.plot(x_plot, profile_norm, label=label)

plt.xlabel(xlabel)
plt.ylabel("Normalized intensity")
plt.title("X profiles of 9 field ROIs")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(os.path.join(save_dir, "profile_overlay_norm.png"), dpi=300)


# ============================================================
# 15. 叠加显示 raw profile
# ============================================================

plt.figure(figsize=(8, 5))

for i, profile_raw in enumerate(profiles_raw):
    x_px, x_deg, _ = profiles_norm[i]

    if X_AXIS_UNIT == "degree":
        x_plot = x_deg
        xlabel = "x angle (degree)"
    else:
        x_plot = x_px
        xlabel = "x (camera pixels)"

    plt.plot(x_plot, profile_raw, label=f"F{i+1}, C_raw={contrasts_raw[i]:.2f}")

plt.xlabel(xlabel)
plt.ylabel("Raw intensity")
plt.title("Raw X profiles of 9 field ROIs")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(os.path.join(save_dir, "profile_overlay_raw.png"), dpi=300)


# ============================================================
# 16. 叠加显示 background-corrected profile
# ============================================================

plt.figure(figsize=(8, 5))

for i, profile_bg in enumerate(profiles_bg):
    x_px, x_deg, _ = profiles_norm[i]

    if X_AXIS_UNIT == "degree":
        x_plot = x_deg
        xlabel = "x angle (degree)"
    else:
        x_plot = x_px
        xlabel = "x (camera pixels)"

    plt.plot(x_plot, profile_bg, label=f"F{i+1}, C_bg={contrasts_bg[i]:.2f}")

plt.xlabel(xlabel)
plt.ylabel("Background-corrected intensity")
plt.title("Background-corrected X profiles of 9 field ROIs")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(os.path.join(save_dir, "profile_overlay_bg.png"), dpi=300)


# ============================================================
# 17. 保存结果 CSV
# ============================================================

csv_path = os.path.join(save_dir, "contrast_and_cpd_results.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "field",
        "center_x_px",
        "center_y_px",

        "contrast_main",
        "main_contrast_profile",
        "contrast_mode",

        "contrast_raw",
        "Imax_raw",
        "Imin_raw",

        "contrast_bg",
        "Imax_bg",
        "Imin_bg",
        "background_value",

        "measured_cpd_fft",
        "measured_period_camera_px",

        "theta_x_deg_per_px",
        "FOV_X_DEG",
        "N_vir_camera_x",
        "known_cpd"
    ])

    for i, (cx, cy, area) in enumerate(targets):
        writer.writerow([
            i + 1,
            cx,
            cy,

            contrasts_main[i],
            MAIN_CONTRAST_PROFILE,
            CONTRAST_MODE,

            contrasts_raw[i],
            Imax_raw_list[i],
            Imin_raw_list[i],

            contrasts_bg[i],
            Imax_bg_list[i],
            Imin_bg_list[i],
            bg_values[i],

            measured_cpds[i],
            measured_periods_px[i],

            theta_x_deg_per_px,
            FOV_X_DEG,
            N_vir_camera_x,
            KNOWN_CPD
        ])


# ============================================================
# 18. 保存 TXT
# ============================================================

txt_path = os.path.join(save_dir, "contrast_results.txt")

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"Input image size: {W_img} x {H_img}\n")
    f.write(f"Virtual bbox: {vx0}, {vy0}, {vx1}, {vy1}\n")
    f.write(f"N_vir_camera_x = {N_vir_camera_x} px\n")
    f.write(f"N_vir_camera_y = {N_vir_camera_y} px\n")
    f.write(f"FOV_X_DEG = {FOV_X_DEG}\n")
    f.write(f"FOV_Y_DEG = {FOV_Y_DEG}\n")
    f.write(f"theta_x_deg_per_px = {theta_x_deg_per_px:.8f}\n")
    f.write(f"Nyquist-x = {0.5 / theta_x_deg_per_px:.4f} cpd\n")
    f.write(f"CONTRAST_MODE = {CONTRAST_MODE}\n")
    f.write(f"MAIN_CONTRAST_PROFILE = {MAIN_CONTRAST_PROFILE}\n")
    f.write(f"BACKGROUND_PERCENTILE = {BACKGROUND_PERCENTILE}\n\n")

    for i, C in enumerate(contrasts_main):
        line = (
            f"Field {i+1}: "
            f"C_main = {C:.4f}, "
            f"C_raw = {contrasts_raw[i]:.4f}, "
            f"C_bg = {contrasts_bg[i]:.4f}, "
            f"measured_cpd = {measured_cpds[i]:.4f}, "
            f"period_camera_px = {measured_periods_px[i]:.4f}, "
            f"bg = {bg_values[i]:.4f}"
        )
        print(line)
        f.write(line + "\n")

print("\nSaved results to:", save_dir)
print("CSV:", csv_path)
print("TXT:", txt_path)

plt.show()