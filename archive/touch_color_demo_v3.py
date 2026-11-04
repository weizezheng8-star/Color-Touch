
"""
TOUCH COLOR 光触感知器 — Python 功能演示

功能：
1. 摄像头实时预览，读取中央识别框内的颜色
2. 识别中文颜色名称、RGB、亮度、冷暖色倾向
3. 通过图像清晰度近似判断“平滑 / 轻纹理 / 强纹理”
4. Lab 空间 K-means 主体色提取与扩展色卡匹配
5. 支持白色校准、多帧降噪、本地图片识别
6. 支持语音播报（安装 pyttsx3 后自动启用）
7. 摄像头不可用时可进入模拟模式

快捷键：
Space：识别当前画面
O：打开图片
V：语音播报
C：白色校准
Q / Esc：退出
"""

from __future__ import annotations

import math
import random
from collections import deque
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk


# -----------------------------
# 可选语音模块
# -----------------------------
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


@dataclass
class ColorResult:
    name: str
    rgb: Tuple[int, int, int]
    brightness: int
    temperature: str
    texture: str
    confidence: int


# 扩展色卡：在基础色之外增加深浅层级，减少“相邻颜色”误判
PALETTE = {
    "黑色": (15, 15, 18),
    "炭黑色": (38, 40, 45),
    "深灰色": (70, 72, 78),
    "灰色": (128, 128, 132),
    "浅灰色": (198, 200, 205),
    "白色": (238, 238, 235),
    "深红色": (135, 25, 35),
    "红色": (220, 45, 45),
    "橙红色": (235, 80, 35),
    "橙色": (240, 130, 30),
    "黄色": (235, 205, 40),
    "黄绿色": (155, 190, 45),
    "绿色": (45, 155, 70),
    "墨绿色": (25, 90, 60),
    "青绿色": (25, 160, 145),
    "青色": (30, 185, 195),
    "天蓝色": (80, 165, 225),
    "蓝色": (45, 90, 210),
    "深蓝色": (30, 45, 115),
    "紫色": (125, 70, 175),
    "紫红色": (180, 55, 145),
    "粉色": (230, 135, 165),
    "棕色": (125, 75, 45),
    "卡其色": (175, 155, 105),
    "米色": (220, 205, 170),
}


def rgb_to_lab(rgb: Tuple[int, int, int]) -> np.ndarray:
    """把 RGB 转换到更接近人眼感知的 Lab 空间。"""
    sample = np.uint8([[list(rgb)]])
    return cv2.cvtColor(sample, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)


PALETTE_LAB = {name: rgb_to_lab(rgb) for name, rgb in PALETTE.items()}


def nearest_color_name(
    rgb: Tuple[int, int, int], cluster_ratio: float = 1.0
) -> Tuple[str, int]:
    """在 Lab 空间内匹配最近色卡，并给出保守置信度。"""
    target = rgb_to_lab(rgb)
    distances = {
        name: float(np.linalg.norm(target - lab))
        for name, lab in PALETTE_LAB.items()
    }
    ordered = sorted(distances.items(), key=lambda item: item[1])
    name, distance = ordered[0]

    # 距离越小、目标区域颜色越统一，置信度越高
    color_score = max(0.0, 100.0 - distance * 2.35)
    uniform_score = max(0.0, min(100.0, cluster_ratio * 100.0))
    confidence = int(round(color_score * 0.82 + uniform_score * 0.18))

    # 低置信度时不装作“绝对准确”，用“接近”提示
    if confidence < 58:
        name = f"接近{name}"
    return name, confidence


def get_center_roi(frame_bgr: np.ndarray, ratio: float = 0.22) -> np.ndarray:
    """提取较小的中央区域，避免背景混入。"""
    h, w = frame_bgr.shape[:2]
    box_w = max(48, int(w * ratio))
    box_h = max(48, int(h * ratio))
    x1 = max(0, w // 2 - box_w // 2)
    y1 = max(0, h // 2 - box_h // 2)
    return frame_bgr[y1:y1 + box_h, x1:x1 + box_w]


def apply_white_balance(
    rgb_image: np.ndarray, gains: Tuple[float, float, float]
) -> np.ndarray:
    """应用由白色校准得到的 RGB 增益。"""
    corrected = rgb_image.astype(np.float32)
    corrected[..., 0] *= gains[0]
    corrected[..., 1] *= gains[1]
    corrected[..., 2] *= gains[2]
    return np.clip(corrected, 0, 255).astype(np.uint8)


def dominant_rgb_from_roi(
    roi_bgr: np.ndarray,
    white_balance_gains: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[Tuple[int, int, int], float]:
    """通过 Lab 空间 K-means 提取主体色，而非直接取整块区域中位数。"""
    roi = cv2.resize(roi_bgr, (120, 120), interpolation=cv2.INTER_AREA)
    roi = cv2.medianBlur(roi, 5)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    rgb = apply_white_balance(rgb, white_balance_gains)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)

    # 去掉极少量过曝与死黑像素，保留白、灰、黑色主体本身
    lightness = pixels[:, 0]
    low, high = np.percentile(lightness, [1.5, 98.5])
    valid = pixels[(lightness >= low) & (lightness <= high)]
    if len(valid) < 100:
        valid = pixels

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.25,
    )
    k = 4
    _compactness, labels, centers = cv2.kmeans(
        valid,
        k,
        None,
        criteria,
        6,
        cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    dominant_index = int(np.argmax(counts))
    dominant_lab = np.uint8([[np.clip(centers[dominant_index], 0, 255)]])
    dominant_rgb = cv2.cvtColor(dominant_lab, cv2.COLOR_LAB2RGB)[0, 0]
    cluster_ratio = float(counts[dominant_index] / counts.sum())

    return tuple(int(v) for v in dominant_rgb), cluster_ratio


def analyze_roi(
    frame_bgr: np.ndarray,
    white_balance_gains: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ColorResult:
    """中央区域分析：白平衡校正 + 主体色聚类 + Lab 色卡匹配。"""
    roi = get_center_roi(frame_bgr)
    dominant_rgb, cluster_ratio = dominant_rgb_from_roi(
        roi, white_balance_gains
    )
    name, confidence = nearest_color_name(dominant_rgb, cluster_ratio)

    hsv = cv2.cvtColor(
        np.uint8([[list(dominant_rgb)]]), cv2.COLOR_RGB2HSV
    )[0, 0]
    hue, saturation, value = int(hsv[0]), int(hsv[1]), int(hsv[2])
    brightness = int(round(value / 255 * 100))

    if saturation < 32:
        temperature = "中性色"
    elif hue < 45 or hue >= 165:
        temperature = "暖色系"
    elif 45 <= hue < 95:
        temperature = "中性偏冷"
    else:
        temperature = "冷色系"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if texture_score < 65:
        texture = "表面较平滑"
    elif texture_score < 210:
        texture = "轻微纹理"
    else:
        texture = "纹理明显"

    return ColorResult(
        name=name,
        rgb=dominant_rgb,
        brightness=brightness,
        temperature=temperature,
        texture=texture,
        confidence=confidence,
    )



class TouchColorApp:
    """TOUCH COLOR V3：保留识别算法，升级为现代化竞赛展示界面。"""

    BG = "#070911"
    CARD = "#10131E"
    CARD_2 = "#151927"
    BORDER = "#282E43"
    PRIMARY = "#705CFF"
    PRIMARY_HOVER = "#836FFF"
    CYAN = "#2DD8E8"
    GREEN = "#72E39B"
    ORANGE = "#FFB45E"
    TEXT = "#F6F7FB"
    SUBTEXT = "#98A2B8"
    MUTED = "#69748C"

    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("TOUCH COLOR · 光触感知器")
        self.root.geometry("1380x820")
        self.root.minsize(1120, 700)
        self.root.configure(fg_color=self.BG)

        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.result: Optional[ColorResult] = None
        self.live_mode = True
        self.simulation_mode = False
        self.frame_buffer = deque(maxlen=7)
        self.white_balance_gains = (1.0, 1.0, 1.0)

        self.tts = None
        if pyttsx3 is not None:
            try:
                self.tts = pyttsx3.init()
                self.tts.setProperty("rate", 165)
            except Exception:
                self.tts = None

        self._build_ui()
        self._bind_keys()
        self._open_camera()
        self._update_video()

    # -----------------------------
    # UI
    # -----------------------------
    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # 顶部导航
        header = ctk.CTkFrame(self.root, fg_color="transparent", height=86)
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(22, 10))
        header.grid_columnconfigure(1, weight=1)

        logo = ctk.CTkLabel(
            header,
            text="T●UCH\nCOLOR",
            justify="left",
            font=ctk.CTkFont(family="Arial", size=25, weight="bold"),
            text_color=self.TEXT,
        )
        logo.grid(row=0, column=0, sticky="w")

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.grid(row=0, column=1, sticky="w", padx=28)
        ctk.CTkLabel(
            title_box,
            text="光触感知器",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=20, weight="bold"),
            text_color=self.TEXT,
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_box,
            text="COLOR · TEXTURE · ACCESSIBLE FEEDBACK",
            font=ctk.CTkFont(family="Arial", size=10),
            text_color=self.SUBTEXT,
        ).pack(anchor="w", pady=(3, 0))

        self.status_pill = ctk.CTkFrame(
            header,
            fg_color="#12242B",
            corner_radius=18,
            height=36,
        )
        self.status_pill.grid(row=0, column=2, sticky="e")
        self.status_label = ctk.CTkLabel(
            self.status_pill,
            text="●  正在连接摄像头",
            text_color=self.CYAN,
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=11, weight="bold"),
        )
        self.status_label.pack(padx=16, pady=8)

        # 主体布局
        body = ctk.CTkFrame(self.root, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=30, pady=(0, 26))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=21)
        body.grid_columnconfigure(1, weight=10)

        # 左侧实时画面卡片
        preview_card = ctk.CTkFrame(
            body,
            fg_color=self.CARD,
            corner_radius=22,
            border_width=1,
            border_color=self.BORDER,
        )
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        preview_card.grid_columnconfigure(0, weight=1)
        preview_card.grid_rowconfigure(1, weight=1)

        preview_head = ctk.CTkFrame(preview_card, fg_color="transparent")
        preview_head.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 12))
        preview_head.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            preview_head,
            text="实时识别",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=17, weight="bold"),
            text_color=self.TEXT,
        ).grid(row=0, column=0, sticky="w")

        self.mode_label = ctk.CTkLabel(
            preview_head,
            text=" LIVE CAMERA ",
            fg_color="#18263A",
            corner_radius=10,
            text_color=self.CYAN,
            font=ctk.CTkFont(family="Arial", size=9, weight="bold"),
        )
        self.mode_label.grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            preview_head,
            text="将目标表面放满中央识别区域，避免阴影与高光。",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=10),
            text_color=self.SUBTEXT,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))

        video_shell = ctk.CTkFrame(
            preview_card,
            fg_color="#05070C",
            corner_radius=16,
            border_width=1,
            border_color="#20263A",
        )
        video_shell.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 14))
        video_shell.grid_columnconfigure(0, weight=1)
        video_shell.grid_rowconfigure(0, weight=1)

        self.video_label = tk.Label(
            video_shell,
            bg="#05070C",
            bd=0,
            highlightthickness=0,
        )
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 操作按钮
        control = ctk.CTkFrame(preview_card, fg_color="transparent")
        control.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
        for i in range(5):
            control.grid_columnconfigure(i, weight=1)

        self.scan_btn = self._button(
            control, "◉  识别颜色", self.scan_current, primary=True
        )
        self.scan_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.open_btn = self._button(control, "▣  打开图片", self.open_image)
        self.open_btn.grid(row=0, column=1, sticky="ew", padx=6)

        self.calibrate_btn = self._button(
            control, "◎  白色校准", self.calibrate_white
        )
        self.calibrate_btn.grid(row=0, column=2, sticky="ew", padx=6)

        self.live_btn = self._button(control, "↻  返回实时", self.return_live)
        self.live_btn.grid(row=0, column=3, sticky="ew", padx=6)

        self.voice_btn = self._button(control, "◖  语音播报", self.speak_result)
        self.voice_btn.grid(row=0, column=4, sticky="ew", padx=(6, 0))

        # 右侧结果卡片
        result_card = ctk.CTkFrame(
            body,
            fg_color=self.CARD,
            corner_radius=22,
            border_width=1,
            border_color=self.BORDER,
        )
        result_card.grid(row=0, column=1, sticky="nsew")
        result_card.grid_columnconfigure(0, weight=1)
        result_card.grid_rowconfigure(4, weight=1)

        head = ctk.CTkFrame(result_card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 12))
        head.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            head,
            text="识别结果",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=17, weight="bold"),
            text_color=self.TEXT,
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            head,
            text="OUTPUT FEEDBACK",
            font=ctk.CTkFont(family="Arial", size=9),
            text_color=self.SUBTEXT,
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

        # 主色块
        self.swatch_frame = ctk.CTkFrame(
            result_card,
            fg_color="#191D2A",
            corner_radius=18,
            height=160,
        )
        self.swatch_frame.grid(row=1, column=0, sticky="ew", padx=22)
        self.swatch_frame.grid_propagate(False)
        self.swatch_frame.grid_columnconfigure(0, weight=1)

        self.color_name_label = ctk.CTkLabel(
            self.swatch_frame,
            text="等待识别",
            font=ctk.CTkFont(
                family="Microsoft YaHei UI", size=30, weight="bold"
            ),
            text_color=self.TEXT,
        )
        self.color_name_label.grid(row=0, column=0, sticky="w", padx=20, pady=(22, 0))

        self.rgb_label = ctk.CTkLabel(
            self.swatch_frame,
            text="RGB  — / — / —",
            font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
            text_color=self.SUBTEXT,
        )
        self.rgb_label.grid(row=1, column=0, sticky="w", padx=20, pady=(5, 0))

        self.hex_label = ctk.CTkLabel(
            self.swatch_frame,
            text="HEX  #------",
            font=ctk.CTkFont(family="Arial", size=10),
            text_color=self.SUBTEXT,
        )
        self.hex_label.grid(row=2, column=0, sticky="w", padx=20, pady=(2, 0))

        # 置信度
        confidence_box = ctk.CTkFrame(result_card, fg_color="transparent")
        confidence_box.grid(row=2, column=0, sticky="ew", padx=22, pady=(17, 10))
        confidence_box.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            confidence_box,
            text="识别置信度",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=10),
            text_color=self.SUBTEXT,
        ).grid(row=0, column=0, sticky="w")

        self.confidence_text = ctk.CTkLabel(
            confidence_box,
            text="0%",
            font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
            text_color=self.TEXT,
        )
        self.confidence_text.grid(row=0, column=1, sticky="e")

        self.confidence_bar = ctk.CTkProgressBar(
            confidence_box,
            height=8,
            corner_radius=5,
            progress_color=self.PRIMARY,
            fg_color="#252A3A",
        )
        self.confidence_bar.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0)
        )
        self.confidence_bar.set(0)

        # 四个信息卡片
        metrics = ctk.CTkFrame(result_card, fg_color="transparent")
        metrics.grid(row=3, column=0, sticky="ew", padx=17, pady=(2, 8))
        metrics.grid_columnconfigure((0, 1), weight=1)

        self.value_labels = {}
        metric_data = [
            ("temperature", "色彩倾向", "◐"),
            ("brightness", "亮度", "☀"),
            ("texture", "触感推测", "⌁"),
            ("battery", "设备电量", "▰"),
        ]
        for index, (key, label, icon) in enumerate(metric_data):
            row, col = divmod(index, 2)
            card, value = self._metric_card(metrics, label, icon)
            card.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
            self.value_labels[key] = value

        self.value_labels["battery"].configure(text="75%")

        # 底部状态与提示
        bottom = ctk.CTkFrame(
            result_card,
            fg_color=self.CARD_2,
            corner_radius=16,
            border_width=1,
            border_color="#252B3F",
        )
        bottom.grid(row=4, column=0, sticky="sew", padx=22, pady=(7, 20))
        bottom.grid_columnconfigure(0, weight=1)

        self.feedback = ctk.CTkLabel(
            bottom,
            text="● 系统待机    ○ 语音待机    ○ 震动待机",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=10, weight="bold"),
            text_color=self.CYAN,
        )
        self.feedback.grid(row=0, column=0, sticky="w", padx=16, pady=(13, 5))

        self.calibration_tip = ctk.CTkLabel(
            bottom,
            text="首次使用：用白纸铺满中央框，点击“白色校准”。",
            wraplength=330,
            justify="left",
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=9),
            text_color=self.SUBTEXT,
        )
        self.calibration_tip.grid(
            row=1, column=0, sticky="w", padx=16, pady=(0, 13)
        )

    def _button(self, parent, text, command, primary=False):
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            height=42,
            corner_radius=12,
            border_width=0 if primary else 1,
            border_color=self.BORDER,
            fg_color=self.PRIMARY if primary else self.CARD_2,
            hover_color=self.PRIMARY_HOVER if primary else "#20263A",
            text_color="#FFFFFF" if primary else self.TEXT,
            font=ctk.CTkFont(
                family="Microsoft YaHei UI",
                size=10,
                weight="bold" if primary else "normal",
            ),
        )

    def _metric_card(self, parent, title, icon):
        card = ctk.CTkFrame(
            parent,
            fg_color=self.CARD_2,
            corner_radius=14,
            border_width=1,
            border_color="#252B3F",
        )
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            card,
            text=icon,
            width=24,
            font=ctk.CTkFont(family="Arial", size=18, weight="bold"),
            text_color=self.PRIMARY_HOVER,
        ).grid(row=0, column=0, rowspan=2, padx=(12, 6), pady=12)

        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(family="Microsoft YaHei UI", size=9),
            text_color=self.SUBTEXT,
        ).grid(row=0, column=1, sticky="sw", pady=(10, 0))

        value = ctk.CTkLabel(
            card,
            text="—",
            font=ctk.CTkFont(
                family="Microsoft YaHei UI", size=13, weight="bold"
            ),
            text_color=self.TEXT,
        )
        value.grid(row=1, column=1, sticky="nw", pady=(0, 10))
        return card, value

    def _set_status(self, text: str, color: str, bg: str) -> None:
        self.status_label.configure(text=text, text_color=color)
        self.status_pill.configure(fg_color=bg)

    # -----------------------------
    # 交互与摄像头
    # -----------------------------
    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda _e: self.scan_current())
        self.root.bind("o", lambda _e: self.open_image())
        self.root.bind("O", lambda _e: self.open_image())
        self.root.bind("v", lambda _e: self.speak_result())
        self.root.bind("V", lambda _e: self.speak_result())
        self.root.bind("c", lambda _e: self.calibrate_white())
        self.root.bind("C", lambda _e: self.calibrate_white())
        self.root.bind("q", lambda _e: self.close())
        self.root.bind("<Escape>", lambda _e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _open_camera(self) -> None:
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.simulation_mode = False
            self._set_status("●  摄像头已连接", self.CYAN, "#12242B")
        else:
            self.simulation_mode = True
            self._set_status("●  模拟模式", self.ORANGE, "#2B2118")
            self.mode_label.configure(text=" SIMULATION ", text_color=self.ORANGE)

    def _update_video(self) -> None:
        if self.live_mode:
            frame = None
            if self.cap is not None and self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok:
                    frame = None

            if frame is None:
                frame = self._simulation_frame()

            self.current_frame = frame
            self.frame_buffer.append(frame.copy())
            display = frame.copy()

            h, w = display.shape[:2]
            box_w = int(w * 0.22)
            box_h = int(h * 0.22)
            x1, y1 = w // 2 - box_w // 2, h // 2 - box_h // 2
            x2, y2 = x1 + box_w, y1 + box_h

            # 使用四角扫描标记，弱化传统矩形框
            color = (232, 135, 112)  # BGR，对应紫色
            length = max(24, int(min(box_w, box_h) * 0.14))
            thickness = 4
            segments = [
                ((x1, y1 + length), (x1, y1), (x1 + length, y1)),
                ((x2 - length, y1), (x2, y1), (x2, y1 + length)),
                ((x1, y2 - length), (x1, y2), (x1 + length, y2)),
                ((x2 - length, y2), (x2, y2), (x2, y2 - length)),
            ]
            for p1, p2, p3 in segments:
                cv2.line(display, p1, p2, color, thickness, cv2.LINE_AA)
                cv2.line(display, p2, p3, color, thickness, cv2.LINE_AA)

            cv2.putText(
                display,
                "COLOR SCAN",
                (x1, max(28, y1 - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (242, 242, 248),
                2,
                cv2.LINE_AA,
            )
            self._show_frame(display)

        self.root.after(30, self._update_video)

    def _simulation_frame(self) -> np.ndarray:
        h, w = 720, 1100
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (16, 18, 27)

        t = (self.root.winfo_pointerx() // 8) % 180
        hsv = np.uint8([[[t, 210, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()

        # 更像产品演示页的渐变背景
        for y in range(h):
            alpha = y / max(h - 1, 1)
            frame[y, :] = (
                int(18 + 10 * alpha),
                int(16 + 7 * alpha),
                int(25 + 16 * alpha),
            )

        cv2.rectangle(frame, (290, 170), (810, 550), bgr, -1)
        cv2.putText(
            frame,
            "TOUCH COLOR / SIMULATION",
            (335, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (215, 218, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Move the mouse to change the sample color",
            (320, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (150, 160, 180),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _show_frame(self, frame_bgr: np.ndarray) -> None:
        label_w = max(640, self.video_label.winfo_width())
        label_h = max(420, self.video_label.winfo_height())

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((label_w, label_h), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (label_w, label_h), "#05070C")
        x = (label_w - image.width) // 2
        y = (label_h - image.height) // 2
        canvas.paste(image, (x, y))

        self.current_photo = ImageTk.PhotoImage(canvas)
        self.video_label.configure(image=self.current_photo)

    # -----------------------------
    # 识别、校准与反馈
    # -----------------------------
    def scan_current(self) -> None:
        if self.current_frame is None:
            messagebox.showwarning("提示", "当前没有可识别画面。")
            return

        self.scan_btn.configure(text="◌  正在分析…", state="disabled")
        self.feedback.configure(
            text="● 正在采样    ● 多帧降噪    ○ 等待输出",
            text_color=self.ORANGE,
        )
        self.root.after(120, self._finish_scan)

    def _finish_scan(self) -> None:
        if self.live_mode and len(self.frame_buffer) >= 3:
            frames = list(self.frame_buffer)
            stable_frame = np.median(np.stack(frames), axis=0).astype(np.uint8)
        else:
            stable_frame = self.current_frame

        self.result = analyze_roi(stable_frame, self.white_balance_gains)
        self._render_result(self.result)

        self.scan_btn.configure(
            text="◉  识别颜色",
            state="normal",
        )
        self.feedback.configure(
            text="● 识别完成    ● 多帧稳定    ● 触觉反馈",
            text_color=self.GREEN,
        )
        self.root.after(
            1600,
            lambda: self.feedback.configure(
                text="● 系统待机    ○ 语音待机    ○ 震动待机",
                text_color=self.CYAN,
            ),
        )

    def calibrate_white(self) -> None:
        if self.current_frame is None:
            messagebox.showwarning("提示", "当前没有可校准画面。")
            return

        if self.live_mode and len(self.frame_buffer) >= 3:
            frame = np.median(
                np.stack(list(self.frame_buffer)), axis=0
            ).astype(np.uint8)
        else:
            frame = self.current_frame

        roi = get_center_roi(frame)
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        median = np.median(rgb.reshape(-1, 3), axis=0)

        if float(np.mean(median)) < 80:
            messagebox.showwarning(
                "校准失败",
                "中央区域太暗。请将白纸放满识别框，并保证光线充足。",
            )
            return

        target = 235.0
        gains = np.clip(target / np.maximum(median, 1.0), 0.55, 2.20)
        self.white_balance_gains = tuple(float(v) for v in gains)

        self._set_status("●  白色校准完成", self.GREEN, "#15281E")
        self.feedback.configure(
            text=f"● 校准成功  R {gains[0]:.2f}  G {gains[1]:.2f}  B {gains[2]:.2f}",
            text_color=self.GREEN,
        )
        self.calibration_tip.configure(
            text="白平衡已应用。保持当前灯光位置，再识别目标颜色。",
            text_color=self.GREEN,
        )

    def _render_result(self, result: ColorResult) -> None:
        r, g, b = result.rgb
        hex_color = f"#{r:02X}{g:02X}{b:02X}"
        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        fg = "#111318" if luminance > 155 else "#FFFFFF"
        sub = "#30343E" if luminance > 155 else "#E4E7F0"

        self.swatch_frame.configure(fg_color=hex_color)
        self.color_name_label.configure(text=result.name, text_color=fg)
        self.rgb_label.configure(
            text=f"RGB  {r} / {g} / {b}",
            text_color=sub,
        )
        self.hex_label.configure(text=f"HEX  {hex_color}", text_color=sub)

        self.value_labels["temperature"].configure(text=result.temperature)
        self.value_labels["brightness"].configure(text=f"{result.brightness}%")
        self.value_labels["texture"].configure(text=result.texture)

        self.confidence_text.configure(text=f"{result.confidence}%")
        self.confidence_bar.set(max(0, min(1, result.confidence / 100)))

        if result.confidence >= 80:
            self.confidence_bar.configure(progress_color=self.GREEN)
        elif result.confidence >= 60:
            self.confidence_bar.configure(progress_color=self.PRIMARY)
        else:
            self.confidence_bar.configure(progress_color=self.ORANGE)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择需要识别的图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("读取失败", "无法读取该图片。")
            return

        self.live_mode = False
        self.current_frame = frame
        self.mode_label.configure(text=" IMAGE MODE ", text_color=self.ORANGE)
        self._set_status("●  图片识别模式", self.ORANGE, "#2B2118")

        display = frame.copy()
        h, w = display.shape[:2]
        box_w = int(w * 0.22)
        box_h = int(h * 0.22)
        x1, y1 = w // 2 - box_w // 2, h // 2 - box_h // 2
        x2, y2 = x1 + box_w, y1 + box_h
        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (232, 135, 112),
            max(2, int(min(w, h) / 300)),
        )
        self._show_frame(display)
        self.scan_current()

    def return_live(self) -> None:
        self.live_mode = True
        if self.simulation_mode:
            self.mode_label.configure(text=" SIMULATION ", text_color=self.ORANGE)
            self._set_status("●  模拟模式", self.ORANGE, "#2B2118")
        else:
            self.mode_label.configure(text=" LIVE CAMERA ", text_color=self.CYAN)
            self._set_status("●  摄像头已连接", self.CYAN, "#12242B")

    def speak_result(self) -> None:
        if self.result is None:
            self.scan_current()
            return

        text = (
            f"识别结果为{self.result.name}，"
            f"{self.result.temperature}，亮度百分之{self.result.brightness}，"
            f"{self.result.texture}。"
        )
        self.feedback.configure(
            text="● 正在语音播报    ○ 震动待机",
            text_color=self.ORANGE,
        )

        if self.tts is None:
            messagebox.showinfo("语音播报内容", text)
            self.feedback.configure(
                text="○ 未安装语音模块    ○ 震动待机",
                text_color=self.ORANGE,
            )
            return

        try:
            self.tts.say(text)
            self.tts.runAndWait()
            self.feedback.configure(
                text="● 系统待机    ○ 语音待机    ○ 震动待机",
                text_color=self.CYAN,
            )
        except Exception as exc:
            messagebox.showwarning("语音模块异常", f"语音播报失败：{exc}")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self.tts is not None:
            try:
                self.tts.stop()
            except Exception:
                pass
        self.root.destroy()


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    TouchColorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
