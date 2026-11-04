# COLOR TOUCH

COLOR TOUCH 是一套软硬件结合的智能色彩识别系统，面向服装、彩妆、商品选购等日常场景，帮助用户快速获取物体颜色信息，并提供直观的可视化分析结果。

硬件端采用 ESP32-S3 作为主控，连接 AS7341 光谱传感器与 OLED 显示屏，实现颜色数据采集、处理与本地显示。

软件端使用 Python 构建 COLOR TOUCH 桌面交互界面，采用灰色与紫色的空间玻璃拟态视觉风格，整合了实时摄像头采集、图片识别、白色校准、Lab 颜色匹配、多帧降噪、颜色历史记录和语音反馈等功能。

系统整体按照“采集 → 分析 → 结果 → 反馈”的逻辑运行，将硬件传感数据、算法处理结果与用户交互界面连接起来，形成完整的色彩识别体验。

## 项目亮点

- ESP32-S3 + AS7341 光谱传感器硬件方案
- 支持 RGB、HEX、Lab 等多种颜色数据表达
- 摄像头实时采集与图片颜色识别
- 白色校准与多帧降噪处理
- 颜色历史记录与识别结果管理
- 语音反馈与交互状态提示
- 灰紫色玻璃拟态 UI 设计
- 模块化的软件系统架构

## 项目结构

```text
color_touch_app.py      # COLOR TOUCH 主界面与交互入口
requirements.txt        # Python 依赖
archive/                # 原始版本与历史压缩包，仅作备份
```

## 安装依赖

```powershell
python -m pip install -r requirements.txt
```

或直接安装：

```powershell
python -m pip install opencv-python numpy Pillow pyttsx3 customtkinter
```

## 运行

```powershell
python color_touch_app.py
```

原版代码与历史版本压缩包已移入 `archive/`，新界面入口为 `color_touch_app.py`。
