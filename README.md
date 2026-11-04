# COLOR TOUCH

灰色与紫色为主的空间玻璃拟态界面，保留摄像头采集、图片识别、白色校准、Lab 颜色匹配、多帧降噪和语音反馈功能。

界面按功能划分为左侧操作栏、实时采集、识别结果、颜色详情、历史记录、光谱签名和语音/触觉反馈七个模块。代码按“采集 → 分析 → 结果 → 反馈”的产品逻辑组织，便于在作品集中展示软件系统设计。

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
