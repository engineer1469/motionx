# 🌀 motionx

**motionx** is a lightweight Python library + CLI tool for **motion extraction in videos or image sequences**.  
It builds on **OpenCV** and provides a clean, modern API for frame differencing, background subtraction, and a colorized weighted motion highlighting method.

---

## ✨ Features

- 📽️ **Frame differencing** (consecutive grayscale subtraction + thresholding)
- 🧩 **Background subtraction (MOG2)** with shadow detection
- 🎨 **Weighted motion (color)**: blend current frame with inverted delayed frame
- 🖼️ Returns **binary motion masks** (diff/mog2) or **color motion** (weighted)
- 📂 Supports **video files, webcams, and image sequences**
- 🛠️ Easy to integrate into **Python pipelines**
- 💻 Comes with a **CLI tool** for quick conversion
- 🧪 Tested & typed (Python ≥3.9, type hints included)

---

## 📦 Installation

### From source
```bash
git clone https://github.com/engineer1469/motionx.git
cd motionx
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Dependencies
- Python 3.9+
- [OpenCV](https://pypi.org/project/opencv-python/)
- NumPy

---

## 🚀 Quickstart

### Python API

```python
import cv2
from motionx import MotionExtractor

# Initialize with "diff", "mog2", or "weighted"
me = MotionExtractor(method="diff")

cap = cv2.VideoCapture("input.mp4")
while True:
    ok, frame = cap.read()
    if not ok:
        break

    motion = me.process(frame)  # mask (diff/mog2) or color (weighted)
    win = "Motion (mask/color)"
    cv2.imshow(win, motion)

    if cv2.waitKey(30) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
```

---

### CLI

#### Video → motion mask video
```bash
motionx -i input.mp4 -o motion_mask.mp4 --method diff --thresh 30 --blur 5
```

#### Webcam → image sequence
```bash
motionx -i 0 --out-dir masks --method mog2
```

#### Image directory → video
```bash
motionx -i frames/ --in-fps 24 -o motion_mask.mp4 --method diff
```

#### Glob → image sequence
```bash
motionx -i "frames/*.png" --in-fps 30 --out-seq "out/mask_%06d.png"
```

#### Weighted motion (color output)
The weighted method produces a color visualization by blending the current frame with the inverted previous (or delayed) frame:

- `alpha`: weight for the current frame
- `beta`: weight for the inverted previous frame
- `offset`: how many frames back to reference (1 = immediate previous)

Example:
```bash
motionx -i input.mp4 -o weighted.mp4 --method weighted --alpha 0.6 --beta 0.6 --offset 2
```

Notes:
- Weighted mode writes a color video (`--o weighted.mp4`) automatically.
- If writing an image sequence, color PNGs/JPEGs are produced.

---

## ⚙️ Configuration

### Frame Differencing (`method="diff"`)
| Parameter     | Default | Description |
|---------------|---------|-------------|
| `thresh`      | 25      | Binary threshold for pixel difference |
| `blur_kernel` | (5, 5)  | Gaussian blur kernel (helps remove noise) |
| `morph_kernel`| (3, 3)  | Morphological kernel size |
| `morph_iters` | 1       | Dilation iterations |

### Background Subtraction (`method="mog2"`)
### Weighted Motion (`method="weighted"`)
| Parameter  | Default | Description |
|------------|---------|-------------|
| `alpha`    | 0.5     | Blend weight for current frame |
| `beta`     | 0.5     | Blend weight for inverted delayed frame |
| `offset`   | 1       | Number of frames to delay (>=1) |

The output is a full-color image where static regions tend toward neutral gray (~127), and motion regions become more vivid due to blending with the inverted previous frame.
| Parameter        | Default | Description |
|------------------|---------|-------------|
| `history`        | 500     | Frames to build background model |
| `var_threshold`  | 16.0    | Variance threshold for foreground detection |
| `detect_shadows` | True    | Whether to mark shadows (~127 gray) |

---

## 📊 Example Outputs

### Frame differencing
```
Original Frame → Motion Mask
```
![Diff Example]()TODO

### Background subtraction
```
Original Frame → MOG2 Mask
```
![MOG2 Example]()TODO

---

## 🧑‍💻 Development

```bash
git clone https://github.com/engineer1469/motionx.git
cd motionx
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"

# Run tests
python -m pip install pytest
pytest -q

# Code formatting + linting
pre-commit install
pre-commit run --all-files
```

---

## 📌 Roadmap

- [ ] Add **Farnebäck optical flow** (per-pixel motion vectors)
- [ ] Motion **bounding box extraction**
- [ ] Support for **region-of-interest (ROI) masking**
- [ ] Temporal **motion smoothing**
- [ ] Real-time camera capture mode (`motionx --camera 0`)
- [ ] Publish to PyPI

---

## 📜 License

MIT © 2025 Sepp Beld

---

## 🙌 Acknowledgements

- [OpenCV](https://opencv.org/) for the core computer vision tools  
- Inspiration from countless "motion detection" snippets floating around the CV community  
- Inspiration from the very interesting videos from [**Consistently Inconsistent**](https://www.youtube.com/watch?v=zFiubdrJqqI)
