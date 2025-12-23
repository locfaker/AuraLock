# 🔐 AuraLock

> **Bức màn bảo vệ vô hình cho tác phẩm nghệ thuật của bạn**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author](https://img.shields.io/badge/Author-locfaker-purple.svg)](https://github.com/locfaker)

## 🎯 Giới thiệu

**AuraLock** là một công cụ bảo vệ hình ảnh nghệ thuật khỏi việc bị AI học và sao chép phong cách. Sử dụng kỹ thuật **Adversarial Perturbation**, công cụ tạo ra một "bức màn bảo vệ vô hình":

- ✅ **Mắt người nhìn**: Hình ảnh bình thường, chất lượng cao
- ❌ **AI nhìn**: Nhiễu loạn, không thể học được style

## ✨ Tính năng

- 🛡️ **FGSM Attack**: Bảo vệ nhanh, hiệu quả
- 🔒 **PGD Attack**: Bảo vệ mạnh hơn, khó bypass
- 📊 **Quality Metrics**: Đo lường PSNR, SSIM
- 💻 **CLI Tool**: Dễ dàng sử dụng từ command line
- 🌐 **Web UI**: Giao diện đẹp với Gradio

## 🚀 Cài đặt nhanh

```bash
# Clone repo
git clone https://github.com/locfaker/AuraLock.git
cd AuraLock

# Tạo môi trường ảo
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -e ".[dev]"
```

## 💻 Cách sử dụng

### 1. Command Line (CLI)

```bash
# Bảo vệ một ảnh
auralock protect artwork.png -o protected.png -e 0.03

# So sánh hai ảnh
auralock analyze original.png protected.png

# Chạy demo
auralock demo
```

### 2. Web UI (Gradio)

```bash
python -m auralock.ui.gradio_app
# Mở browser: http://127.0.0.1:7860
```

### 3. Python API

```python
from auralock.attacks import FGSM, PGD
from auralock.core.image import load_image, save_image
from torchvision.models import resnet18, ResNet18_Weights

# Load ảnh và model
image = load_image("artwork.png")
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Áp dụng bảo vệ FGSM
attack = FGSM(model, epsilon=0.03)
protected = attack(image.unsqueeze(0))

# Lưu kết quả
save_image(protected, "protected.png")
```

## 📦 Cấu trúc Project

```
AuraLock/
├── src/
│   ├── auralock/
│   │   ├── core/           # Image utilities, metrics
│   │   ├── attacks/        # FGSM, PGD attacks
│   │   ├── ui/             # Gradio Web UI
│   │   └── cli.py          # Command line interface
│   └── tests/              # Unit tests (23 tests)
├── examples/               # Demo scripts
├── notebooks/              # Jupyter tutorials
└── docs/                   # Documentation
```

## 🔬 Các phương pháp bảo vệ

| Phương pháp | Mô tả | Tốc độ | Hiệu quả |
|-------------|-------|--------|----------|
| **FGSM** | Fast Gradient Sign Method | ⚡ Nhanh | ⭐⭐⭐ |
| **PGD** | Projected Gradient Descent | 🐢 Chậm hơn | ⭐⭐⭐⭐⭐ |

## 📊 Benchmark

| Epsilon | Attack Success | PSNR (dB) | SSIM | Chất lượng |
|---------|----------------|-----------|------|------------|
| 0.01 | 100% | 40.0 | 0.9994 | Excellent |
| 0.03 | 100% | 30.5 | 0.9948 | Acceptable |
| 0.05 | 100% | 26.2 | 0.9858 | Poor |

## 🛠️ Tech Stack

- **Python 3.10+**
- **PyTorch** - Deep Learning framework
- **Gradio** - Web UI
- **Typer + Rich** - CLI
- **scikit-image** - Image metrics

## 📖 Tài liệu

- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) - Kế hoạch triển khai
- [Research Roadmap](docs/RESEARCH_ROADMAP.md) - Lộ trình nghiên cứu

## 🧪 Chạy tests

```bash
pytest src/tests/ -v
```

## 📝 Giấy phép

Dự án được phân phối dưới giấy phép **MIT License** - xem file [LICENSE](LICENSE).

## 👤 Tác giả

**locfaker**

- GitHub: [@locfaker](https://github.com/locfaker)

---

⭐ Nếu thấy hữu ích, hãy cho mình một star nhé!
