# 🧠 AI Model for Rhinoplasty

---

## 📌 Overview

This project documents the investigation, verification, and integration of an undocumented, pre-trained Pix2Pix (conditional GAN) model used to generate AI-predicted post-operative rhinoplasty outcome previews from patient side-profile photographs.

The trained checkpoint was sourced from a prior academic research project (CS31-2, University of Sydney capstone) with **no accompanying documentation, configuration file, or architecture code** — only the raw trained weights. This repository documents the reverse-engineering process used to identify the correct model architecture with zero available documentation, and the resulting CPU-optimized inference pipeline built around it.

---

## 🚀 Key Features

**🔍 Architecture Reverse-Engineering**
Identified the correct model architecture from raw checkpoint tensor data alone — no config file, README, or code existed for the original checkpoint.

**✅ Verified Correctness**
Confirmed the identified architecture via `load_state_dict(strict=True)` — a perfect match with zero missing or unexpected parameters.

**⚡ CPU-Only Inference**
No GPU dependency anywhere in the pipeline — optimized for zero-cost, CPU-only deployment. Achieves ~1.7 second inference time and <900MB peak memory.

**📦 Offline-Capable**
Model loads entirely from local cache after first download — no runtime network dependency.

**🧪 Verified Preprocessing Pipeline**
Reproduces the exact input format the model expects (4-channel tensor, fixed 512×1536 canvas, precise normalization) — confirmed through direct inspection of the original model's serving code.

---

## 🧠 System Architecture

The pipeline consists of the following components:

- **checkpoint_inspector.py**
  Inspects the raw checkpoint's internal structure (layer names, tensor shapes) without loading it into any model class — used during the architecture investigation phase.

- **architecture_verification.py**
  Loads the checkpoint into the identified candidate architecture using strict state-dict matching, confirming correctness.

- **preprocessing.py**
  Builds the exact 4-channel, 512×1536 input tensor the model expects — resize, crop/pad, channel construction, normalization.

- **inference.py**
  Runs the verified model on CPU, wrapped for integration into a FastAPI backend service.

- **postprocessing.py**
  Converts the model's raw output tensor back into a viewable image.

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.x
- PyTorch (CPU build)
- Required Python packages:
  - torch
  - opencv-python
  - Pillow
  - numpy

---

## 📥 Installation

Clone the repository
```
git clone https://github.com/priyanshkhandelwall/ai-model-for-rhinoplasty.git
cd ai-model-for-rhinoplasty
```

Install dependencies
```
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the architecture verification script (confirms the checkpoint loads correctly)
```
python architecture_verification.py
```

Run inference on a test image
```
python inference.py --input path/to/image.jpg --output path/to/output.jpg
```

---

## 👥 Credit

**Priyansh Khandelwal** — Model architecture investigation, verification pipeline, CPU inference optimization, integration engineering

**Original model & training** — CS31-2 Capstone Project, University of Sydney (Jingyao Zhang), supervised by Dr. Ali. Used with permission for academic and portfolio purposes.

---

## 📄 License

This project was developed as part of an academic capstone at the University of Sydney. The underlying Pix2Pix/CycleGAN architecture is derived from the original open-source pix2pix research codebase (BSD License). Used and shared here with permission from the project supervisor.
