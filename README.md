# ⚡ MOTORE — llama-server Vulkan GUI Launcher

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Qt-PyQt6-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="PyQt6">
  <img src="https://img.shields.io/badge/Vulkan-Supported-E6232E?style=for-the-badge&logo=vulkan&logoColor=white" alt="Vulkan">
  <img src="https://img.shields.io/badge/llama.cpp-Engine-000000?style=for-the-badge" alt="llama.cpp">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License MIT">
</p>

<p align="center">
  <b>MOTORE</b> is a modern, lightweight, and optimized <b>PyQt6</b> graphical launcher designed to manage and execute <code>llama-server</code> with <b>Vulkan</b> GPU acceleration, advanced <b>Speculative Decoding</b> support (Gemma 4 MTP, external draft models, and N-gram), and real-time performance metrics.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Speculative Decoding Support](#-speculative-decoding-support)
- [Configuration Parameters](#-configuration-parameters)
- [License](#-license)

---

## 🔍 Overview

**MOTORE** streamlines local LLM inference orchestration. It automates the discovery of Vulkan-enabled `llama-server` binaries, scans local GGUF model directories (with seamless LM Studio integration), and provides precise control over context limits, quantized KV Caching, and speculative decoding techniques.

---

## ✨ Key Features

- 🔍 **Automatic Discovery**:
  - Automatically locates `llama-server` binaries (LM Studio backends, extension builds, environment variables, or system `$PATH`).
  - Scans `~/Apps/LM Studio/models` for GGUF model files.
  - Intelligently pairs matching Draft/Assistant models with selected primary GGUF models.

- 🚀 **Advanced Speculative Decoding**:
  - **Internal MTP (Gemma 4)**: Uses embedded Multi-Token Prediction heads (`--spec-type draft-mtp`).
  - **External MTP (Gemma 4)**: Pair with external GGUF Gemma 4 Assistant/MTP files.
  - **External Draft**: Standard draft model support (`--spec-type draft-simple`).
  - **N-gram**: Pattern-based speculative decoding requiring no secondary model (`--spec-type ngram-mod`).

- 📊 **Real-time Metrics Dashboard**:
  - Monitors output timings directly from `llama-server` logs.
  - Displays generation speed (`tok/s`), total token throughput, and **Draft Acceptance Rate (%)**.

- 🎨 **Modern Themeable Interface**:
  - Toggle between **Dark** and **Light** themes.
  - Vector geometric brand identity rendered via Qt graphics.
  - Real-time CLI command preview prior to server execution.

---

## 🛠️ Prerequisites

- **Operating System**: Linux (Fedora / KDE Plasma Wayland recommended), Windows, or macOS.
- **Graphics Driver**: Vulkan API support configured and operational.
- **Python**: Version 3.10 or higher.
- **llama-server**: Vulkan-compiled binary (provided via LM Studio backend or custom build).

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/motore.git](https://github.com/your-username/motore.git)
   cd motore
