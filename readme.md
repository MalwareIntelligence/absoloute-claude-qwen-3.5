# lowvram_gui_gguf – Low-VRAM GUI for GGUF Models (llama.cpp)

A Tkinter GUI for running GGUF-quantized language models locally via
[`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) — works even on cards with
limited VRAM, since only part of the layers (`n_gpu_layers`) are loaded onto the GPU, the rest
stays in regular RAM.

Includes:
- File attachments as context (text/code files are prepended to the prompt)
- Model-initiated tool calls (read/write file, list directory, optionally run Python, web search)
- `<think>...</think>` splitting for reasoning models (Qwen3, QwQ, DeepSeek-R1-Distills)
- Effort presets (Low/Medium/High), analogous to a "thinking effort" parameter
- Persistent project memory (`context.md`), appended to the latest message instead of the system
  message so the prefix cache stays valid

**Model used:** [`Malwareintelligence/Absoloute-claude-qwen3.5-9b`](https://huggingface.co/Malwareintelligence/Absoloute-claude-qwen3.5-9b)
(a Qwen3.5-9B finetune, as a GGUF quantization). The repo currently only contains `LICENSE`/`README.md`
placeholders — place the actual `.gguf` file there, or download it from the repo's Files tab once
uploaded, and set the filename below accordingly.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Check NVIDIA driver](#2-check-nvidia-driver)
3. [Install CUDA Toolkit](#3-install-cuda-toolkit)
4. [Install cuDNN](#4-install-cudnn)
5. [Set up Python environment](#5-set-up-python-environment)
6. [Install PyTorch with CUDA](#6-install-pytorch-with-cuda)
7. [Install llama-cpp-python with CUDA support](#7-install-llama-cpp-python-with-cuda-support)
8. [Download the model](#8-download-the-model)
9. [Run the script](#9-run-the-script)
10. [Using the GUI](#10-using-the-gui)
11. [Optional features](#11-optional-features)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Requirements

| Component | Recommendation |
|---|---|
| OS | Windows 10/11 (this guide), Linux works analogously |
| GPU | NVIDIA GPU with CUDA support (Compute Capability 6.0+) |
| VRAM | usable from 4–6 GB (thanks to `n_gpu_layers` offloading); more VRAM = more layers on GPU = faster |
| RAM | 16 GB minimum, 32 GB+ for larger models |
| Python | 3.10–3.12 (3.11 recommended) |
| Disk space | 5–10 GB for a 9B model depending on quantization |

---

## 2. Check NVIDIA driver

Install/check the current driver: https://www.nvidia.com/Download/index.aspx

Then in a console (PowerShell/CMD):

```powershell
nvidia-smi
```

The top-right of the output shows the highest CUDA version supported by your driver — the CUDA
Toolkit in step 3 should be equal to or lower than that.

---

## 3. Install CUDA Toolkit

1. Download the CUDA Toolkit (e.g. 12.4, compatible with most current `llama-cpp-python` CUDA wheels):
   https://developer.nvidia.com/cuda-downloads
2. Select OS/architecture/version (Windows → x86_64 → 10/11 → exe (local)).
3. Run the installer, choose **"Express (Recommended)"**.
4. Verify afterwards:

```powershell
nvcc --version
```

If you get "nvcc not found", the PATH entry is missing:

```
%CUDA_PATH%\bin
```

(`CUDA_PATH` is normally set automatically by the installer, e.g.
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`) — restart your system if the variable
isn't picked up yet.

---

## 4. Install cuDNN

cuDNN isn't strictly required for `llama-cpp-python` itself, but is useful for PyTorch CUDA
acceleration and other ML workloads:

1. Create/log in to an NVIDIA Developer account: https://developer.nvidia.com/cudnn
2. Download the cuDNN version matching your installed CUDA version (e.g. cuDNN 8.9/9.x for CUDA 12.x).
3. Extract the downloaded archive. It contains the `bin`, `include`, and `lib` folders.
4. Copy the contents into the CUDA Toolkit directory (merge folders, don't overwrite):

```
Source (from the cuDNN archive)        Destination
bin\*.dll        →  %CUDA_PATH%\bin
include\*.h      →  %CUDA_PATH%\include
lib\x64\*.lib    →  %CUDA_PATH%\lib\x64
```

5. Confirm `%CUDA_PATH%\bin` is in your PATH (see step 3).

---

## 5. Set up Python environment

Use a dedicated virtual environment so package versions don't clash with other projects:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

---

## 6. Install PyTorch with CUDA

Not strictly required for this script itself (it uses `llama-cpp-python`, not PyTorch), but standard
practice if you also want to run other models/scripts (e.g. `transformers`) with CUDA in the same
environment:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

(Adjust `cu124` to your installed CUDA Toolkit version, e.g. `cu121`, `cu128` — the current matching
combination is always listed at https://pytorch.org/get-started/locally/.)

Test whether PyTorch sees the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `True <your GPU name>`

---

## 7. Install llama-cpp-python with CUDA support

This is the step that actually matters for `lowvram_gui_gguf.py` — the plain PyPI version of
`llama-cpp-python` is usually a CPU-only build. GPU offloading (`n_gpu_layers`) needs a
CUDA (cuBLAS) build.

**Option A – prebuilt CUDA wheel (simplest on Windows):**

```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

(Adjust `cu124` again to match your CUDA version; available variants are listed in the
[llama-cpp-python repo](https://github.com/abetlen/llama-cpp-python#installation-with-hardware-acceleration).)

**Option B – build from source with the CUDA flag** (if no matching wheel exists; also requires the
"Desktop development with C++" workload from Visual Studio 2022 Build Tools):

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

Then install the script's remaining dependencies:

```powershell
pip install numpy
```

`selenium` is only needed if you want the optional web search feature to run through a real browser
instead of OpenSERP/the DuckDuckGo HTML fallback:

```powershell
pip install selenium
```

---

## 8. Download the model

1. Open the repo: https://huggingface.co/Malwareintelligence/Absoloute-claude-qwen3.5-9b
2. Under "Files and versions", pick the appropriate `.gguf` file (quantization level depends on your
   VRAM/RAM, e.g. Q4_K_M is a solid size/quality tradeoff) and download it, e.g. via `curl`:

```powershell
curl -L -o model.gguf https://huggingface.co/Malwareintelligence/Absoloute-claude-qwen3.5-9b/resolve/main/<FILENAME>.gguf
```

Replace `<FILENAME>` with the actual file name shown in the repo's Files view.

3. Put the `.gguf` file in a folder you'll remember (e.g. `models\`) — you'll pick this path from the
   GUI's file picker at startup.

---

## 9. Run the script

```powershell
venv\Scripts\activate
python lowvram_gui_gguf.py
```

---

## 10. Using the GUI

- **Model path:** use the file picker to select the downloaded `.gguf` file.
- **n_gpu_layers:** controls how many layers get loaded onto the GPU.
  - `0` = everything runs on CPU/RAM (slow, but works without a GPU)
  - `999` (or another very high value) = as many layers as possible on the GPU
  - With limited VRAM, increase this gradually until VRAM gets tight (watch it via `nvidia-smi`).
- **Effort (Low/Medium/High):** controls how thoroughly/long the model "thinks" before answering.
- **Attach files:** text/code files get prepended to the prompt so the model "sees" the content directly.
- **Tools:** depending on the chat template (Qwen/Hermes-style), the model can actively read/write
  files, list directories, and optionally run Python code (the latter disabled by default, since it
  executes arbitrary code locally — only enable it if you trust the model/input).
- **context.md:** compressed project memory across sessions, updated automatically or manually as
  needed, and kept out of the actual chat history (so the prefix cache stays stable).

---

## 11. Optional features

**Web search (OpenSERP):** for cleaner search results instead of HTML scraping, you can optionally
run a self-hosted [OpenSERP](https://github.com/karust/openserp) server:

```powershell
docker run -p 7000:7000 karust/openserp serve
```

The GUI expects the server at `http://127.0.0.1:7000` by default; if none is reachable, the script
automatically falls back to DuckDuckGo HTML scraping.

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `nvcc` not found | `%CUDA_PATH%\bin` missing from PATH, restart the system |
| `llama_cpp` only uses CPU despite CUDA wheel | Wrong `cuXXX` wheel installed for your CUDA version; `pip uninstall llama-cpp-python` and reinstall with the matching index |
| `torch.cuda.is_available()` → `False` | Driver/toolkit/PyTorch wheel CUDA versions don't match — check the driver via `nvidia-smi`, adjust `cuXXX` in the pip command if needed |
| Out-of-memory while loading | Reduce `n_gpu_layers`, or use a smaller quantization level (e.g. Q4 instead of Q8) |
| Build errors with Option B (source build) | Visual Studio Build Tools ("Desktop development with C++") missing or wrong version |

---

## License note

The linked model is under the license stated in its repo (`nocopy`) — check the terms there before
use/redistribution.
