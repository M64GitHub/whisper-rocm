# whisper-rocm

Local speech-to-text with AMD ROCm GPU acceleration - an open-source alternative to Wispr Flow

A browser-based voice transcription app powered by OpenAI's Whisper model, optimized for AMD GPUs using ROCm. Built specifically for the new AMD Ryzen AI processors with Radeon 800M series integrated graphics.

<img width="3096" height="1345" alt="image" src="https://github.com/user-attachments/assets/72b5e4c5-9c86-49c6-b4c2-9624eef4bf4c" />

## About
I built this app to have an alternative to **Wispr Flow** (Mac-only) on my new linux laptop with an AMD Ryzen AI 9 HX 370 w/ Radeon 890M.  
Since **AMD GPUs now have first-class support**: ROCm finally supports Strix Point (gfx1150), I can run it **fully local**, no cloud APIs, no subscriptions, voice data stays on the machine. And it is pretty **fast** -  GPU-accelerated transcription with the medium Whisper model.

## Features

- Real-time audio visualization with cyberpunk-themed UI
- Press-and-hold recording (mouse or touch)
- GPU-accelerated transcription via ROCm/CUDA
- One-click copy to clipboard
- Mobile-responsive design
- Automatic language detection

## Hardware

**Tested on:**
- AMD Ryzen AI 9 HX 370 + Radeon 890M (Strix Point / gfx1150)
- TUXEDO laptop running Linux

**Compatible with:**
- AMD GPUs with ROCm support (gfx1150, gfx1100, etc.)
- NVIDIA GPUs (CUDA)
- CPU fallback (slower, but works)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/M64GitHub/whisper-rocm.git
cd whisper-rocm
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch with ROCm support

**For AMD Radeon 890M / 880M (gfx1150 - Strix Point):**

```bash
pip install --index-url https://repo.amd.com/rocm/whl/gfx1150/ torch
```

**For other AMD GPUs**, check available builds at: https://repo.amd.com/rocm/whl/

**For NVIDIA GPUs or CPU**, see: https://pytorch.org/get-started/locally/

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

```bash
python App.py
```

Open http://localhost:8000 in your browser.

**How to use:**
1. Click and hold the "HOLD TO RECORD" button
2. Speak clearly into your microphone
3. Release the button to transcribe
4. Click "COPY" or use the auto-selected text

## Configuration

### Whisper Model Size

Edit `App.py` line 20 to change the model:

```python
model = whisper.load_model("medium", device=device)  # Options: tiny, base, small, medium, large
```

| Model  | Parameters | VRAM  | Speed   | Accuracy |
|--------|------------|-------|---------|----------|
| tiny   | 39M        | ~1GB  | Fastest | Basic    |
| base   | 74M        | ~1GB  | Fast    | Good     |
| small  | 244M       | ~2GB  | Medium  | Better   |
| medium | 769M       | ~5GB  | Slower  | Great    |
| large  | 1550M      | ~10GB | Slowest | Best     |

## Tech Stack

- **Backend:** FastAPI + Uvicorn
- **ML Model:** OpenAI Whisper
- **GPU Acceleration:** PyTorch + ROCm 7.10
- **Frontend:** Vanilla HTML/CSS/JavaScript
- **Audio:** Web Audio API + MediaRecorder

## Troubleshooting

### Check if ROCm detects your GPU

```bash
source venv/bin/activate
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### Microphone permissions

Make sure your browser has microphone access.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — the speech recognition model
- [AMD ROCm](https://www.amd.com/en/products/software/rocm.html) — GPU compute platform
- [FastAPI](https://fastapi.tiangolo.com/) — modern Python web framework
