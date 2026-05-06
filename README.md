# VisualEcho 👁️🔊

**Real-time scene narration for the visually impaired.**

VisualEcho uses your webcam and an AI vision model to continuously describe what it sees — out loud. It is designed to help visually impaired users understand their surroundings without needing to read a screen.

---

## Demo

> *"a person sitting at a desk with a laptop and a cup of coffee"*
> *"a kitchen with white cabinets and a window on the left"*
> *"two people standing outside near a street with cars parked behind them"*

<!-- Add a demo GIF here: assets/demo.gif -->
<!-- ![Demo](assets/demo.gif) -->

---

## Features

- 🎥 **Live webcam feed** with real-time caption overlay
- 🧠 **AI-powered scene understanding** using BLIP (no API key, runs locally)
- 🔊 **Text-to-speech narration** — works offline via `pyttsx3`, or online via `gTTS`
- ⏱️ **Adjustable interval** — control how often it describes the scene
- ⌨️ **Manual trigger** — press SPACE to describe on demand
- 💻 **CPU-compatible** — no GPU required

---

## How It Works

```
Webcam frame
     │
     ▼
BLIP Vision Model  ──►  "a person sitting at a desk with a laptop"
     │
     ▼
Text-to-Speech  ──►  🔊 spoken aloud
     │
     ▼
Caption overlay on video feed
```

The captioning and TTS each run in their own background thread, so the video feed stays smooth at all times.

---

## Installation

**Requirements:** Python 3.9+

```bash
git clone https://github.com/YOUR_USERNAME/VisualEcho.git
cd VisualEcho
pip install -r requirements.txt
```

> On first run, the BLIP model (~900MB) will be downloaded automatically from HuggingFace.

### System TTS (for pyttsx3)

| OS | Required |
|----|----------|
| Windows | Built-in SAPI5 — nothing to install |
| macOS | Built-in `nsss` — nothing to install |
| Linux | `sudo apt install espeak` |

---

## Usage

```bash
python visual_echo.py
```

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Describe the scene immediately |
| `+` / `=` | Increase interval between descriptions |
| `-` | Decrease interval between descriptions |
| `Q` | Quit |

---

## Configuration

You can edit these values at the top of `visual_echo.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interval` | `5.0` | Seconds between auto-descriptions |
| `model_id` | `Salesforce/blip-image-captioning-base` | HuggingFace model to use |

To use the larger, more accurate BLIP-2 model (requires GPU):
```python
# In visual_echo.py, replace the model loading section with:
from transformers import Blip2Processor, Blip2ForConditionalGeneration
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Webcam capture | OpenCV |
| Scene captioning | BLIP via HuggingFace Transformers |
| Text-to-speech | pyttsx3 (offline) / gTTS (online fallback) |
| Image processing | Pillow |
| Deep learning | PyTorch |

---

## Limitations

- Caption quality depends on lighting and camera angle
- On CPU, inference takes 1–3 seconds per frame (hence the interval-based approach)
- BLIP describes scenes in English only

---

## Future Work

- Depth estimation to add spatial context ("chair to your left")
- Language selection for non-English speakers
- Object tracking to only re-describe when the scene changes significantly
- Mobile version using phone camera

---

## License

MIT License — free to use, modify, and distribute.

---

## References

- [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086) — Salesforce Research
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [OpenCV](https://opencv.org/)
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3)
