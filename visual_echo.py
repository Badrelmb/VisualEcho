"""
VisualEcho — Real-time scene narration for the visually impaired.
Captures webcam frames, generates natural language descriptions,
and speaks them aloud using text-to-speech.
"""

import cv2
import time
import threading
import queue
import sys
import numpy as np
from PIL import Image

# ── TTS setup ─────────────────────────────────────────────────────────────────

def init_tts():
    """Try pyttsx3 (offline) first, fall back to gTTS (online)."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1.0)
        print("[TTS] Using pyttsx3 (offline)")
        return 'pyttsx3', engine
    except Exception:
        print("[TTS] pyttsx3 unavailable, falling back to gTTS (requires internet)")
        return 'gtts', None


def speak_pyttsx3(engine, text):
    engine.say(text)
    engine.runAndWait()


def speak_gtts(text):
    try:
        from gtts import gTTS
        import pygame
        import io
        tts = gTTS(text=text, lang='en', slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        pygame.mixer.init()
        pygame.mixer.music.load(mp3_fp, 'mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"[TTS] gTTS error: {e}")


# ── Captioning setup ───────────────────────────────────────────────────────────

def load_captioning_model():
    """
    Load BLIP (base) for CPU-friendly inference.
    BLIP-2 is more powerful but requires more VRAM.
    """
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    print("[Model] Loading BLIP captioning model (first run downloads ~900MB)...")
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlipForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    print(f"[Model] BLIP loaded on {device.upper()}")
    return processor, model, device


def generate_caption(processor, model, device, frame_rgb):
    """Generate a caption for a single RGB frame."""
    import torch
    pil_image = Image.fromarray(frame_rgb)
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=4,
            repetition_penalty=1.3
        )
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


# ── Overlay helpers ────────────────────────────────────────────────────────────

def wrap_text(text, max_chars=55):
    """Wrap text into lines of max_chars for the OpenCV overlay."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current += (" " if current else "") + word
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_overlay(frame, caption, status, caption_age, interval):
    """Draw a semi-transparent caption bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Background bar
    bar_h = 110
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Caption text
    lines = wrap_text(caption if caption else "Initializing...")
    y_start = h - bar_h + 22
    for i, line in enumerate(lines[:3]):
        cv2.putText(frame, line, (12, y_start + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    # Progress bar showing time until next caption
    elapsed = min(caption_age, interval)
    progress = int((elapsed / interval) * (w - 24))
    cv2.rectangle(frame, (12, h - 14), (12 + progress, h - 8), (80, 200, 120), -1)
    cv2.rectangle(frame, (12, h - 14), (w - 12, h - 8), (80, 80, 80), 1)

    # Status indicator top-right
    color = (80, 200, 120) if status == "ready" else (80, 150, 255)
    label = "LIVE" if status == "ready" else "THINKING..."
    cv2.putText(frame, label, (w - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Controls hint top-left
    cv2.putText(frame, "Q: quit  |  SPACE: describe now  |  +/-: interval",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  VisualEcho — Real-time Scene Narration")
    print("=" * 50)

    # Load models
    processor, model, device = load_captioning_model()
    tts_engine_type, tts_engine = init_tts()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State
    caption = ""
    status = "thinking"
    last_caption_time = 0
    caption_start_time = time.time()
    interval = 5.0          # seconds between auto-captions
    force_caption = False   # triggered by SPACE key

    # TTS runs in a background thread so it doesn't block the video feed
    tts_queue = queue.Queue()

    def tts_worker():
        while True:
            text = tts_queue.get()
            if text is None:
                break
            if tts_engine_type == 'pyttsx3':
                speak_pyttsx3(tts_engine, text)
            else:
                speak_gtts(text)
            tts_queue.task_done()

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    # Captioning runs in a background thread so video stays smooth
    caption_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()

    def caption_worker():
        while True:
            frame_rgb = caption_queue.get()
            if frame_rgb is None:
                break
            nonlocal status
            status = "thinking"
            cap_text = generate_caption(processor, model, device, frame_rgb)
            result_queue.put(cap_text)
            caption_queue.task_done()

    caption_thread = threading.Thread(target=caption_worker, daemon=True)
    caption_thread.start()

    print("\n[Ready] Webcam open. Press SPACE to describe now, Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Lost webcam feed.")
            break

        now = time.time()
        caption_age = now - last_caption_time

        # Check if a new caption result is ready
        if not result_queue.empty():
            caption = result_queue.get()
            last_caption_time = now
            caption_age = 0
            status = "ready"
            print(f"[Caption] {caption}")
            if tts_queue.empty():
                tts_queue.put(caption)

        # Trigger captioning if interval elapsed or SPACE pressed
        if (caption_age >= interval or force_caption) and caption_queue.empty():
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            caption_queue.put(frame_rgb)
            force_caption = False

        # Draw overlay
        frame = draw_overlay(frame, caption, status, caption_age, interval)
        cv2.imshow("VisualEcho", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            force_caption = True
        elif key == ord('+') or key == ord('='):
            interval = min(interval + 1, 30)
            print(f"[Interval] {interval:.0f}s")
        elif key == ord('-'):
            interval = max(interval - 1, 2)
            print(f"[Interval] {interval:.0f}s")

    # Cleanup
    caption_queue.put(None)
    tts_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    print("\n[VisualEcho] Session ended.")


if __name__ == "__main__":
    main()
