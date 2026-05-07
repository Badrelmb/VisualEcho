"""
VisualEcho — Voice-controlled scene narration for the visually impaired.

Flow:
  1. App starts → speaks welcome + instructions
  2. Listens for voice command
     "start" → captures frame → describes scene → speaks description → speaks instructions again
     "quit"  → speaks goodbye → exits
"""

import cv2
import sys
import time
import os
import io
import tempfile
import numpy as np
from PIL import Image


# ── Instructions ───────────────────────────────────────────────────────────────

WELCOME      = "Visual Echo is ready. The camera is on."
INSTRUCTIONS = "Say Start to describe what is in front of you. Say Quit to exit."
GOODBYE      = "Goodbye. Visual Echo is closing."
THINKING     = "Got it. Analyzing the scene, please wait."


# ── TTS (gTTS — natural voice, cross-platform, requires internet) ──────────────

def init_tts():
    try:
        from gtts import gTTS
        import pygame
        pygame.mixer.init()
        print("[TTS] Using gTTS (online, natural voice)")
        return 'gtts'
    except ImportError:
        print("[TTS] gTTS not found. Run: pip install gTTS pygame")
        sys.exit(1)


def speak(text, tts_type=None):
    """Speak text using gTTS and block until done."""
    print(f"[Speaking] {text}")
    try:
        from gtts import gTTS
        import pygame

        tts = gTTS(text=text, lang='en', slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        # Write to a temp file (more reliable than BytesIO on some systems)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            f.write(mp3_fp.read())
            tmp_path = f.name

        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

        pygame.mixer.music.unload()
        os.remove(tmp_path)

        # Short pause after speaking so mic doesn't catch audio tail
        time.sleep(0.4)

    except Exception as e:
        print(f"[TTS] Error: {e}")


# ── Speech Recognition ─────────────────────────────────────────────────────────

def init_recognizer():
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 2500
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        print("[STT] Speech recognition ready (Google STT)")
        return recognizer
    except ImportError:
        print("[STT] SpeechRecognition not installed. Run: pip install SpeechRecognition pyaudio")
        sys.exit(1)


def listen_for_command(recognizer):
    """
    Listen for a single voice command.
    Returns 'start', 'quit', 'timeout', or 'unknown'.
    """
    import speech_recognition as sr

    with sr.Microphone() as source:
        print("[Listening] Say 'Start' or 'Quit'...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            return 'timeout'

    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"[Heard] {text}")
        if any(w in text for w in ['start', 'describe', 'go', 'scan', 'look', 'what']):
            return 'start'
        elif any(w in text for w in ['quit', 'exit', 'stop', 'close', 'bye', 'end']):
            return 'quit'
        else:
            return 'unknown'
    except sr.UnknownValueError:
        print("[STT] Could not understand audio")
        return 'unknown'
    except sr.RequestError as e:
        print(f"[STT] Google STT request failed: {e}")
        speak("Sorry, I could not connect to the speech service. Please check your internet connection.")
        return 'unknown'
    except Exception as e:
        print(f"[STT] Unexpected error: {e}")
        return 'unknown'


# ── Captioning ─────────────────────────────────────────────────────────────────

def load_captioning_model():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    print("[Model] Loading BLIP captioning model (first run downloads ~1.5GB)...")
    model_id = "Salesforce/blip-image-captioning-large"
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
    import torch
    pil_image = Image.fromarray(frame_rgb.astype(np.uint8)).convert("RGB")
    pixel_values = processor.image_processor(pil_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    prompt = "a photo showing"
    text_inputs = processor.tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            pixel_values=pixel_values,
            input_ids=text_inputs.input_ids,
            max_new_tokens=75,
            num_beams=5,
            repetition_penalty=1.5,
            length_penalty=1.2
        )
    caption = processor.decode(output[0], skip_special_tokens=True)
    if caption.startswith(prompt):
        caption = caption[len(prompt):].strip()
    return caption


# ── Overlay ────────────────────────────────────────────────────────────────────

def wrap_text(text, max_chars=55):
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


def draw_overlay(frame, caption, status):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    bar_h = 120
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    status_colors = {
        'listening': (80,  200, 120),
        'thinking':  (80,  150, 255),
        'speaking':  (255, 180,  50),
        'ready':     (160, 160, 160),
    }
    status_labels = {
        'listening': 'LISTENING...',
        'thinking':  'ANALYZING...',
        'speaking':  'SPEAKING...',
        'ready':     'READY',
    }
    color = status_colors.get(status, (160, 160, 160))
    label = status_labels.get(status, status.upper())
    cv2.putText(frame, label, (w - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    lines = wrap_text(caption if caption else "Waiting for command...")
    y_start = h - bar_h + 24
    for i, line in enumerate(lines[:4]):
        cv2.putText(frame, line, (12, y_start + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  VisualEcho — Voice-Controlled Scene Narration")
    print("=" * 50)

    processor, model, device = load_captioning_model()
    init_tts()
    recognizer = init_recognizer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    caption = ""
    status = "ready"

    def show_frame():
        ret, frame = cap.read()
        if not ret:
            return None
        display = draw_overlay(frame.copy(), caption, status)
        cv2.imshow("VisualEcho", display)
        cv2.waitKey(1)
        return frame

    # Welcome
    status = "speaking"
    show_frame()
    speak(WELCOME)
    speak(INSTRUCTIONS)

    # Main loop
    while True:
        show_frame()

        status = "listening"
        show_frame()
        command = listen_for_command(recognizer)

        if command == 'quit':
            status = "speaking"
            show_frame()
            speak(GOODBYE)
            break

        elif command == 'start':
            status = "speaking"
            show_frame()
            speak(THINKING)

            ret, frame = cap.read()
            if not ret:
                speak("Sorry, I could not read the camera.")
            else:
                status = "thinking"
                show_frame()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                caption = generate_caption(processor, model, device, frame_rgb)
                print(f"[Caption] {caption}")

                status = "speaking"
                show_frame()
                speak(caption)

            speak(INSTRUCTIONS)
            status = "ready"

        elif command == 'timeout':
            # No speech detected — keep listening silently
            status = "listening"

        else:
            status = "speaking"
            show_frame()
            speak("Sorry, I did not understand. " + INSTRUCTIONS)
            status = "ready"

    cap.release()
    cv2.destroyAllWindows()
    print("\n[VisualEcho] Session ended.")


if __name__ == "__main__":
    main()
