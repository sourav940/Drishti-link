"""
Drishti-Link — Accessibility Dashboard
High-contrast deep-black / neon-yellow Tkinter UI
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import threading
import queue
import time
import math
import platform
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk       # pillow

from main import DrishtiPipeline, ProcessedFrame, PRIORITY_MAP

# ══════════════════════════════════════════════════════════════════════════
#  Palette — Deep Black / Neon Yellow accessibility theme
# ══════════════════════════════════════════════════════════════════════════
C = {
    "bg":          "#0a0a0a",
    "panel":       "#111111",
    "border":      "#1e1e1e",
    "neon":        "#e8ff00",          # neon yellow
    "neon_dim":    "#8a9900",
    "neon_pulse":  "#ffff66",
    "accent_red":  "#ff2d55",
    "accent_teal": "#00ffcc",
    "text":        "#f0f0f0",
    "text_dim":    "#666666",
    "warn":        "#ff6b00",
    "safe":        "#00e676",
    "btn_hover":   "#1c1c00",
}

FONT_MONO  = ("Courier New", 11)
FONT_HUD   = ("Courier New", 13, "bold")
FONT_LABEL = ("Helvetica", 11)
FONT_BIG   = ("Helvetica", 22, "bold")
FONT_TITLE = ("Helvetica", 30, "bold")
FONT_STAT  = ("Courier New", 16, "bold")

VIDEO_W, VIDEO_H = 760, 428          # 16:9 inside the panel


# ══════════════════════════════════════════════════════════════════════════
#  Heartbeat Canvas Widget
# ══════════════════════════════════════════════════════════════════════════
class HeartbeatWidget(tk.Canvas):
    """Animated ECG-style heartbeat line to show the pipeline is alive."""

    HISTORY = 120

    def __init__(self, master, **kw):
        super().__init__(master, bg=C["panel"], highlightthickness=0,
                         height=60, **kw)
        self._data   = [0.0] * self.HISTORY
        self._beat   = 0.0
        self._alive  = True
        self._dead   = False
        self.after(30, self._tick)

    def pulse(self, value: float = 1.0):
        """Feed a new sample (0-1)."""
        self._beat = max(self._beat, value)

    def set_dead(self, dead: bool):
        self._dead = dead

    def _tick(self):
        # Decay current beat into history
        self._data.pop(0)
        self._data.append(self._beat)
        self._beat *= 0.55

        self._draw()
        self.after(30, self._tick)

    def _draw(self):
        self.delete("all")
        w = self.winfo_width() or 400
        h = self.winfo_height() or 60
        if w < 10:
            return

        n   = len(self._data)
        pts = []
        for i, v in enumerate(self._data):
            x = int(i * w / (n - 1))
            y = int((h / 2) - v * (h / 2 - 6))
            pts.extend([x, y])

        color = C["text_dim"] if self._dead else C["neon"]
        if len(pts) >= 4:
            self.create_line(pts, fill=color, width=2, smooth=True)

        # Glow dot at the head
        if not self._dead:
            lx, ly = pts[-2], pts[-1]
            r = 4
            self.create_oval(lx - r, ly - r, lx + r, ly + r,
                             fill=C["neon_pulse"], outline="")


# ══════════════════════════════════════════════════════════════════════════
#  Detection Feed List
# ══════════════════════════════════════════════════════════════════════════
class DetectionLog(tk.Frame):
    MAX_ROWS = 8

    def __init__(self, master, **kw):
        super().__init__(master, bg=C["panel"], **kw)
        tk.Label(self, text="LIVE DETECTIONS", font=FONT_HUD,
                 bg=C["panel"], fg=C["neon"]).pack(anchor="w", padx=8, pady=(8, 4))

        self._rows: list[tk.Label] = []
        for _ in range(self.MAX_ROWS):
            lbl = tk.Label(self, text="", font=FONT_MONO,
                           bg=C["panel"], fg=C["text"], anchor="w",
                           padx=8, pady=2)
            lbl.pack(fill="x")
            self._rows.append(lbl)

    def update_detections(self, detections):
        priority_colors = {1: C["accent_red"], 2: C["warn"],
                           3: C["accent_teal"], 4: C["text_dim"], 5: C["text_dim"]}
        for i, row in enumerate(self._rows):
            if i < len(detections):
                d   = detections[i]
                pri = d.priority
                tag = f"[P{pri}] {d.label:<18} {d.confidence:>5.1%}"
                row.config(text=tag, fg=priority_colors.get(pri, C["text"]))
            else:
                row.config(text="", fg=C["text"])


# ══════════════════════════════════════════════════════════════════════════
#  Large Accessible Button
# ══════════════════════════════════════════════════════════════════════════
class BigButton(tk.Button):
    def __init__(self, master, text, command, color=None, **kw):
        color = color or C["neon"]
        super().__init__(
            master,
            text=text,
            command=command,
            font=FONT_BIG,
            bg=C["border"],
            fg=color,
            activebackground=C["btn_hover"],
            activeforeground=color,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=20,
            pady=14,
            **kw,
        )
        self.bind("<Enter>", lambda e: self.config(bg="#222200"))
        self.bind("<Leave>", lambda e: self.config(bg=C["border"]))


# ══════════════════════════════════════════════════════════════════════════
#  Main Dashboard Window
# ══════════════════════════════════════════════════════════════════════════
class DrishtiDashboard(tk.Tk):

    DEFAULT_URL = "http://192.168.1.100:8080/video"

    def __init__(self):
        super().__init__()
        self.title("Drishti-Link  ·  Assistive Vision System")
        self.configure(bg=C["bg"])
        self.minsize(1200, 720)

        self._pipeline: Optional[DrishtiPipeline] = None
        self._running  = False
        self._frame_id = 0
        self._last_fps_detections = 0

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(33, self._poll_results)   # ~30 Hz UI refresh

    # ── Layout ────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top title bar
        top = tk.Frame(self, bg=C["bg"], pady=6)
        top.pack(fill="x", padx=16)

        tk.Label(top, text="◉ DRISHTI-LINK", font=FONT_TITLE,
                 bg=C["bg"], fg=C["neon"]).pack(side="left")
        tk.Label(top, text="Real-Time Assistive Vision  |  100% Offline",
                 font=FONT_LABEL, bg=C["bg"], fg=C["text_dim"]).pack(side="left", padx=16)

        self._status_lbl = tk.Label(top, text="● OFFLINE", font=FONT_HUD,
                                    bg=C["bg"], fg=C["accent_red"])
        self._status_lbl.pack(side="right")

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # ── Body: left video | right controls
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=12, pady=8)

        # Left column — video
        left = tk.Frame(body, bg=C["panel"], bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self._video_lbl = tk.Label(
            left, bg="black",
            width=VIDEO_W, height=VIDEO_H,
        )
        self._video_lbl.pack(pady=(8, 4), padx=8)

        # HUD strip under video
        hud = tk.Frame(left, bg=C["panel"])
        hud.pack(fill="x", padx=8, pady=(0, 4))

        self._fps_lbl    = tk.Label(hud, text="FPS  --", font=FONT_HUD,
                                    bg=C["panel"], fg=C["neon_dim"])
        self._fps_lbl.pack(side="left", padx=8)

        self._det_count  = tk.Label(hud, text="DETECTIONS  0", font=FONT_HUD,
                                    bg=C["panel"], fg=C["neon_dim"])
        self._det_count.pack(side="left", padx=8)

        self._tts_lbl    = tk.Label(hud, text="🔇 TTS IDLE", font=FONT_HUD,
                                    bg=C["panel"], fg=C["text_dim"])
        self._tts_lbl.pack(side="right", padx=8)

        # Heartbeat
        self._heartbeat = HeartbeatWidget(left)
        self._heartbeat.pack(fill="x", padx=8, pady=(0, 8))

        # Right column — controls
        right = tk.Frame(body, bg=C["bg"], width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # ── URL input
        tk.Label(right, text="CAMERA URL", font=FONT_HUD,
                 bg=C["bg"], fg=C["neon"]).pack(anchor="w", pady=(4, 2))

        url_frame = tk.Frame(right, bg=C["border"], pady=1)
        url_frame.pack(fill="x", pady=(0, 10))
        self._url_var = tk.StringVar(value=self.DEFAULT_URL)
        url_entry = tk.Entry(url_frame, textvariable=self._url_var,
                             font=FONT_MONO, bg=C["panel"], fg=C["neon"],
                             insertbackground=C["neon"],
                             relief="flat", bd=6)
        url_entry.pack(fill="x")

        # ── Big buttons
        self._start_btn = BigButton(right, text="▶  START VISION",
                                    command=self._start, color=C["neon"])
        self._start_btn.pack(fill="x", pady=4)

        self._stop_btn  = BigButton(right, text="■  STOP",
                                    command=self._stop, color=C["accent_red"])
        self._stop_btn.pack(fill="x", pady=4)
        self._stop_btn.config(state="disabled")

        BigButton(right, text="⚙  CALIBRATE CAMERA",
                  command=self._calibrate, color=C["accent_teal"]).pack(fill="x", pady=4)

        BigButton(right, text="🔊  TEST VOICE",
                  command=self._test_voice, color=C["warn"]).pack(fill="x", pady=4)

        tk.Frame(right, bg=C["border"], height=1).pack(fill="x", pady=10)

        # ── Detection log
        self._det_log = DetectionLog(right)
        self._det_log.pack(fill="both", expand=True)

        tk.Frame(right, bg=C["border"], height=1).pack(fill="x", pady=8)

        # ── Priority legend
        legend = tk.Frame(right, bg=C["bg"])
        legend.pack(fill="x")
        tk.Label(legend, text="PRIORITY KEY", font=FONT_HUD,
                 bg=C["bg"], fg=C["neon"]).pack(anchor="w")
        rows = [
            ("P1 — Person / Vehicle", C["accent_red"]),
            ("P2 — Stairs / Doors",   C["warn"]),
            ("P3 — Furniture",        C["accent_teal"]),
            ("P4 — Small Items",      C["text_dim"]),
        ]
        for txt, clr in rows:
            tk.Label(legend, text=f"  {txt}", font=FONT_MONO,
                     bg=C["bg"], fg=clr, anchor="w").pack(fill="x")

    # ── Pipeline controls ─────────────────────────────────────────────────
    def _start(self):
        url = self._url_var.get().strip()
        if not url:
            self._flash_status("⚠ Enter a valid URL", C["warn"])
            return

        self._pipeline = DrishtiPipeline(url)
        self._pipeline.start()
        self._running = True

        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._status_lbl.config(text="● LIVE", fg=C["safe"])
        self._heartbeat.set_dead(False)
        log.info("Dashboard: pipeline started.")

    def _stop(self):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        self._running = False
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._status_lbl.config(text="● OFFLINE", fg=C["accent_red"])
        self._heartbeat.set_dead(True)
        self._fps_lbl.config(text="FPS  --")
        self._det_count.config(text="DETECTIONS  0")
        # Clear video frame
        self._video_lbl.config(image="", bg="black")

    def _calibrate(self):
        """Placeholder for camera calibration routine."""
        self._flash_status("Calibration: capture reference frame…", C["accent_teal"])

    def _test_voice(self):
        if self._pipeline:
            self._pipeline.speech_q.put_nowait(
                "Drishti-Link voice test. System is operational."
            )
        else:
            # Speak without a live pipeline
            import pyttsx3, threading
            def _speak():
                e = pyttsx3.init()
                e.say("Drishti-Link voice test. System is operational.")
                e.runAndWait()
            threading.Thread(target=_speak, daemon=True).start()

    # ── Result polling ─────────────────────────────────────────────────────
    def _poll_results(self):
        if self._running and self._pipeline:
            try:
                pf: ProcessedFrame = self._pipeline.result_q.get_nowait()
                self._update_video(pf.frame)
                self._det_log.update_detections(pf.detections)
                self._fps_lbl.config(text=f"FPS  {pf.fps:.1f}")
                self._det_count.config(text=f"DETECTIONS  {len(pf.detections)}")

                # Feed heartbeat
                beat = min(1.0, len(pf.detections) / 5 + 0.3)
                self._heartbeat.pulse(beat)

                # TTS indicator
                if self._pipeline.tts.speaking:
                    self._tts_lbl.config(text="🔊 SPEAKING", fg=C["neon"])
                else:
                    self._tts_lbl.config(text="🔇 TTS IDLE", fg=C["text_dim"])

                # Pipeline health check
                if not self._pipeline.is_live:
                    self._flash_status("⚠ Thread died — restart needed", C["accent_red"])

            except queue.Empty:
                pass

        self.after(33, self._poll_results)

    def _update_video(self, frame: np.ndarray):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img       = Image.fromarray(frame_rgb).resize(
            (VIDEO_W, VIDEO_H), Image.LANCZOS
        )
        imgtk = ImageTk.PhotoImage(img)
        self._video_lbl.config(image=imgtk)
        self._video_lbl.image = imgtk   # keep reference

    # ── Utilities ─────────────────────────────────────────────────────────
    def _flash_status(self, msg: str, color: str):
        self._status_lbl.config(text=msg, fg=color)
        self.after(3000, lambda: self._status_lbl.config(
            text="● LIVE" if self._running else "● OFFLINE",
            fg=C["safe"] if self._running else C["accent_red"],
        ))

    def _on_close(self):
        self._stop()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    app = DrishtiDashboard()
    app.mainloop()