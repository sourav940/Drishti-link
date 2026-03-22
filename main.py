"""
Drishti-Link — Real-time Assistive Vision System
Architecture: Multi-threaded Producer-Consumer with Spatial Priority Queue
Author: Senior Architect Build (Production-Grade)
"""

import threading
import queue
import time
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12s] %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("drishti")


# ══════════════════════════════════════════════════════════════════════════
#  Data Contracts
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple          # (x1, y1, x2, y2) normalised [0-1]
    priority: int        # lower = higher priority


@dataclass
class ProcessedFrame:
    frame: np.ndarray
    detections: list[Detection]
    timestamp: float = field(default_factory=time.time)
    fps: float = 0.0


# ══════════════════════════════════════════════════════════════════════════
#  Spatial Priority Queue — object importance map
# ══════════════════════════════════════════════════════════════════════════
import yaml
import os

_DEFAULT_CONFIG = {
    "CONFIDENCE_THRESHOLD": 0.45,
    "ANNOUNCE_COOLDOWN": 5.0,
    "COORD_CHANGE_THRESH": 0.15,
    "MAX_FRAME_QUEUE": 2,
    "MAX_SPEECH_QUEUE": 5,
    "PRIORITY_MAP": {
        "person": 1, "car": 1, "truck": 1, "bus": 1, "motorcycle": 1, "bicycle": 1,
        "stairs": 2, "door": 2, "traffic light": 2, "fire hydrant": 2,
        "chair": 3, "bench": 3, "table": 3, "couch": 3,
        "bottle": 4, "cup": 4, "book": 4, "cell phone": 4, "laptop": 4,
    }
}

def load_config(path="config.yaml"):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    # Merge loaded config with defaults
                    merged = _DEFAULT_CONFIG.copy()
                    merged.update(cfg)
                    return merged
        except Exception as e:
            log.warning(f"Failed to load {path}: {e}")
    return _DEFAULT_CONFIG

_config = load_config()

PRIORITY_MAP         = _config.get("PRIORITY_MAP", _DEFAULT_CONFIG["PRIORITY_MAP"])
CONFIDENCE_THRESHOLD = _config.get("CONFIDENCE_THRESHOLD", _DEFAULT_CONFIG["CONFIDENCE_THRESHOLD"])
ANNOUNCE_COOLDOWN    = _config.get("ANNOUNCE_COOLDOWN", _DEFAULT_CONFIG["ANNOUNCE_COOLDOWN"])
COORD_CHANGE_THRESH  = _config.get("COORD_CHANGE_THRESH", _DEFAULT_CONFIG["COORD_CHANGE_THRESH"])
MAX_FRAME_QUEUE      = _config.get("MAX_FRAME_QUEUE", _DEFAULT_CONFIG["MAX_FRAME_QUEUE"])
MAX_SPEECH_QUEUE     = _config.get("MAX_SPEECH_QUEUE", _DEFAULT_CONFIG["MAX_SPEECH_QUEUE"])


# ══════════════════════════════════════════════════════════════════════════
#  Thread A — Frame Producer
# ══════════════════════════════════════════════════════════════════════════
class FrameProducer(threading.Thread):
    def __init__(self, source: str, frame_q: queue.Queue, stop_evt: threading.Event):
        super().__init__(name="FrameProducer", daemon=True)
        self.source    = source
        self.frame_q   = frame_q
        self.stop_evt  = stop_evt
        self.connected = threading.Event()
        self.fps_actual = 0.0

    def run(self):
        import requests
        log.info(f"Producer: connecting to {self.source}")
        
        is_ip_webcam = self.source.endswith('/video')
        shot_url = self.source.replace('/video', '/shot.jpg') if is_ip_webcam else self.source
        
        if not is_ip_webcam:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                log.error("Producer: cannot open stream — check IP/URL.")
                return

        self.connected.set()
        log.info("Producer: stream opened ✓")
        t0 = time.time()
        frames = 0

        while not self.stop_evt.is_set():
            try:
                if is_ip_webcam:
                    resp = requests.get(shot_url, timeout=2.0)
                    img_arr = np.array(bytearray(resp.content), dtype=np.uint8)
                    frame = cv2.imdecode(img_arr, -1)
                    ret = frame is not None
                else:
                    ret, frame = self.cap.read()
            except Exception as e:
                log.warning(f"Producer: frame read error, retrying… ({e})")
                time.sleep(0.5)
                continue

            if not ret:
                time.sleep(0.1)
                continue

            frames += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                self.fps_actual = frames / elapsed
                frames = 0
                t0 = time.time()

            # Drop frames instead of buffering lag
            if self.frame_q.full():
                try:
                    self.frame_q.get_nowait()
                except queue.Empty:
                    pass

            self.frame_q.put_nowait(frame)

        if not is_ip_webcam and hasattr(self, 'cap'):
            self.cap.release()
        log.info("Producer: stream released.")


# ══════════════════════════════════════════════════════════════════════════
#  Thread B — YOLO Worker
# ══════════════════════════════════════════════════════════════════════════
class YOLOWorker(threading.Thread):
    def __init__(
        self,
        frame_q:     queue.Queue,
        result_q:    queue.Queue,
        speech_q:    queue.Queue,
        stop_evt:    threading.Event,
        model_path:  str = "yolov8n.pt",
    ):
        super().__init__(name="YOLOWorker", daemon=True)
        self.frame_q    = frame_q
        self.result_q   = result_q
        self.speech_q   = speech_q
        self.stop_evt   = stop_evt
        self.model_path = model_path
        self.model      = None
        self.ready      = threading.Event()

        # Detection memory: label -> last_time
        self._memory: dict[str, float] = {}
        self._mem_lock = threading.Lock()

        self.fps = 0.0

    # ── Detection Memory ─────────────────────────────────────────────────
    def _get_speakable_message(self, detections: list[Detection]) -> str:
        # Group by label to count objects ("2 person")
        label_groups = defaultdict(list)
        for d in detections:
            # Removed the d.priority <= 2 filter so it announces EVERY object detected
            label_groups[d.label].append(d)

        now = time.time()
        to_speak = []

        with self._mem_lock:
            for label, group in label_groups.items():
                last_t = self._memory.get(label, 0)
                # Strict cooldown per-label prevents oscillation tracking bugs
                if (now - last_t) >= ANNOUNCE_COOLDOWN:
                    self._memory[label] = now
                    to_speak.append((label, len(group)))

        if to_speak:
            parts = [f"{cnt} {lbl}" if cnt > 1 else lbl for lbl, cnt in to_speak]
            return "Warning: " + ", ".join(parts) + " detected."
        return ""

    # ── Model boot ───────────────────────────────────────────────────────
    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            log.info(f"Worker: YOLOv8 loaded ({self.model_path}) ✓")
            self.ready.set()
        except Exception as exc:
            log.error(f"Worker: YOLO load failed — {exc}")

    # ── Main loop ─────────────────────────────────────────────────────────
    def run(self):
        self._load_model()
        if self.model is None:
            return

        t0 = time.time()
        frames = 0

        while not self.stop_evt.is_set():
            try:
                frame = self.frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            h, w = frame.shape[:2]
            # Speed up CPU inference by drastically downscaling inputs internally (imgsz=320)
            results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, imgsz=320)[0]

            detections: list[Detection] = []
            for box in results.boxes:
                cls_id  = int(box.cls[0])
                label   = self.model.names[cls_id].lower()
                conf    = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Normalise coords
                bbox = (x1 / w, y1 / h, x2 / w, y2 / h)
                priority = PRIORITY_MAP.get(label, 5)

                detections.append(Detection(label, conf, bbox, priority))

            # Sort by priority then confidence
            detections.sort(key=lambda d: (d.priority, -d.confidence))

            # Speech announcements — grouped by label with strict cooldown
            msg = self._get_speakable_message(detections)
            if msg and not self.speech_q.full():
                self.speech_q.put_nowait(msg)

            # FPS
            frames += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                self.fps = frames / elapsed
                frames = 0
                t0 = time.time()

            # Push annotated result
            annotated = self._annotate(frame.copy(), detections, w, h)
            pf = ProcessedFrame(annotated, detections, fps=self.fps)

            if self.result_q.full():
                try:
                    self.result_q.get_nowait()
                except queue.Empty:
                    pass
            self.result_q.put_nowait(pf)

    def _annotate(self, frame, detections, w, h) -> np.ndarray:
        NEON = (0, 255, 180)
        RED  = (0, 80, 255)
        for det in detections:
            x1 = int(det.bbox[0] * w); y1 = int(det.bbox[1] * h)
            x2 = int(det.bbox[2] * w); y2 = int(det.bbox[3] * h)
            color = RED if det.priority == 1 else NEON
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f"{det.label} {det.confidence:.0%}"
            cv2.putText(frame, tag, (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return frame


# ══════════════════════════════════════════════════════════════════════════
#  Thread C — TTS Consumer
# ══════════════════════════════════════════════════════════════════════════
class TTSConsumer(threading.Thread):
    def __init__(self, speech_q: queue.Queue, stop_evt: threading.Event):
        super().__init__(name="TTSConsumer", daemon=True)
        self.speech_q = speech_q
        self.stop_evt = stop_evt
        self.engine   = None
        self.ready    = threading.Event()
        self.speaking = False

    def run(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate",   160)
            self.engine.setProperty("volume", 1.0)

            voices = self.engine.getProperty("voices")
            if voices:
                self.engine.setProperty("voice", voices[0].id)

            self.ready.set()
            log.info("TTS: engine ready ✓")
        except Exception as exc:
            log.error(f"TTS: init failed — {exc}")
            return

        while not self.stop_evt.is_set():
            try:
                msg = self.speech_q.get(timeout=0.3)
            except queue.Empty:
                continue

            log.info(f"TTS: «{msg}»")
            self.speaking = True
            try:
                self.engine.say(msg)
                self.engine.runAndWait()
            except Exception as exc:
                log.warning(f"TTS: speak error — {exc}")
            finally:
                self.speaking = False


# ══════════════════════════════════════════════════════════════════════════
#  Pipeline Orchestrator (agent-ready façade)
# ══════════════════════════════════════════════════════════════════════════
class DrishtiPipeline:
    """
    Single-object façade to start / stop the full detection pipeline.
    Expose result_q so the UI (or any agent) can consume ProcessedFrames.
    """

    def __init__(self, source: str, model_path: str = "yolov8n.pt"):
        self.source     = source
        self.model_path = model_path

        self.frame_q  = queue.Queue(maxsize=MAX_FRAME_QUEUE)
        self.result_q = queue.Queue(maxsize=4)
        self.speech_q = queue.Queue(maxsize=MAX_SPEECH_QUEUE)
        self.stop_evt = threading.Event()

        self.producer = FrameProducer(source, self.frame_q, self.stop_evt)
        self.worker   = YOLOWorker(self.frame_q, self.result_q, self.speech_q,
                                   self.stop_evt, model_path)
        self.tts      = TTSConsumer(self.speech_q, self.stop_evt)

    def start(self):
        log.info("Pipeline: starting threads…")
        self.tts.start()
        self.producer.start()
        self.worker.start()

    def stop(self):
        log.info("Pipeline: stopping…")
        self.stop_evt.set()

    @property
    def is_live(self) -> bool:
        return (
            self.producer.is_alive()
            and self.worker.is_alive()
            and self.tts.is_alive()
        )

    @property
    def inference_fps(self) -> float:
        return self.worker.fps

    @property
    def capture_fps(self) -> float:
        return self.producer.fps_actual


if __name__ == "__main__":
    # Headless smoke-test (runs without GUI)
    SOURCE = "http://192.0.0.4:8080/video"   # change to your IP Webcam URL
    pipe = DrishtiPipeline(SOURCE)
    pipe.start()
    try:
        while True:
            time.sleep(5)
            log.info(f"System Health: FPS={pipe.inference_fps:.1f} | Live={pipe.is_live}")
    except KeyboardInterrupt:
        pipe.stop()