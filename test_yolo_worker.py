import pytest
import queue
import threading
from unittest.mock import patch

from main import YOLOWorker, Detection, ANNOUNCE_COOLDOWN

@pytest.fixture
def worker():
    frame_q = queue.Queue()
    result_q = queue.Queue()
    speech_q = queue.Queue()
    stop_evt = threading.Event()
    return YOLOWorker(frame_q, result_q, speech_q, stop_evt)

def test_announce_single_item(worker):
    det = [Detection(label="person", confidence=0.8, bbox=(0.1, 0.1, 0.3, 0.3), priority=1)]
    with patch('time.time', return_value=100.0):
        msg = worker._get_speakable_message(det)
        assert msg == "Warning: person detected."
        assert worker._memory["person"] == 100.0

def test_announce_multiple_items_same_label(worker):
    det = [
        Detection(label="car", confidence=0.8, bbox=(0.1, 0.1, 0.3, 0.3), priority=1),
        Detection(label="car", confidence=0.9, bbox=(0.5, 0.5, 0.6, 0.6), priority=1)
    ]
    with patch('time.time', return_value=100.0):
        msg = worker._get_speakable_message(det)
        assert msg == "Warning: 2 car detected."
        assert worker._memory["car"] == 100.0

def test_cooldown_logic(worker):
    det = [Detection(label="person", confidence=0.8, bbox=(0.1, 0.1, 0.3, 0.3), priority=1)]
    with patch('time.time', return_value=100.0):
        worker._get_speakable_message(det) # Trigger first announce
        
    with patch('time.time', return_value=100.0 + ANNOUNCE_COOLDOWN - 1.0):
        # Within cooldown, shouldn't announce
        msg = worker._get_speakable_message(det)
        assert msg == ""
        
    with patch('time.time', return_value=100.0 + ANNOUNCE_COOLDOWN + 1.0):
        # Outside cooldown, should announce
        msg = worker._get_speakable_message(det)
        assert msg == "Warning: person detected."

def test_multiple_labels(worker):
    det = [
        Detection(label="person", confidence=0.8, bbox=(0.1, 0.1, 0.3, 0.3), priority=1),
        Detection(label="car", confidence=0.9, bbox=(0.5, 0.5, 0.6, 0.6), priority=1)
    ]
    with patch('time.time', return_value=100.0):
        # Preset person memory
        worker._memory["person"] = 100.0 - ANNOUNCE_COOLDOWN + 1.0 # 1s left on cooldown
        msg = worker._get_speakable_message(det)
        # Person is skipped, car is announced
        assert msg == "Warning: car detected."
