#!/usr/bin/env python3
"""
bluebird_cam.py

Pi 5 plus Pi Camera Module v3 (or USB webcam) offline bird classifier with:
- Motion gated classification
- Web interface with MJPEG stream and JSON status
- Autofocus enabled
- Auto white balance enabled
- Auto exposure enabled
- Camera frames treated as BGR
- Image flipped before processing
- Offline text to speech using Piper in pure Python (no subprocess)
- Cached audio for: Bluebird, Not bluebird, Not a bird
- Differentiates between Bluebird, Not bluebird, Not a bird
- Stops motion detection and classification when the image is too dark
- No special characters in any text

Install:
  sudo apt update
  sudo apt install -y python3-picamera2 python3-opencv libsndfile1
  pip install -U flask transformers pillow torch piper-tts sounddevice numpy

Run:
  python bluebird_cam.py \
    --model-dir ./models/Birds-Classifier-EfficientNetB2 \
    --piper-model /home/pi/piper-voices/en_US-lessac-medium.onnx \
    --host 0.0.0.0 --port 8000
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import time
import threading
import queue
import subprocess
import re
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

import cv2
from flask import Flask, Response, jsonify

import sounddevice as sd
from piper import PiperVoice


# -------------------------------------------------
# Piper TTS cached phrases (pure Python, no subprocess)
# -------------------------------------------------

def piper_stream_to_float32(voice: PiperVoice, text: str) -> np.ndarray:
    parts: List[np.ndarray] = []
    for ch in voice.synthesize(text):
        arr = np.asarray(ch.audio_float_array, dtype=np.float32).reshape(-1)
        if arr.size:
            parts.append(arr)
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0)


class CachedPiperTTS:
    def __init__(self, model_path: str, phrases: List[str], device: Union[str, int, None] = None) -> None:
        self.voice = PiperVoice.load(model_path)
        self.sample_rate = int(self.voice.config.sample_rate)
        self.device = device

        self.cache: Dict[str, np.ndarray] = {}
        for p in phrases:
            wav = piper_stream_to_float32(self.voice, p)
            self.cache[p] = np.ascontiguousarray(wav, dtype=np.float32)

        self.q: "queue.Queue[str]" = queue.Queue(maxsize=20)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
        self.muted = False
        self.volume = 1.0
        self.lock = threading.Lock()

    def say(self, phrase: str) -> None:
        if phrase not in self.cache:
            return
        try:
            self.q.put_nowait(phrase)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._stop.set()
    
    def set_muted(self, muted: bool) -> None:
        with self.lock:
            self.muted = muted
    
    def set_volume(self, volume: float) -> None:
        with self.lock:
            self.volume = max(0.0, min(1.0, volume))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                phrase = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                with self.lock:
                    is_muted = self.muted
                    vol = self.volume
                
                if not is_muted:
                    audio = self.cache.get(phrase)
                    if audio is not None and audio.size:
                        audio_scaled = audio * vol
                        sd.play(audio_scaled, samplerate=self.sample_rate, device=self.device, blocking=True)
            except Exception:
                pass
            finally:
                self.q.task_done()


# -------------------------------------------------
# Motion detection and darkness detection
# -------------------------------------------------

def motion_score_bgr(
    bgr: np.ndarray,
    prev_gray: Optional[np.ndarray],
    small_size: Tuple[int, int],
    diff_threshold: int,
) -> Tuple[float, np.ndarray]:
    small = cv2.resize(bgr, small_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        return 0.0, gray

    diff = cv2.absdiff(gray, prev_gray)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    score = float(np.count_nonzero(mask)) / float(mask.size)
    return score, gray


def brightness_mean_bgr(bgr: np.ndarray, small_size: Tuple[int, int]) -> float:
    small = cv2.resize(bgr, small_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


# -------------------------------------------------
# Shared state
# -------------------------------------------------

@dataclass
class Status:
    running: bool = True
    ts: float = 0.0

    too_dark: bool = False
    brightness: float = 0.0
    exposure_comp: float = 0.0
    analogue_gain: float = 1.0

    motion_score: float = 0.0
    motion: bool = False
    motion_threshold: float = 0.03
    dark_threshold: float = 50.0
    last_classify_ts: float = 0.0

    category: str = ""
    top1_label: str = ""
    top1_score: float = 0.0
    topk: Optional[List[Dict[str, Any]]] = None

    last_spoken_ts: float = 0.0
    
    power_watts: float = 0.0
    power_avg_watts: float = 0.0
    battery_runtime_hours: float = 0.0

    storage_free_gb: float = 0.0
    storage_total_gb: float = 0.0
    
    muted: bool = True
    volume: float = 1.0
    recording: bool = False
    flip: bool = True


class Shared:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: Optional[bytes] = None
        self.status = Status()


# -------------------------------------------------
# Power monitoring
# -------------------------------------------------

def read_pmic_adc() -> Optional[Dict[str, Dict[str, float]]]:
    """Read PMIC ADC values using vcgencmd."""
    try:
        result = subprocess.run(
            ['vcgencmd', 'pmic_read_adc'],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0
        )
        
        data = {}
        for line in result.stdout.strip().split('\n'):
            current_match = re.match(r'\s*(\S+)\s+current\(\d+\)=([\d.]+)A', line)
            volt_match = re.match(r'\s*(\S+)\s+volt\(\d+\)=([\d.]+)V', line)
            
            if current_match:
                name, value = current_match.groups()
                data[name] = data.get(name, {})
                data[name]['current'] = float(value)
            elif volt_match:
                name, value = volt_match.groups()
                data[name] = data.get(name, {})
                data[name]['voltage'] = float(value)
        
        return data
    except Exception:
        return None


def calculate_total_power(pmic_data: Dict[str, Dict[str, float]]) -> float:
    """Calculate total power consumption from PMIC data."""
    total_power = 0.0
    
    voltage_rails = {}
    current_rails = {}
    
    for name, measurements in pmic_data.items():
        if name.endswith('_V'):
            base_name = name[:-2]
            if 'voltage' in measurements:
                voltage_rails[base_name] = measurements['voltage']
        elif name.endswith('_A'):
            base_name = name[:-2]
            if 'current' in measurements:
                current_rails[base_name] = measurements['current']
    
    for base_name in voltage_rails:
        if base_name in current_rails:
            power = voltage_rails[base_name] * current_rails[base_name]
            if power > 0.001:  # Only count meaningful power
                total_power += power
    
    return total_power


def get_storage_stats(path: str = "/") -> Tuple[float, float]:
    """Return (total_gb, free_gb) for the filesystem at path."""
    st = os.statvfs(path)
    total = float(st.f_frsize * st.f_blocks)
    free = float(st.f_frsize * st.f_bavail)
    gb = 1024.0 ** 3
    return total / gb, free / gb


def power_monitor_worker(shared: Shared, battery_wh: float, avg_window: int) -> None:
    """Background thread to monitor power consumption."""
    power_history = deque(maxlen=avg_window)
    
    while True:
        try:
            pmic_data = read_pmic_adc()
            
            if pmic_data:
                total_power = calculate_total_power(pmic_data)
                power_history.append(total_power)
                
                avg_power = sum(power_history) / len(power_history) if power_history else 0.0
                runtime_hours = battery_wh / avg_power if avg_power > 0 else 0.0
                
                with shared.lock:
                    shared.status.power_watts = total_power
                    shared.status.power_avg_watts = avg_power
                    shared.status.battery_runtime_hours = runtime_hours

            try:
                total_gb, free_gb = get_storage_stats("/")
            except Exception:
                total_gb, free_gb = 0.0, 0.0

            with shared.lock:
                shared.status.storage_total_gb = total_gb
                shared.status.storage_free_gb = free_gb
        except Exception:
            pass
        
        time.sleep(2.0)


# -------------------------------------------------
# Web app
# -------------------------------------------------

def make_app(shared: Shared, tts: CachedPiperTTS) -> Flask:
    app = Flask(__name__)

    INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Bluebird Camera</title>
<style>
body { font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }
.wrap { max-width: 1000px; margin: auto; }
.box { background: #1b1b1b; border: 1px solid #333; border-radius: 10px; padding: 12px; margin-bottom: 12px; }
img { width: 100%; border-radius: 8px; border: 1px solid #333; }
pre { white-space: pre-wrap; margin: 0; }
.row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
.pill { background: #242424; border: 1px solid #333; border-radius: 999px; padding: 6px 10px; }
a { color: #9db7ff; text-decoration: none; }
</style>
</head>
<body>
<div class="wrap">
  <div class="box row">
    <div class="pill">Bluebird Camera</div>
    <div class="pill"><a href="/status">status</a></div>
    <div class="pill" id="clock">time</div>
  </div>

  <div class="box">
    <img src="/stream" alt="stream"/>
  </div>

  <div class="box">
    <div class="row">
      <div class="pill" id="dark">Dark</div>
      <div class="pill" id="motion">Motion</div>
      <div class="pill" id="category">Category</div>
      <div class="pill" id="top1">Top1</div>
      <div class="pill" id="last">Last</div>
    </div>
    <pre id="topk"></pre>
  </div>

  <div class="box">
    <div class="row">
      <div class="pill" id="power">Power</div>
      <div class="pill" id="battery">Runtime</div>
            <div class="pill" id="storage">Storage</div>
    </div>
  </div>

    <div class="box">
        <div style="margin-bottom: 8px;"><strong>Recording</strong></div>
        <div class="row">
            <button id="recordBtn" class="pill" style="cursor: pointer; border: 2px solid #555;">Record off</button>
        </div>
    </div>

    <div class="box">
        <div style="margin-bottom: 8px;"><strong>Camera</strong></div>
        <div class="row">
            <button id="flipBtn" class="pill" style="cursor: pointer; border: 2px solid #555;">Flip off</button>
        </div>
    </div>

    <div class="box">
        <div style="margin-bottom: 8px;"><strong>Exposure Control</strong></div>
        <div class="row" style="gap: 16px;">
            <div class="pill" style="flex: 1; max-width: 400px;">
                <label for="exposureSlider" style="margin-right: 8px;">Exposure Comp</label>
                <input type="range" id="exposureSlider" min="-4" max="4" value="0" step="0.2" style="width: 200px; vertical-align: middle;"/>
                <span id="exposureValue" style="margin-left: 8px;">0.0</span>
            </div>
            <div class="pill" style="flex: 1; max-width: 400px;">
                <label for="gainSlider" style="margin-right: 8px;">Analogue Gain</label>
                <input type="range" id="gainSlider" min="1" max="8" value="1" step="0.1" style="width: 200px; vertical-align: middle;"/>
                <span id="gainValue" style="margin-left: 8px;">1.0</span>
            </div>
        </div>
    </div>

    <div class="box">
        <div style="margin-bottom: 8px;"><strong>Detection Thresholds</strong></div>
        <div class="row" style="gap: 16px;">
            <div class="pill" style="flex: 1; max-width: 400px;">
                <label for="motionSlider" style="margin-right: 8px;">Motion</label>
                <input type="range" id="motionSlider" min="0" max="100" value="3" style="width: 200px; vertical-align: middle;"/>
                <span id="motionValue" style="margin-left: 8px;">0.03</span>
            </div>
            <div class="pill" style="flex: 1; max-width: 400px;">
                <label for="darkSlider" style="margin-right: 8px;">Dark</label>
                <input type="range" id="darkSlider" min="0" max="200" value="50" style="width: 200px; vertical-align: middle;"/>
                <span id="darkValue" style="margin-left: 8px;">50</span>
            </div>
        </div>
    </div>

  <div class="box">
    <div style="margin-bottom: 8px;"><strong>Audio Control</strong></div>
    <div class="row">
      <button id="muteBtn" class="pill" style="cursor: pointer; border: 2px solid #555;">Mute</button>
      <div class="pill" style="flex: 1; max-width: 400px;">
        <label for="volumeSlider" style="margin-right: 8px;">Volume</label>
        <input type="range" id="volumeSlider" min="0" max="100" value="100" style="width: 200px; vertical-align: middle;"/>
        <span id="volumeValue" style="margin-left: 8px;">100%</span>
      </div>
    </div>
  </div>
</div>

<script>
async function tick() {
  try {
    const r = await fetch('/status', {cache:'no-store'});
    const s = await r.json();

    document.getElementById('clock').textContent = new Date().toLocaleString();

    document.getElementById('dark').textContent =
      'Dark ' + (s.too_dark ? 'yes' : 'no') + ' ' + (s.brightness || 0).toFixed(1);

    document.getElementById('motion').textContent =
      'Motion ' + (s.motion ? 'yes' : 'no') + ' ' + (s.motion_score || 0).toFixed(3);

    document.getElementById('category').textContent =
      s.category ? 'Category ' + s.category : 'Category none';

    document.getElementById('top1').textContent =
      s.top1_label ? 'Top1 ' + s.top1_label + ' ' + (s.top1_score || 0).toFixed(3) : 'Top1 none';

    document.getElementById('last').textContent =
      s.last_classify_ts ? 'Last ' + new Date(s.last_classify_ts*1000).toLocaleTimeString() : 'Last none';

    document.getElementById('topk').textContent =
      (s.topk && s.topk.length) ? s.topk.map(p => p.label + ' ' + p.score.toFixed(3)).join('\\n') : '';

    // Power stats
    const pwr = s.power_watts || 0;
    const avg = s.power_avg_watts || 0;
    const hrs = s.battery_runtime_hours || 0;
    const days = Math.floor(hrs / 24);
    const h = Math.floor(hrs % 24);
    const m = Math.floor((hrs * 60) % 60);
    const runtime = days > 0 ? days + 'd ' + h + 'h ' + m + 'm' : h + 'h ' + m + 'm';
    
    document.getElementById('power').textContent =
      'Power ' + pwr.toFixed(2) + 'W avg ' + avg.toFixed(2) + 'W';
    
    document.getElementById('battery').textContent =
      'Runtime ' + runtime + ' on 50Wh';

        const freeGb = s.storage_free_gb || 0;
        const totalGb = s.storage_total_gb || 0;
        const pctFree = totalGb > 0 ? (freeGb / totalGb) * 100 : 0;
        document.getElementById('storage').textContent =
            'Storage ' + freeGb.toFixed(1) + '/' + totalGb.toFixed(1) + ' GB (' + pctFree.toFixed(0) + '% free)';

        // Threshold sliders
        const motionSlider = document.getElementById('motionSlider');
        const motionValue = document.getElementById('motionValue');
        const motionPercent = Math.round((s.motion_threshold || 0) * 100);
        if (!motionSlider.matches(':active')) {
            motionSlider.value = motionPercent;
        }
        motionValue.textContent = (s.motion_threshold || 0).toFixed(3);

        const darkSlider = document.getElementById('darkSlider');
        const darkValue = document.getElementById('darkValue');
        const darkVal = Math.round(s.dark_threshold || 0);
        if (!darkSlider.matches(':active')) {
            darkSlider.value = darkVal;
        }
        darkValue.textContent = darkVal.toString();

        // Exposure controls
        const exposureSlider = document.getElementById('exposureSlider');
        const exposureValue = document.getElementById('exposureValue');
        const expComp = s.exposure_comp || 0;
        if (!exposureSlider.matches(':active')) {
            exposureSlider.value = expComp.toFixed(1);
        }
        exposureValue.textContent = expComp.toFixed(1);

        const gainSlider = document.getElementById('gainSlider');
        const gainValue = document.getElementById('gainValue');
        const ag = s.analogue_gain || 1.0;
        if (!gainSlider.matches(':active')) {
            gainSlider.value = ag.toFixed(1);
        }
        gainValue.textContent = ag.toFixed(1);
    
    // Update mute button
    const muteBtn = document.getElementById('muteBtn');
    if (s.muted) {
      muteBtn.textContent = 'Unmute';
      muteBtn.style.background = '#aa3333';
    } else {
      muteBtn.textContent = 'Mute';
      muteBtn.style.background = '#242424';
    }
    
    // Update volume display (without triggering change event)
    const volSlider = document.getElementById('volumeSlider');
    const volValue = document.getElementById('volumeValue');
    const volumePercent = Math.round((s.volume || 1.0) * 100);
    if (!volSlider.matches(':active')) {
      volSlider.value = volumePercent;
    }
    volValue.textContent = volumePercent + '%';

        // Recording button
        const recordBtn = document.getElementById('recordBtn');
        if (s.recording) {
            recordBtn.textContent = 'Recording on';
            recordBtn.style.background = '#337733';
        } else {
            recordBtn.textContent = 'Record off';
            recordBtn.style.background = '#242424';
        }

        // Flip button
        const flipBtn = document.getElementById('flipBtn');
        if (s.flip) {
            flipBtn.textContent = 'Flip on';
            flipBtn.style.background = '#337733';
        } else {
            flipBtn.textContent = 'Flip off';
            flipBtn.style.background = '#242424';
        }
  } catch (e) {}
}
setInterval(tick, 500);
tick();

// Mute button handler
document.getElementById('muteBtn').addEventListener('click', async () => {
  try {
    const r = await fetch('/status', {cache:'no-store'});
    const s = await r.json();
    const newMuted = !s.muted;
    await fetch('/mute?value=' + (newMuted ? '1' : '0'));
    tick();
  } catch (e) {}
});

// Volume slider handler
let volumeTimeout = null;
document.getElementById('volumeSlider').addEventListener('input', (e) => {
  const vol = parseInt(e.target.value);
  document.getElementById('volumeValue').textContent = vol + '%';
  
  // Debounce API calls
  if (volumeTimeout) clearTimeout(volumeTimeout);
  volumeTimeout = setTimeout(async () => {
    try {
      await fetch('/volume?value=' + (vol / 100));
    } catch (e) {}
  }, 200);
});

// Motion slider handler
let motionTimeout = null;
document.getElementById('motionSlider').addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    document.getElementById('motionValue').textContent = (val / 100).toFixed(3);
    if (motionTimeout) clearTimeout(motionTimeout);
    motionTimeout = setTimeout(async () => {
        try {
            await fetch('/motion_threshold?value=' + (val / 100));
        } catch (e) {}
    }, 200);
});

// Dark slider handler
let darkTimeout = null;
document.getElementById('darkSlider').addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    document.getElementById('darkValue').textContent = val.toString();
    if (darkTimeout) clearTimeout(darkTimeout);
    darkTimeout = setTimeout(async () => {
        try {
            await fetch('/dark_threshold?value=' + val);
        } catch (e) {}
    }, 200);
});

// Record button handler
document.getElementById('recordBtn').addEventListener('click', async () => {
    try {
        const r = await fetch('/status', {cache:'no-store'});
        const s = await r.json();
        const newRecording = !s.recording;
        await fetch('/record?value=' + (newRecording ? '1' : '0'));
        tick();
    } catch (e) {}
});

// Flip button handler
document.getElementById('flipBtn').addEventListener('click', async () => {
    try {
        const r = await fetch('/status', {cache:'no-store'});
        const s = await r.json();
        const newFlip = !s.flip;
        await fetch('/flip?value=' + (newFlip ? '1' : '0'));
        tick();
    } catch (e) {}
});

// Exposure compensation slider handler
let exposureTimeout = null;
document.getElementById('exposureSlider').addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    document.getElementById('exposureValue').textContent = val.toFixed(1);
    if (exposureTimeout) clearTimeout(exposureTimeout);
    exposureTimeout = setTimeout(async () => {
        try {
            await fetch('/exposure_comp?value=' + val);
        } catch (e) {}
    }, 200);
});

// Analogue gain slider handler
let gainTimeout = null;
document.getElementById('gainSlider').addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    document.getElementById('gainValue').textContent = val.toFixed(1);
    if (gainTimeout) clearTimeout(gainTimeout);
    gainTimeout = setTimeout(async () => {
        try {
            await fetch('/analogue_gain?value=' + val);
        } catch (e) {}
    }, 200);
});
</script>
</body>
</html>
"""

    @app.route("/")
    def index():
        return Response(INDEX_HTML, mimetype="text/html")

    @app.route("/status")
    def status():
        with shared.lock:
            return jsonify(asdict(shared.status))

    @app.route("/stream")
    def stream():
        def gen():
            while True:
                with shared.lock:
                    frame = shared.jpeg
                if frame is None:
                    time.sleep(0.05)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame + b"\r\n"
                )
                time.sleep(0.03)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")
    
    @app.route("/mute")
    def mute():
        from flask import request
        value = request.args.get('value', '0')
        muted = value == '1'
        tts.set_muted(muted)
        with shared.lock:
            shared.status.muted = muted
        return jsonify({"muted": muted})
    
    @app.route("/volume")
    def volume():
        from flask import request
        value = request.args.get('value', '1.0')
        try:
            vol = float(value)
            vol = max(0.0, min(1.0, vol))
            tts.set_volume(vol)
            with shared.lock:
                shared.status.volume = vol
            return jsonify({"volume": vol})
        except ValueError:
            return jsonify({"error": "invalid value"}), 400

    @app.route("/motion_threshold")
    def motion_threshold():
        from flask import request
        value = request.args.get('value')
        try:
            mt = float(value)
            # Clamp between 0 and 1 to avoid invalid ratios
            mt = max(0.0, min(1.0, mt))
            with shared.lock:
                shared.status.motion_threshold = mt
            return jsonify({"motion_threshold": mt})
        except (TypeError, ValueError):
            return jsonify({"error": "invalid value"}), 400

    @app.route("/dark_threshold")
    def dark_threshold():
        from flask import request
        value = request.args.get('value')
        try:
            dt = float(value)
            dt = max(0.0, dt)
            with shared.lock:
                shared.status.dark_threshold = dt
            return jsonify({"dark_threshold": dt})
        except (TypeError, ValueError):
            return jsonify({"error": "invalid value"}), 400

    @app.route("/record")
    def record():
        from flask import request
        value = request.args.get('value', '0')
        recording = value == '1'
        with shared.lock:
            shared.status.recording = recording
        return jsonify({"recording": recording})

    @app.route("/flip")
    def flip():
        from flask import request
        value = request.args.get('value', '0')
        flip_enabled = value == '1'
        with shared.lock:
            shared.status.flip = flip_enabled
        return jsonify({"flip": flip_enabled})

    @app.route("/exposure_comp")
    def exposure_comp():
        from flask import request
        value = request.args.get('value')
        try:
            ec = float(value)
            ec = max(-4.0, min(4.0, ec))
            with shared.lock:
                shared.status.exposure_comp = ec
            return jsonify({"exposure_comp": ec})
        except (TypeError, ValueError):
            return jsonify({"error": "invalid value"}), 400

    @app.route("/analogue_gain")
    def analogue_gain():
        from flask import request
        value = request.args.get('value')
        try:
            ag = float(value)
            ag = max(1.0, min(8.0, ag))
            with shared.lock:
                shared.status.analogue_gain = ag
            return jsonify({"analogue_gain": ag})
        except (TypeError, ValueError):
            return jsonify({"error": "invalid value"}), 400

    return app


# -------------------------------------------------
# Categorization
# -------------------------------------------------

def categorize(preds: List[Dict[str, Any]], blue_kw: str, not_bird_threshold: float) -> str:
    """Categorize based on top 5 predictions. Check if any are bluebird."""
    if not preds:
        return ""
    
    # Check all top-5 predictions for bluebird keyword
    if blue_kw:
        for pred in preds[:5]:
            label = str(pred["label"]).lower()
            if blue_kw in label:
                return "Bluebird"

    # Check if top-1 score is below not-bird threshold
    top1_score = float(preds[0]["score"])
    if top1_score < not_bird_threshold:
        return "Not a bird"
    
    return "Not bluebird"


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "unknown"


def save_classified_frame(bgr: np.ndarray, category: str, ts: float) -> None:
    if not category:
        return

    try:
        slug = _slugify_label(category)
        base_dir = os.path.join(os.path.expanduser("~"), "data", slug)
        os.makedirs(base_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(ts))
        ms = int((ts - int(ts)) * 1000)
        filename = f"{timestamp}-{ms:03d}.jpg"
        path = os.path.join(base_dir, filename)

        # Save current frame to disk; ignore failures to avoid breaking the loop
        cv2.imwrite(path, bgr)
    except Exception:
        pass


# -------------------------------------------------
# Camera worker
# -------------------------------------------------

def camera_worker(args, shared: Shared, tts: CachedPiperTTS) -> None:
    size = (args.width, args.height)
    small_size_motion = (args.motion_w, args.motion_h)
    small_size_brightness = (args.brightness_w, args.brightness_h)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir, local_files_only=True)
    model.to(device)

    clf = pipeline(
        "image-classification",
        model=model,
        image_processor=processor,
        device=0 if device == "cuda" else -1,
    )

    # Select camera source at runtime to support both Pi CSI and USB webcams.
    picam = None
    cap = None

    if args.camera == "pi":
        try:
            from picamera2 import Picamera2  # Lazy import so USB mode works without the dependency
        except ImportError as exc:
            raise RuntimeError("picamera2 is required for --camera pi") from exc

        picam = Picamera2()
        picam.configure(picam.create_preview_configuration(main={"size": size, "format": "RGB888"}))
        picam.start()

        picam.set_controls({
            "AfMode": 2,
            "AfTrigger": 0,
            "AeEnable": True,
            "AwbEnable": True,
            "AnalogueGain": args.analogue_gain,
            "ExposureValue": args.exposure_comp,
        })

        def read_frame() -> Optional[np.ndarray]:
            try:
                frame_rgb = picam.capture_array()
                return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                return None

    else:
        cap = cv2.VideoCapture(args.video_device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, max(1.0, 1.0 / max(args.frame_interval, 0.001)))
        
        # Enable auto exposure and auto white balance for USB camera
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Auto exposure enabled
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)      # Autofocus enabled
        # Exposure compensation: higher values = brighter (typical range -4 to 4)
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure_comp)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video device {args.video_device}")

        def read_frame() -> Optional[np.ndarray]:
            ok, frame = cap.read()
            return frame if ok else None

    blue_kw = args.bluebird_keyword.strip().lower()

    prev_gray: Optional[np.ndarray] = None
    last_classify = 0.0
    last_preds: Optional[List[Dict[str, Any]]] = None

    last_motion = 0.0
    motion_now = False
    motion_start_time = 0.0  # When motion was first detected
    
    last_category = ""
    last_spoken = 0.0

    too_dark = False
    brightness = 0.0

    try:
        while True:
            frame = read_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            # Read flip state from shared status
            with shared.lock:
                should_flip = shared.status.flip
            
            bgr = cv2.flip(frame, args.flip_code) if should_flip else frame

            now = time.time()

            # Darkness check first
            brightness = brightness_mean_bgr(bgr, small_size_brightness)
            with shared.lock:
                motion_threshold = shared.status.motion_threshold
                dark_threshold = shared.status.dark_threshold

            too_dark = brightness < dark_threshold

            if too_dark:
                # Stop processing when too dark
                prev_gray = None
                motion_now = False
                motion_start_time = 0.0
                last_motion = 0.0

                # Optional: clear last preds while dark
                if args.clear_on_dark:
                    last_preds = None
                    last_category = "Too dark"

            else:
                # Motion detection
                ms, prev_gray = motion_score_bgr(bgr, prev_gray, small_size_motion, args.motion_diff_threshold)
                last_motion = ms
                motion_now = (ms >= motion_threshold)
                
                # Track motion window
                if motion_now and motion_start_time == 0.0:
                    # Motion just started
                    motion_start_time = now
                
                # Classify during motion window (motion detected OR within 30s of last motion)
                in_motion_window = motion_now or (motion_start_time > 0.0 and (now - motion_start_time) < 30.0)
                
                if in_motion_window and (now - last_classify >= args.cooldown_classify):
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    preds = clf(Image.fromarray(rgb), top_k=args.topk)
                    last_preds = preds
                    last_classify = now

                    top1_label = str(preds[0]["label"])
                    top1_score = float(preds[0]["score"])
                    category = categorize(preds, blue_kw, args.not_bird_threshold)

                    with shared.lock:
                        is_recording = shared.status.recording

                    if is_recording and category and category != last_category:
                        save_classified_frame(bgr.copy(), category, now)
                    
                    # Speak only when the category changes (with cooldown)
                    if category != last_category and (now - last_spoken >= args.tts_cooldown):
                        tts.say(category)
                        last_spoken = now

                    last_category = category
                
                # Motion window expired
                if motion_start_time > 0.0 and (now - motion_start_time) >= 30.0:
                    motion_start_time = 0.0

            # Overlay
            cv2.putText(
                bgr,
                f"{'Too Dark' if too_dark else ''}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                bgr,
                f"Motion {last_motion:.3f} {'yes' if motion_now else 'no'}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                bgr,
                f"Category {last_category}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if (not too_dark) and last_preds:
                y = 115
                for p in last_preds:
                    cv2.putText(
                        bgr,
                        f"{p['label']} {float(p['score']):.2f}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    y += 20

            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])
            if ok:
                with shared.lock:
                    shared.jpeg = buf.tobytes()
                    shared.status.running = True
                    shared.status.ts = now

                    shared.status.too_dark = bool(too_dark)
                    shared.status.brightness = float(brightness)

                    shared.status.motion_score = float(last_motion)
                    shared.status.motion = bool(motion_now)
                    shared.status.last_classify_ts = float(last_classify)

                    shared.status.category = str(last_category)
                    shared.status.last_spoken_ts = float(last_spoken)

                    if last_preds and (not too_dark):
                        shared.status.top1_label = str(last_preds[0]["label"])
                        shared.status.top1_score = float(last_preds[0]["score"])
                        shared.status.topk = [{"label": str(p["label"]), "score": float(p["score"])} for p in last_preds]
                    else:
                        shared.status.top1_label = ""
                        shared.status.top1_score = 0.0
                        shared.status.topk = None

            time.sleep(args.frame_interval)

    finally:
        try:
            if picam is not None:
                picam.stop()
        except Exception:
            pass

        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


# -------------------------------------------------
# Main
# -------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Local folder of the downloaded HF model")
    ap.add_argument("--piper-model", required=True, help="Path to Piper voice onnx file")

    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)

    ap.add_argument("--camera", choices=["pi", "usb"], default="pi", help="Camera source: pi CSI or USB webcam")
    ap.add_argument("--video-device", type=int, default=0, help="USB camera device index when using --camera usb")

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--jpeg-quality", type=int, default=80)

    ap.add_argument("--frame-interval", type=float, default=0.03)

    ap.add_argument("--motion-w", type=int, default=192)
    ap.add_argument("--motion-h", type=int, default=144)
    ap.add_argument("--motion-threshold", type=float, default=0.03)
    ap.add_argument("--motion-diff-threshold", type=int, default=30)

    # Darkness control
    ap.add_argument("--dark-threshold", type=float, default=50.0, help="Mean gray below this is too dark")
    ap.add_argument("--brightness-w", type=int, default=160)
    ap.add_argument("--brightness-h", type=int, default=120)
    ap.add_argument("--clear-on-dark", action="store_true", help="Clear predictions and set category to Too dark while dark")

    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cooldown-classify", type=float, default=1.0)

    ap.add_argument("--bluebird-keyword", default="bluebird")
    ap.add_argument("--not-bird-threshold", type=float, default=0.75)

    ap.add_argument("--flip-code", type=int, default=-1, help="OpenCV flip code: -1 both, 0 vertical, 1 horizontal")

    ap.add_argument("--exposure-comp", type=float, default=0.0, help="Exposure compensation (-4.0 to 4.0, higher=brighter)")
    ap.add_argument("--analogue-gain", type=float, default=1.0, help="Analogue gain for Pi camera (1.0 to 8.0)")

    ap.add_argument("--tts-cooldown", type=float, default=1.0)
    ap.add_argument("--speak-mode", choices=["change", "repeat"], default="change")
    ap.add_argument("--repeat-interval", type=float, default=5.0)

    args = ap.parse_args()

    shared = Shared()
    shared.status.motion_threshold = args.motion_threshold
    shared.status.dark_threshold = args.dark_threshold
    shared.status.flip = args.flip_code != 0  # Initialize flip based on args
    shared.status.exposure_comp = args.exposure_comp
    shared.status.analogue_gain = args.analogue_gain
    
    # Initialize TTS
    tts_phrases = ["Bluebird", "Not bluebird", "Not a bird"]
    tts = CachedPiperTTS(model_path=args.piper_model, phrases=tts_phrases)
    
    app = make_app(shared, tts)

    t = threading.Thread(target=camera_worker, args=(args, shared, tts), daemon=True)
    t.start()
    
    # Start power monitoring thread
    power_thread = threading.Thread(
        target=power_monitor_worker,
        args=(shared, 50.0, 30),  # 50Wh battery, 30 sample average
        daemon=True
    )
    power_thread.start()

    print("Server running")
    print(f"http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
