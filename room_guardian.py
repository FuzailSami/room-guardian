#!/usr/bin/env python3
"""
Room guardian: a LangGraph agent that watches the room and alerts your phone.

Always on (cheap): camera at ~2 fps, OpenCV motion detection, face check on motion.
On an event, a LangGraph graph runs:

  gate (snooze + YOLO person/animal check)
    +- nothing important ---------------------------------------------> log
    +- important -> quick_alert (photo to phone in ~1 s) -> describe (Moondream)
                                                                -> notify (details follow-up) -> log
  If YOLO isn't available, Moondream + rules decide instead:
    describe -> assess -+- unsure -> recapture -> describe (max 2x)
                        +- alert  -> notify -> log
                        +- ignore ----------> log

Phone (ntfy app, no account): alerts arrive with a photo and two buttons, "Take another"
and "Snooze 1h". You can also send text to the command topic:
  snapshot | snooze <minutes> | resume | status | any question ("is my laptop on the desk?")

Every event is also published over MQTT to fuzail-pi/guardian/event.

Run: python3 room_guardian.py
"""

import base64
import json
import os
import queue
import re
import secrets
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import TypedDict

import cv2
import numpy as np
import requests
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from tinydb import TinyDB

# --- Config ---
BASE_DIR = os.path.expanduser("~/guardian")
EVENTS_DIR = os.path.join(BASE_DIR, "events")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DB_PATH = os.path.join(BASE_DIR, "events.json")
STATE_PATH = os.path.join(BASE_DIR, "state.json")

# Presence: your phone sends "away" / "home" to the command topic (iOS Shortcuts
# location automation). While you're home, motion is ignored; snapshots and questions still work.
PRESENCE_ENABLED = True

FRAME_SIZE = (1280, 960)
CHECK_FPS = 4
MOTION_THRESHOLD = 0.01  # fraction of the frame that must change
MOTION_FRAMES = 2  # consecutive motion frames needed (filters noise)
EVENT_COOLDOWN = 20  # seconds between automatic events
MIN_ALERT_GAP = 20  # seconds between phone alerts
MAX_RECAPTURES = 2
EVENT_QUEUE_SIZE = 3  # events waiting for AI while the camera keeps watching

# Sharper photos: short-exposure camera mode + a burst, keeping the best frame
SHORT_EXPOSURE = True
BURST_FRAMES = 3
BURST_INTERVAL = 0.25  # seconds between burst frames

# Person detector: YOLO11n as ONNX, run by OpenCV (no PyTorch). Confirms people
# even when Moondream's description misses them. Skipped if the file is missing.
PERSON_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolo11n.onnx")
PERSON_CONF = 0.40
PERSON_INPUT = 640  # model input size (smaller = faster, misses distant people)
MAX_IMAGES = 500  # oldest event photos are deleted beyond this

# AWS backup: photos to S3; event records go over MQTT -> IoT rule -> DynamoDB.
# Both are deleted after RETENTION_DAYS (S3 lifecycle rule + DynamoDB TTL on expires_at).
UPLOAD_TO_S3 = True
S3_BUCKET = "fuzail-ras-pi-data"
S3_REGION = "us-east-1"
S3_PREFIX = "guardian/events"
RETENTION_DAYS = 30

# 1. Fast alert: send the photo immediately, the description follows
FAST_ALERT = True

# 2. YOLO person/animal detector decides alerts (falls back to Moondream if unavailable)
USE_YOLO = False
YOLO_MODEL_PATH = os.path.expanduser("~/streetgame/models/yolo26n_ncnn_model")
YOLO_CONF = 0.45
YOLO_CLASSES = {"person", "cat", "dog", "bird"}

# 3. Vision servers, tried in order. Put your laptop first to offload Moondream, e.g.
#    ["http://100.x.y.z:11434", "http://localhost:11434"]
VISION_SERVERS = ["http://localhost:11434"]
VISION_MODEL = "moondream"
VISION_THREADS = 2  # CPU threads on the Pi (4 = faster, needs the 27W supply)
VISION_SIZE = (640, 480)  # image size sent to Moondream (smaller = faster)
VISION_MAX_TOKENS = 45  # cap the description length
VISION_KEEP_ALIVE = "30m"  # keep Moondream loaded between events

AGENT_MODEL = "llama3.2:1b"
USE_LLM_ASSESS = False  # True = Llama decides alerts when YOLO is off (slower)
PERSON_PATTERN = re.compile(
    r"\b(person|people|man|men|woman|women|boy|girl|child|children|kid|kids|someone|somebody|"
    r"human|figure|silhouette|hand|hands|arm|legs?|face)\b")
SUSPICIOUS_PATTERN = re.compile(
    r"\b(mask|masked|balaclava|hood|hooded|knife|gun|weapon|crowbar|broken|shattered|"
    r"smoke|fire|flames?|intruder|burglar|break-in)\b")
NEGATED_PATTERN = re.compile(r"\b(no|without)\s+(people|person|one|humans?)\b|\bnobody\b|\bno one\b")
# Stock openings Moondream uses, removed to make descriptions read naturally
FILLER_PATTERNS = [
    re.compile(r"^(the|this) (image|picture|photo|scene) (shows|features|depicts|displays|captures|"
               r"contains|is of|presents)\s+", re.I),
    re.compile(r"^in (the|this) (image|picture|photo),?\s+(there is |there are )?", re.I),
    re.compile(r"^(a|an) (close-up|closeup|wide|blurry|dark|low-angle) (view|shot|image|photo) of\s+", re.I),
]
NTFY_SERVER = "https://ntfy.sh"

# Offline alert: pings a healthchecks.io URL (set "healthcheck_url" in ~/guardian/config.json).
# If the pings stop (Pi off, crashed, offline, guardian stopped), healthchecks.io alerts you.
HEARTBEAT_SECONDS = 20

# Questions about the past ("when did someone last come in?") are answered from the event log
HISTORY_PATTERN = re.compile(
    r"\b(when|earlier|yesterday|today|tonight|this (morning|afternoon|evening|week)|while i was|"
    r"since i|how many|did (anyone|someone|anybody|somebody)|has (anyone|someone|anybody|somebody)|"
    r"anything happen|what happened|history|recently|last (time|\d+)|past \d+)\b")
HISTORY_MAX_EVENTS = 15  # events shown to the LLM
DESCRIBE_PROMPT = "Describe this image."  # Moondream garbles longer prompts
DESCRIPTION_SENTENCES = 1  # keep only the first sentences of the description

CASCADE_PATHS = [
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    os.path.join(getattr(getattr(cv2, "data", None), "haarcascades", ""),
                 "haarcascade_frontalface_default.xml"),
]

llm = ChatOllama(model=AGENT_MODEL, temperature=0, num_thread=2, format="json")
llm_text = ChatOllama(model=AGENT_MODEL, temperature=0, num_thread=2, num_predict=80)

# Runtime state shared by the graph and the command handler
runtime = {
    "snooze_until": 0.0,
    "last_alert": 0.0,
    "recent": deque(maxlen=8),  # (local time, description, alerted)
    "mode": "away",  # away = armed, home = motion ignored
    "away_since": time.time(),
    "last_away": None,  # [start, end] of the most recent completed away period
}
watcher = None
detector = None
person_detector = None
db_lock = threading.Lock()  # TinyDB is shared by the main loop and the AI worker
jobs = queue.Queue(maxsize=EVENT_QUEUE_SIZE)
topics = {}


# --- Setup ---
def load_topics():
    """Random ntfy topic names act as passwords. Generated once and kept in config.json."""
    os.makedirs(EVENTS_DIR, exist_ok=True)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    name = f"fuzail-guardian-{secrets.token_hex(8)}"
    cfg = {"alert_topic": name, "cmd_topic": f"{name}-cmd"}
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    os.chmod(CONFIG_PATH, 0o600)
    return cfg


def load_detector():
    if not USE_YOLO:
        return None
    try:
        from ultralytics import YOLO

        model = YOLO(YOLO_MODEL_PATH, task="detect")
        model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=640, verbose=False)  # warm up
        print("YOLO loaded: alerts decided by person/animal detection", flush=True)
        return model
    except Exception as e:
        print(f"YOLO unavailable ({e}); Moondream will decide alerts", flush=True)
        return None


def detect_objects(frame):
    """Return {label: best confidence} for YOLO_CLASSES found in the frame."""
    result = detector.predict(frame, imgsz=640, conf=YOLO_CONF, verbose=False)[0]
    found = {}
    for cls, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        label = result.names[int(cls)]
        if label in YOLO_CLASSES:
            found[label] = max(found.get(label, 0.0), round(float(conf), 2))
    return found


# --- Camera + motion ---
class Watcher:
    def __init__(self, camera=None):
        if camera is None:
            from picamera2 import Picamera2

            camera = Picamera2()
            # RGB888 in Picamera2 gives BGR order, which is what OpenCV expects
            camera.configure(camera.create_video_configuration(
                main={"size": FRAME_SIZE, "format": "RGB888"}))
            if SHORT_EXPOSURE:
                try:  # shorter exposure + more gain = less motion blur
                    from libcamera import controls

                    camera.set_controls({"AeExposureMode": controls.AeExposureModeEnum.Short})
                except Exception as e:
                    print(f"Short exposure not available ({e})", flush=True)
            camera.start()
            time.sleep(2)  # let exposure settle
        self.cam = camera
        self.avg = None
        self.motion_count = 0
        cascade = next((p for p in CASCADE_PATHS if os.path.exists(p)), None)
        # OpenCV 5 removed CascadeClassifier; YOLO still detects people without it
        self.faces = cv2.CascadeClassifier(cascade) if cascade and hasattr(cv2, "CascadeClassifier") else None

    def grab(self):
        return self.cam.capture_array()

    def burst(self, first_frame):
        """first_frame plus BURST_FRAMES-1 more, BURST_INTERVAL apart."""
        frames = [first_frame]
        for _ in range(BURST_FRAMES - 1):
            time.sleep(BURST_INTERVAL)
            frames.append(self.grab())
        return frames

    def reset(self):
        """Make the current scene the new background (e.g. after an event)."""
        self.avg = None
        self.motion_count = 0

    def count_faces(self, frame):
        if self.faces is None:
            return 0
        gray = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2GRAY)
        return len(self.faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)))

    def check(self):
        """Return an event dict when motion is detected, else None."""
        frame = self.grab()
        small = cv2.resize(frame, (320, 240))
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        if self.avg is None:
            self.avg = gray.astype("float")
            return None

        delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))
        cv2.accumulateWeighted(gray, self.avg, 0.05)
        mask = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion = cv2.countNonZero(mask) / mask.size

        self.motion_count = self.motion_count + 1 if motion >= MOTION_THRESHOLD else 0
        if self.motion_count < MOTION_FRAMES:
            return None
        faces = self.count_faces(frame)
        return {"trigger": "face" if faces else "motion", "faces": faces,
                "motion": round(motion, 3), "frame": frame}


class PersonDetector:
    """YOLO11n (ONNX) run by OpenCV's DNN module, so no PyTorch is needed."""

    def __init__(self, path):
        self.net = cv2.dnn.readNetFromONNX(path)

    def detect(self, frame):
        """Return (number of people, best confidence)."""
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (PERSON_INPUT, PERSON_INPUT), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()[0].T  # rows: cx, cy, w, h, then 80 class scores (0 = person)
        scores = out[:, 4]
        keep = scores >= PERSON_CONF
        if not keep.any():
            return 0, 0.0
        rects = [[float(cx - w / 2), float(cy - h / 2), float(w), float(h)] for cx, cy, w, h in out[keep, :4]]
        people = cv2.dnn.NMSBoxes(rects, scores[keep].tolist(), PERSON_CONF, 0.45)
        return len(people), round(float(scores[keep].max()), 2)


def load_person_detector():
    if not os.path.exists(PERSON_MODEL_PATH):
        print(f"Person detector off ({PERSON_MODEL_PATH} not found); Moondream decides", flush=True)
        return None
    try:
        model = PersonDetector(PERSON_MODEL_PATH)
        model.detect(np.zeros((480, 640, 3), np.uint8))  # warm up
        print("Person detector loaded (YOLO11n ONNX)", flush=True)
        return model
    except Exception as e:
        print(f"Person detector failed to load ({e}); Moondream decides", flush=True)
        return None


def sharpness(frame):
    gray = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def pick_best(frames):
    """Choose the frame that shows a person most clearly, else the sharpest one.
    Returns (frame, labels) where labels is {"person": confidence} or {}."""
    scored = []
    for frame in frames:
        people, conf = person_detector.detect(frame) if person_detector else (0, 0.0)
        scored.append((conf, sharpness(frame), frame, people))
    conf, _, frame, people = max(scored, key=lambda s: (s[0], s[1]))
    return frame, ({"person": conf} if people else {})


def save_frame(frame, name):
    path = os.path.join(EVENTS_DIR, f"{name}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


# --- Vision + notifications ---
def first_sentences(text, n=DESCRIPTION_SENTENCES):
    """Keep the first n complete sentences and remove stock openings
    ("The image shows a close-up view of ...") so it reads naturally."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    complete = [p for p in parts if p.endswith((".", "!", "?"))]
    text = " ".join(complete[:n]) if complete else text.strip()
    for pattern in FILLER_PATTERNS:
        text = pattern.sub("", text, count=1)
    return text[:1].upper() + text[1:]


def describe_image(path, prompt):
    small = cv2.resize(cv2.imread(path), VISION_SIZE)
    image_b64 = base64.b64encode(cv2.imencode(".jpg", small)[1].tobytes()).decode()
    last_error = None
    for server in VISION_SERVERS:
        options = {"num_predict": VISION_MAX_TOKENS}
        if "localhost" in server or "127.0.0.1" in server:
            options["num_thread"] = VISION_THREADS
        try:
            resp = requests.post(
                f"{server}/api/generate",
                json={"model": VISION_MODEL, "prompt": prompt, "images": [image_b64],
                      "stream": False, "keep_alive": VISION_KEEP_ALIVE, "options": options},
                timeout=(3, 600),  # 3 s to connect, so an offline laptop is skipped quickly
            )
            resp.raise_for_status()
            return first_sentences(resp.json()["response"])
        except requests.RequestException as e:
            last_error = e
    raise RuntimeError(f"no vision server worked ({last_error})")


def _header(text, limit=900):
    # HTTP headers must be ASCII and single-line
    return " ".join(text.split()).encode("ascii", "ignore").decode()[:limit]


def send_ntfy(title, message, image_path=None, tags="", priority=3, buttons=True):
    url = f"{NTFY_SERVER}/{topics['alert_topic']}"
    cmd_url = f"{NTFY_SERVER}/{topics['cmd_topic']}"
    headers = {"Title": _header(title, 200), "Priority": str(priority)}
    if tags:
        headers["Tags"] = tags
    if buttons:
        headers["Actions"] = (
            f"http, Take another, {cmd_url}, body=snapshot, clear=true; "
            f"http, Snooze 1h, {cmd_url}, body=snooze 60, clear=true"
        )
    try:
        if image_path:
            headers["Message"] = _header(message)
            headers["Filename"] = os.path.basename(image_path)
            with open(image_path, "rb") as f:
                requests.put(url, data=f, headers=headers, timeout=30).raise_for_status()
        else:
            requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=30).raise_for_status()
    except requests.RequestException as e:
        print(f"ntfy send failed: {e}", flush=True)


def upload_photo(path, key):
    """Upload an event photo to S3 in the background so alerts aren't delayed."""
    def _upload():
        try:
            import boto3

            boto3.client("s3", region_name=S3_REGION).upload_file(
                path, S3_BUCKET, key, ExtraArgs={"ContentType": "image/jpeg"})
        except Exception as e:  # no internet / no credentials: photo stays local only
            print(f"S3 upload failed for {key}: {e}", flush=True)

    threading.Thread(target=_upload, daemon=True).start()


def publish_mqtt(event):
    try:
        from mqtt_publish import publish  # same folder as this script

        publish("guardian/event", event)
    except Exception as e:  # broker down shouldn't stop the guardian
        print(f"MQTT publish failed: {e}", flush=True)


def listen_commands(out_queue):
    """Stream messages sent to the command topic (from the phone) into out_queue."""
    url = f"{NTFY_SERVER}/{topics['cmd_topic']}/json"
    while True:
        try:
            with requests.get(url, stream=True, timeout=(10, 90)) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    msg = json.loads(line)
                    if msg.get("event") == "message" and msg.get("message", "").strip():
                        out_queue.put(msg["message"].strip())
        except Exception as e:
            print(f"Command stream error: {e}; reconnecting", flush=True)
            time.sleep(5)


# --- LangGraph ---
class GuardianState(TypedDict, total=False):
    event_id: str
    mode: str  # auto | snapshot | question
    trigger: str  # motion | face | manual
    faces: int
    motion: float
    question: str
    image_path: str
    labels: dict  # YOLO detections {label: confidence}
    decided: bool  # alert decision already made (gate)
    quick_sent: bool  # photo alert already sent (fast alert)
    description: str
    alert: bool
    unsure: bool
    reason: str
    attempts: int


def _clock():
    return datetime.now().strftime("%H:%M")


def _what(state):
    labels = state.get("labels") or {}
    if labels:
        return " and ".join(labels).capitalize()
    return "Face" if state["trigger"] == "face" else "Motion"


def _details(state):
    """Description, plus a note when YOLO saw a person the description doesn't mention."""
    text = state.get("description", "")
    conf = (state.get("labels") or {}).get("person")
    if conf and not PERSON_PATTERN.search(text.lower()):
        text += f" (Person detector: {conf:.0%} sure someone is in the photo.)"
    return text


def _verdict(state):
    """(title, ntfy tags, priority) for the follow-up notification."""
    reason = state.get("reason", "")
    if not state.get("alert"):
        return "Nothing important", "white_check_mark", 2
    if reason.startswith("Looks suspicious"):
        return "Something suspicious", "warning,rotating_light", 5
    return "Person spotted", "rotating_light", 4


def gate(state: GuardianState):
    """Cheap checks before any AI: snooze, alert gap, YOLO person/animal detection."""
    if state["mode"] != "auto":
        return {}
    now = time.time()
    if now < runtime["snooze_until"]:
        return {"decided": True, "alert": False, "reason": "Snoozed"}
    if now - runtime["last_alert"] < MIN_ALERT_GAP:
        return {"decided": True, "alert": False, "reason": "Alert already sent in the last minute"}
    if detector is None:
        return {"decided": False}

    try:
        labels = detect_objects(cv2.imread(state["image_path"]))
    except Exception as e:
        print(f"YOLO failed ({e}); Moondream will decide", flush=True)
        return {"decided": False}
    if labels:
        reason = ", ".join(f"{k} {v:.0%}" for k, v in labels.items())
        return {"decided": True, "alert": True, "labels": labels,
                "reason": "Person spotted" if "person" in labels else f"YOLO saw {reason}"}
    if state["faces"] > 0:
        return {"decided": True, "alert": True, "labels": {}, "reason": "Person spotted"}
    return {"decided": True, "alert": False, "labels": {}, "reason": "No person or animal detected"}


def after_gate(state: GuardianState):
    if state["mode"] != "auto":
        return "describe"
    if state.get("decided") and not state.get("alert"):
        return "log"  # nothing important: skip Moondream entirely
    return "quick_alert" if FAST_ALERT else "describe"


def quick_alert(state: GuardianState):
    important = bool(state.get("labels") or state["faces"])
    if important:  # already know it's a person (face or YOLO): alert loudly now
        send_ntfy(f"{_what(state)} detected ({_clock()})", "Checking what's happening...",
                  state["image_path"], "rotating_light", 4)
    else:  # plain motion: send the photo quietly, the follow-up says if it matters
        send_ntfy(f"Motion ({_clock()})", "Checking what it is...", state["image_path"],
                  "eyes", 2, buttons=False)
    runtime["last_alert"] = time.time()
    return {"quick_sent": True}


def describe(state: GuardianState):
    try:
        text = describe_image(state["image_path"], state.get("question") or DESCRIBE_PROMPT)
    except Exception as e:
        text = f"(vision failed: {e})"
    return {"description": text}


def after_describe(state: GuardianState):
    if state["mode"] != "auto" or state.get("decided"):
        return "notify"
    return "assess"


def assess(state: GuardianState):
    """Used only when YOLO isn't available: decide from Moondream's description."""
    if not USE_LLM_ASSESS:
        text = state["description"].lower()
        suspicious = SUSPICIOUS_PATTERN.search(text)
        person = None if NEGATED_PATTERN.search(text) else PERSON_PATTERN.search(text)
        if suspicious:
            return {"alert": True, "unsure": False, "reason": f"Looks suspicious ({suspicious.group(0)})"}
        if state["faces"] > 0 or person or (state.get("labels") or {}).get("person"):
            return {"alert": True, "unsure": False, "reason": "Person spotted"}
        return {"alert": False, "unsure": False, "reason": "Nothing important"}

    recent = "\n".join(
        f"- {t}: {d} ({'alerted' if a else 'not alerted'})" for t, d, a in runtime["recent"]
    ) or "none"
    prompt = (
        "You are a home security assistant deciding whether to alert the owner's phone.\n"
        f"Current time: {datetime.now().strftime('%A %H:%M')}\n"
        f"Trigger: {state['trigger']} (faces detected: {state['faces']}, "
        f"motion: {state['motion'] * 100:.0f}% of the frame)\n"
        f"What the camera sees: {state['description']}\n"
        f"Recent events (newest last):\n{recent}\n\n"
        "Alert only for a person passing by or something suspicious "
        "(a mask, a weapon, forced entry, smoke, fire). Do NOT alert for animals, lighting changes, "
        "shadows, an empty scene, or the same thing already alerted in the last few minutes. "
        'Set "unsure" to true only if the description is too vague to decide.\n'
        'Reply with JSON: {"alert": true or false, "unsure": true or false, '
        '"reason": "<one short sentence>"}'
    )
    try:
        data = json.loads(llm.invoke(prompt).content)
        return {
            "alert": bool(data.get("alert")),
            "unsure": bool(data.get("unsure")),
            "reason": str(data.get("reason", "")).strip() or "No reason given",
        }
    except Exception as e:
        # LLM unavailable or bad JSON: fall back to a simple rule
        return {"alert": state["faces"] > 0, "unsure": False, "reason": f"Rule fallback ({e})"}


def after_assess(state: GuardianState):
    if state.get("unsure") and state.get("attempts", 0) < MAX_RECAPTURES:
        return "recapture"
    if state.get("quick_sent") or state.get("alert"):
        return "notify"
    return "log"


def recapture(state: GuardianState):
    time.sleep(2)
    attempts = state.get("attempts", 0) + 1
    path = save_frame(watcher.grab(), f"{state['event_id']}_{attempts}")
    return {"image_path": path, "attempts": attempts}


def notify(state: GuardianState):
    mode = state["mode"]
    if mode == "question":
        send_ntfy(f"Answer ({_clock()})", state["description"], state["image_path"], "speech_balloon", 3)
    elif mode == "snapshot":
        send_ntfy(f"Snapshot ({_clock()})", state["description"], state["image_path"], "camera", 3)
    elif state.get("quick_sent"):
        # Photo already sent: follow up with what it is. Loud only if it matters.
        title, tags, priority = _verdict(state)
        if (state.get("labels") or state["faces"]) and priority < 5:
            priority = 3  # the photo alert was already loud; don't buzz twice
        send_ntfy(f"{title} ({_clock()})", _details(state), tags=tags, priority=priority,
                  buttons=state.get("alert", False))
    else:
        title, tags, priority = _verdict(state)
        send_ntfy(f"{title} ({_clock()})", _details(state), state["image_path"], tags, priority,
                  buttons=state.get("alert", False))
        runtime["last_alert"] = time.time()
    return {"alert": bool(state.get("alert")) or mode != "auto"}


def log(state: GuardianState):
    event = {
        "id": state["event_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": state["mode"],
        "trigger": state["trigger"],
        "faces": state.get("faces", 0),
        "motion": state.get("motion", 0.0),
        "labels": state.get("labels") or {},
        "description": state.get("description", ""),
        "alert": bool(state.get("alert")),
        "reason": state.get("reason", ""),
        "image": os.path.basename(state["image_path"]),
        "s3_key": f"{S3_PREFIX}/{os.path.basename(state['image_path'])}" if UPLOAD_TO_S3 else "",
        "expires_at": int(time.time()) + RETENTION_DAYS * 86400,  # DynamoDB TTL (epoch seconds)
    }
    with db_lock:
        TinyDB(DB_PATH).insert(event)
    if UPLOAD_TO_S3:
        upload_photo(state["image_path"], event["s3_key"])
    publish_mqtt(event)
    runtime["recent"].append((_clock(), event["description"][:150] or event["reason"], event["alert"]))

    photos = sorted(f for f in os.listdir(EVENTS_DIR) if f.endswith(".jpg"))
    for old in photos[:-MAX_IMAGES]:
        os.remove(os.path.join(EVENTS_DIR, old))
    print(f"[{event['id']}] {event['trigger']}: {(event['description'] or '-')[:80]} "
          f"-> alert={event['alert']} ({event['reason']})", flush=True)
    return {}


def build_graph():
    g = StateGraph(GuardianState)
    for name, fn in [("gate", gate), ("quick_alert", quick_alert), ("describe", describe),
                     ("assess", assess), ("recapture", recapture), ("notify", notify), ("log", log)]:
        g.add_node(name, fn)
    g.add_edge(START, "gate")
    g.add_conditional_edges("gate", after_gate, ["log", "quick_alert", "describe"])
    g.add_edge("quick_alert", "describe")
    g.add_conditional_edges("describe", after_describe, ["notify", "assess"])
    g.add_conditional_edges("assess", after_assess, ["recapture", "notify", "log"])
    g.add_edge("recapture", "describe")
    g.add_edge("notify", "log")
    g.add_edge("log", END)
    return g.compile()


graph = build_graph()


def run_event(mode, trigger, frame=None, faces=0, motion=0.0, question=None, labels=None):
    now = datetime.now()
    event_id = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    if frame is None:
        frame = watcher.grab()
    graph.invoke({
        "event_id": event_id, "mode": mode, "trigger": trigger, "faces": faces,
        "motion": motion, "question": question or "", "attempts": 0, "labels": labels or {},
        "image_path": save_frame(frame, event_id),
    })


def db_all():
    with db_lock:
        return TinyDB(DB_PATH).all()


def enqueue(**event):
    """Hand an event to the AI worker so the camera keeps watching meanwhile."""
    try:
        jobs.put_nowait(event)
    except queue.Full:
        print("AI busy: event dropped (queue full)", flush=True)


def ai_worker():
    while True:
        event = jobs.get()
        try:
            if event.pop("kind", None) == "history":
                answer_history(event["question"])
            else:
                run_event(**event)
        except Exception as e:
            print(f"Event failed: {e}", flush=True)


# --- Presence (away / home) ---
def load_state():
    """Restore away/home mode after a restart."""
    try:
        with open(STATE_PATH) as f:
            saved = json.load(f)
        runtime["mode"] = saved.get("mode", "away")
        runtime["away_since"] = saved.get("away_since", time.time())
        runtime["last_away"] = saved.get("last_away")
    except (OSError, ValueError):
        pass


def save_state():
    with open(STATE_PATH, "w") as f:
        json.dump({"mode": runtime["mode"], "away_since": runtime["away_since"],
                   "last_away": runtime["last_away"]}, f)


def _duration(seconds):
    hours, minutes = int(seconds // 3600), int(seconds % 3600 // 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def away_summary(since):
    """One-paragraph summary of automatic events since `since` (epoch seconds)."""
    events = [e for e in db_all()
              if e.get("mode") == "auto" and datetime.fromisoformat(e["timestamp"]).timestamp() >= since]
    people = sum(1 for e in events if e.get("reason") == "Person spotted")
    suspicious = sum(1 for e in events if str(e.get("reason", "")).startswith("Looks suspicious"))
    away_for = _duration(time.time() - since)
    if not events:
        return f"All quiet while you were out ({away_for})."
    parts = [f"{len(events)} motion event{'s' * (len(events) != 1)}",
             f"{people} {'person' if people == 1 else 'people'} spotted",
             f"{suspicious} suspicious" if suspicious else "nothing suspicious"]
    alerts = [e for e in events if e.get("alert")]
    last = alerts[-1] if alerts else events[-1]
    when = datetime.fromisoformat(last["timestamp"]).astimezone().strftime("%H:%M")
    return (f"While you were out ({away_for}): {', '.join(parts)}. "
            f"Last: {(last.get('description') or last.get('reason', ''))[:120].rstrip('.')} ({when}).")


def set_away():
    if runtime["mode"] == "away":
        send_ntfy("Already armed", "Room guardian is already watching.", tags="shield", buttons=False)
        return
    runtime["mode"], runtime["away_since"] = "away", time.time()
    save_state()
    send_ntfy("Armed - you're away", "I'll alert you if a person passes by or anything looks suspicious.",
              tags="shield", buttons=False)


def set_home():
    if runtime["mode"] == "home":
        return
    since = runtime["away_since"]
    runtime["mode"] = "home"
    runtime["last_away"] = [since, time.time()]
    save_state()
    send_ntfy("Welcome back", away_summary(since), tags="house", buttons=False)


# --- Questions about the past ---
def is_history_question(text):
    return bool(HISTORY_PATTERN.search(text.lower()))


def history_window(question):
    """(start, end, label) in epoch seconds for the period a question is about."""
    q, now = question.lower(), time.time()
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    if m := re.search(r"\b(?:last|past) (\d+) (minute|min|hour|hr|day)s?\b", q):
        n, unit = int(m.group(1)), m.group(2)
        seconds = n * {"minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400}[unit]
        return now - seconds, now, f"the last {n} {unit if unit in ('minute', 'hour', 'day') else 'hour'}s"
    if re.search(r"\b(while i was|since i (left|was)|when i was (out|away|gone))", q):
        if runtime["mode"] == "away":
            return runtime["away_since"], now, "since you left"
        if runtime["last_away"]:
            start, end = runtime["last_away"]
            return start, end, "while you were out"
    if "yesterday" in q:
        return midnight - 86400, midnight, "yesterday"
    if "this week" in q:
        return now - 7 * 86400, now, "the last 7 days"
    if re.search(r"\b(today|tonight|this (morning|afternoon|evening))\b", q):
        return midnight, now, "today"
    return now - 86400, now, "the last 24 hours"


def answer_history(question):
    """Answer a question about past events: code finds the facts, Llama phrases them."""
    start, end, label = history_window(question)
    events = [e for e in db_all() if e.get("mode") == "auto"
              and start <= datetime.fromisoformat(e["timestamp"]).timestamp() <= end]
    people = [e for e in events if e.get("reason") == "Person spotted"]
    suspicious = [e for e in events if str(e.get("reason", "")).startswith("Looks suspicious")]

    def line(e):
        when = datetime.fromisoformat(e["timestamp"]).astimezone().strftime("%a %H:%M")
        return f"- {when}: {e.get('reason') or 'Motion'} - {(e.get('description') or '').rstrip('.')}"

    facts = (f"Period: {label}. Motion events: {len(events)}. People spotted: {len(people)}. "
             f"Suspicious: {len(suspicious)}.\nEvents (oldest first):\n"
             + ("\n".join(line(e) for e in events[-HISTORY_MAX_EVENTS:]) or "- none"))
    # Fallback answer built from facts only (used if the LLM is unavailable)
    if not events:
        answer = f"Nothing was recorded {'in ' if label.startswith('the ') else ''}{label}."
    else:
        key = (suspicious or people or events)[-1]
        answer = (f"{label.capitalize()}: {len(events)} event(s), {len(people)} with people, "
                  f"{len(suspicious)} suspicious. Most recent: {line(key)[2:]}.")
    if events:
        try:
            reply = llm_text.invoke(
                "You answer questions about a home security camera's event log. Use ONLY the facts "
                "below; never invent events or times. Answer in 1-2 short sentences.\n\n"
                f"{facts}\n\nQuestion: {question}\nAnswer:").content.strip()
            answer = reply or answer
        except Exception as e:
            print(f"History LLM failed ({e}); sending the plain summary", flush=True)

    photo = None
    if events:
        key = (suspicious or people or events)[-1]
        path = os.path.join(EVENTS_DIR, key.get("image", ""))
        photo = path if key.get("image") and os.path.exists(path) else None
    send_ntfy(f"Answer ({_clock()})", answer, photo, "speech_balloon", 3, buttons=False)
    print(f"History answer ({label}, {len(events)} events): {answer}", flush=True)


def heartbeat(url):
    """Ping healthchecks.io so it can alert you if the Pi or the guardian goes silent."""
    while True:
        try:
            requests.get(url, timeout=10)
        except requests.RequestException:
            pass  # offline: the missing pings are exactly what triggers the alert
        time.sleep(HEARTBEAT_SECONDS)


# --- Phone commands ---
def status_text():
    today = datetime.now().strftime("%Y%m%d")
    events = [e for e in db_all() if e["id"].startswith(today)]
    alerts = sum(1 for e in events if e["alert"])
    if time.time() < runtime["snooze_until"]:
        state = "Snoozed until " + datetime.fromtimestamp(runtime["snooze_until"]).strftime("%H:%M")
    elif PRESENCE_ENABLED and runtime["mode"] == "home":
        state = "Home (motion alerts off)"
    else:
        state = "Away (armed)"
    detection = "YOLO" if (detector or person_detector) else "Moondream"
    last = f"Last: {(events[-1]['description'] or events[-1]['reason'])[:120]}" if events else "No events yet today."
    return f"{state} ({detection} detection). Today: {len(events)} events, {alerts} alerts. {last}"


def handle_command(text):
    t = text.lower().strip()
    print(f"Command: {text}", flush=True)
    if t in ("snapshot", "photo", "pic", "take another"):
        frame, labels = pick_best(watcher.burst(watcher.grab()))
        enqueue(mode="snapshot", trigger="manual", frame=frame, labels=labels)
    elif t.startswith("snooze"):
        digits = re.findall(r"\d+", t)
        minutes = int(digits[0]) if digits else 60
        runtime["snooze_until"] = time.time() + minutes * 60
        send_ntfy("Snoozed", f"Alerts paused for {minutes} min. Send 'resume' to re-arm.",
                  tags="zzz", buttons=False)
    elif t in ("resume", "arm", "unsnooze"):
        runtime["snooze_until"] = 0.0
        send_ntfy("Armed", "Alerts are back on.", tags="shield", buttons=False)
    elif t in ("away", "leave", "leaving", "left"):
        set_away()
    elif t in ("home", "back", "arrived", "im home", "i'm home"):
        set_home()
    elif t == "status":
        send_ntfy("Status", status_text(), tags="bar_chart", buttons=False)
    elif is_history_question(text):
        enqueue(kind="history", question=text)
    else:
        enqueue(mode="question", trigger="manual", frame=watcher.grab(), question=text)


# --- Main loop ---
def main():
    global watcher, detector, person_detector, topics
    topics = load_topics()
    load_state()
    print(f"Mode: {runtime['mode']}")
    print("Subscribe to these in the ntfy app:")
    print(f"  Alerts:   {topics['alert_topic']}")
    print(f"  Commands: {topics['cmd_topic']}  (send: snapshot, snooze 30, resume, status, or a question)")

    detector = load_detector()
    person_detector = load_person_detector()
    watcher = Watcher()
    threading.Thread(target=ai_worker, daemon=True).start()
    if topics.get("healthcheck_url"):
        threading.Thread(target=heartbeat, args=(topics["healthcheck_url"],), daemon=True).start()
        print("Heartbeat on: you'll be alerted if the guardian goes offline", flush=True)
    else:
        print("Heartbeat off (no healthcheck_url in config.json)", flush=True)
    commands = queue.Queue()
    threading.Thread(target=listen_commands, args=(commands,), daemon=True).start()
    send_ntfy("Room guardian is on", "Watching the room. Send 'status' to the command topic anytime.",
              tags="shield", buttons=False)

    last_event = 0.0
    while True:
        try:
            while not commands.empty():
                handle_command(commands.get_nowait())
                watcher.reset()

            hit = watcher.check()
            if hit and PRESENCE_ENABLED and runtime["mode"] == "home":
                watcher.reset()  # you're home: ignore your own movement
                hit = None
            if hit and time.time() - last_event >= EVENT_COOLDOWN:
                last_event = time.time()
                frame, labels = pick_best(watcher.burst(hit["frame"]))
                seen = f", person {labels['person']:.0%}" if labels else ""
                print(f"Event: {hit['trigger']} {hit['motion'] * 100:.0f}% of frame{seen}, analyzing...", flush=True)
                enqueue(mode="auto", trigger=hit["trigger"], frame=frame, faces=hit["faces"],
                        motion=hit["motion"], labels=labels)
                watcher.reset()  # the current scene becomes the new background
        except Exception as e:
            print(f"Error: {e}", flush=True)
            time.sleep(5)
        time.sleep(1 / CHECK_FPS)


if __name__ == "__main__":
    main()
