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

FRAME_SIZE = (1280, 960)
CHECK_FPS = 2
MOTION_THRESHOLD = 0.02  # fraction of the frame that must change
MOTION_FRAMES = 2  # consecutive motion frames needed (filters noise)
EVENT_COOLDOWN = 60  # seconds between automatic events
MIN_ALERT_GAP = 60  # seconds between phone alerts
MAX_RECAPTURES = 2
MAX_IMAGES = 500  # oldest event photos are deleted beyond this

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
DESCRIBE_PROMPT = "Describe this image."  # Moondream garbles longer prompts
DESCRIPTION_SENTENCES = 1  # keep only the first sentences of the description

CASCADE_PATHS = [
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    os.path.join(getattr(getattr(cv2, "data", None), "haarcascades", ""),
                 "haarcascade_frontalface_default.xml"),
]

llm = ChatOllama(model=AGENT_MODEL, temperature=0, num_thread=2, format="json")

# Runtime state shared by the graph and the command handler
runtime = {
    "snooze_until": 0.0,
    "last_alert": 0.0,
    "recent": deque(maxlen=8),  # (local time, description, alerted)
}
watcher = None
detector = None
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
        if state["faces"] > 0 or person:
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
        send_ntfy(f"{title} ({_clock()})", state["description"], tags=tags, priority=priority,
                  buttons=state.get("alert", False))
    else:
        title, tags, priority = _verdict(state)
        send_ntfy(f"{title} ({_clock()})", state["description"], state["image_path"], tags, priority,
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
    }
    TinyDB(DB_PATH).insert(event)
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


def run_event(mode, trigger, frame=None, faces=0, motion=0.0, question=None):
    event_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if frame is None:
        frame = watcher.grab()
    graph.invoke({
        "event_id": event_id, "mode": mode, "trigger": trigger, "faces": faces,
        "motion": motion, "question": question or "", "attempts": 0,
        "image_path": save_frame(frame, event_id),
    })


# --- Phone commands ---
def status_text():
    today = datetime.now().strftime("%Y%m%d")
    events = [e for e in TinyDB(DB_PATH).all() if e["id"].startswith(today)]
    alerts = sum(1 for e in events if e["alert"])
    if time.time() < runtime["snooze_until"]:
        state = "Snoozed until " + datetime.fromtimestamp(runtime["snooze_until"]).strftime("%H:%M")
    else:
        state = "Armed"
    detection = "YOLO" if detector else "Moondream"
    last = f"Last: {(events[-1]['description'] or events[-1]['reason'])[:120]}" if events else "No events yet today."
    return f"{state} ({detection} detection). Today: {len(events)} events, {alerts} alerts. {last}"


def handle_command(text):
    t = text.lower().strip()
    print(f"Command: {text}", flush=True)
    if t in ("snapshot", "photo", "pic", "take another"):
        run_event("snapshot", "manual")
    elif t.startswith("snooze"):
        digits = re.findall(r"\d+", t)
        minutes = int(digits[0]) if digits else 60
        runtime["snooze_until"] = time.time() + minutes * 60
        send_ntfy("Snoozed", f"Alerts paused for {minutes} min. Send 'resume' to re-arm.",
                  tags="zzz", buttons=False)
    elif t in ("resume", "arm", "unsnooze"):
        runtime["snooze_until"] = 0.0
        send_ntfy("Armed", "Alerts are back on.", tags="shield", buttons=False)
    elif t == "status":
        send_ntfy("Status", status_text(), tags="bar_chart", buttons=False)
    else:
        run_event("question", "manual", question=text)


# --- Main loop ---
def main():
    global watcher, detector, topics
    topics = load_topics()
    print("Subscribe to these in the ntfy app:")
    print(f"  Alerts:   {topics['alert_topic']}")
    print(f"  Commands: {topics['cmd_topic']}  (send: snapshot, snooze 30, resume, status, or a question)")

    detector = load_detector()
    watcher = Watcher()
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
            if hit and time.time() - last_event >= EVENT_COOLDOWN:
                last_event = time.time()
                print("Event:", hit["trigger"], round(hit["motion"] * 100), "% of frame, analyzing...", flush=True)
                run_event("auto", hit["trigger"], hit["frame"], hit["faces"], hit["motion"])
                watcher.reset()  # the current scene becomes the new background
        except Exception as e:
            print(f"Error: {e}", flush=True)
            time.sleep(5)
        time.sleep(1 / CHECK_FPS)


if __name__ == "__main__":
    main()
