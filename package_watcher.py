#!/usr/bin/env python3
"""
Package Watcher: a focused LangGraph camera & temperature agent on a Raspberry Pi.

Always on: camera motion check at ~4 fps, DHT11 temperature/humidity loop.
On motion, a LangGraph pipeline runs:

  gate (snooze / cooldown) -> analyze (Moondream: box/package check + description)
    -> decide:
        - no active package + box seen  -> delivered: read DHT11 -> SLM bring-in advice -> notify + log
        - active package + box gone     -> picked_up: clear active package -> quiet notify + log
        - active package + box still in -> still_present: suppress duplicate alerts
        - no active package + no box    -> empty: ignore (keep timeline focused)

While a package is outside, temperature_loop() re-checks DHT11 readings and sends a
follow-up SLM advisory if conditions turn too hot, freezing/cold, or overly humid.

Phone (ntfy) and Web Dashboard ("Ask"):
  Users can ask "Should I bring the package inside?" and optionally include details about
  what is inside ("It has chocolate and skincare"), which the SLM combines with live sensor data.

Run: python3 package_watcher.py
"""

import base64
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import TypedDict

import cv2
import requests
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from tinydb import TinyDB

# --- Config ---
BASE_DIR = os.path.expanduser("~/package-watcher")
EVENTS_DIR = os.path.join(BASE_DIR, "events")
THUMBS_DIR = os.path.join(EVENTS_DIR, "thumbs")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LEGACY_CONFIG_PATH = os.path.expanduser("~/guardian/config.json")
DB_PATH = os.path.join(BASE_DIR, "events.json")
STATE_PATH = os.path.join(BASE_DIR, "state.json")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

FRAME_SIZE = (1280, 960)
CHECK_FPS = 4
MOTION_THRESHOLD = 0.01  # fraction of the frame that must change
MOTION_FRAMES = 2  # consecutive motion frames needed (filters noise)
EVENT_COOLDOWN = 20  # seconds between automatic events
EVENT_QUEUE_SIZE = 3

SHORT_EXPOSURE = True
BURST_FRAMES = 3
BURST_INTERVAL = 0.25
MAX_IMAGES = 500

# AWS backup: photos to S3; event records go over MQTT -> IoT rule -> DynamoDB.
# Kept on guardian/events & guardian/event by default so existing AWS IoT/S3 rules work unchanged.
UPLOAD_TO_S3 = True
S3_BUCKET = "fuzail-ras-pi-data"
S3_REGION = "us-east-1"
S3_PREFIX = "guardian/events"
MQTT_EVENT_SUBTOPIC = "guardian/event"
RETENTION_DAYS = 30

# Temperature & humidity (DHT11 on GPIO17, physical pin 11)
TEMP_ENABLED = True
TEMP_GPIO = 17
TEMP_READ_SECONDS = 60
TEMP_LOG_SECONDS = 600
TEMP_DB_PATH = os.path.join(BASE_DIR, "temperature.json")
TEMP_KEEP_DAYS = 7
TEMP_LOW_C = 4.0       # ~39.2F: cold / freeze risk for packages outside
TEMP_HIGH_C = 27.0     # ~80.6F: heat / melt / spoilage risk for packages outside
HUMIDITY_HIGH = 80.0   # %: moisture / soggy cardboard risk
HEAT_LIMIT_C = 35.0    # ~95F: extreme ambient safety alert even with no active package
HEAT_JUMP_C = 5.0
HEAT_JUMP_MINUTES = 10
TEMP_ALERT_COOLDOWN = 1800

# Buzzer actuator (GPIO27 = physical pin 13) over MQTT
BUZZER_ENABLED = True
BUZZER_GPIO = 27
BUZZER_PASSIVE = False
BUZZER_TONE_HZ = 3000
BUZZER_ACTIVE_LOW = False
BUZZER_VOLUME = 0.6
BUZZER_ON_DELIVERY = "double"    # pattern on package drop-off, or None
BUZZER_ON_TEMP_ALERT = "alarm"   # pattern on unsafe temp while package outside, or None
BUZZER_ACTIONS = ("on", "off", "beep", "double", "alarm", "chirp")
MQTT_HOST = "localhost"
BUZZER_CMD_TOPIC = "fuzail-pi/cmd/buzzer"
BUZZER_STATE_TOPIC = "fuzail-pi/state/buzzer"

# Vision & SLM (Ollama)
VISION_SERVERS = ["http://localhost:11434"]
VISION_MODEL = "moondream"
VISION_THREADS = 2
VISION_SIZE = (640, 480)
VISION_MAX_TOKENS = 60
VISION_KEEP_ALIVE = "30m"

AGENT_MODEL = "llama3.2:1b"
AGENT_ENGINE = "tools"  # "tools" | "rules"
TOOL_MODEL = AGENT_MODEL
ASK_MAX_TOKENS = 180
AGENT_LOG_PATH = os.path.join(BASE_DIR, "agent_log.jsonl")
NTFY_SERVER = "https://ntfy.sh"
HEARTBEAT_SECONDS = 60

WEB_ENABLED = True
WEB_PORT = 8080
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
CPU_TEMP_PATH = "/sys/class/thermal/thermal_zone0/temp"

PACKAGE_PROMPT = (
    "Is there a cardboard box or delivery package in this image? "
    "Start with Yes or No, then describe what you see in one short sentence."
)
DESCRIPTION_SENTENCES = 2

PACKAGE_PATTERN = re.compile(
    r"\b(package|packages|parcel|parcels|box|boxes|cardboard|crate|mailer|envelope|delivery|deliveries)\b", re.I
)
NEGATED_PACKAGE_PATTERN = re.compile(
    r"\b(no|not|without|neither|none)\b[^.?!]{0,25}"
    r"\b(package|packages|parcel|parcels|box|boxes|cardboard|crate|mailer|envelope|delivery)\b", re.I
)
FILLER_PATTERNS = [
    re.compile(r"^(the|this) (image|picture|photo|scene) (shows|features|depicts|displays|captures|"
               r"contains|is of|presents)\s+", re.I),
    re.compile(r"^in (the|this) (image|picture|photo),?\s+(there is |there are )?", re.I),
    re.compile(r"^(a|an) (close-up|closeup|wide|blurry|dark|low-angle) (view|shot|image|photo) of\s+", re.I),
]
TEMP_PATTERN = re.compile(r"\b(temp|temperature|hot|cold|warm|cool|humid|humidity|degrees?|weather|cpu|thermal)\b", re.I)
ADVICE_PATTERN = re.compile(
    r"\b(bring|grab|leave|safe|wait|spoil|melt|freeze|inside|indoors|perish|fragile|chocolate|food|"
    r"grocer\w*|medicine|insulin|electronic\w*|vinyl|skincare|should i)\b", re.I
)
HISTORY_PATTERN = re.compile(
    r"\b(when|earlier|yesterday|today|tonight|this (morning|afternoon|evening|week)|"
    r"how many|did (any|a)|has (any|a)|anything|what happened|history|recently|last (time|\d+)|past \d+)\b", re.I
)
SUMMARY_PATTERN = re.compile(r"\b(summar\w*|recap|overview|report|digest)\b", re.I)
COMMAND_PATTERN = re.compile(
    r"^\s*(please\s+)?(turn|switch|shut|disable|enable|delete|erase|remove|unlock|lock|open|close|kill|"
    r"reboot|restart|shutdown|format|send|email|text|call|upload|record|change|set|start|stop)\b", re.I
)
HISTORY_MAX_EVENTS = 15

llm_answer = ChatOllama(model=AGENT_MODEL, temperature=0, num_thread=2, num_predict=ASK_MAX_TOKENS)

runtime = {
    "snooze_until": 0.0,
    "active_package": None,  # {"id", "arrived_at", "description", "image", "notes", "advice", "temp_c", "humidity", "last_temp_alert"}
    "temp": None,            # {"c", "f", "humidity", "time"}
    "temp_recent": deque(maxlen=60),
    "last_temp_alert": 0.0,
    "buzzer_volume": None,
}
watcher = None
buzzer = None
db_lock = threading.Lock()
state_lock = threading.Lock()
ask_lock = threading.Lock()
_dbs = {}
jobs = queue.Queue(maxsize=EVENT_QUEUE_SIZE)
topics = {}


# --- Setup ---
def load_topics():
    """Load or create ntfy topics & web_key; migrates ~/guardian/config.json on first run if present."""
    os.makedirs(EVENTS_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_PATH) and os.path.exists(LEGACY_CONFIG_PATH):
        shutil.copy2(LEGACY_CONFIG_PATH, CONFIG_PATH)
        print(f"Copied existing config from {LEGACY_CONFIG_PATH} to {CONFIG_PATH}", flush=True)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    else:
        name = f"fuzail-package-{secrets.token_hex(8)}"
        cfg = {"alert_topic": name, "cmd_topic": f"{name}-cmd"}
    if "web_key" not in cfg:
        cfg["web_key"] = secrets.token_urlsafe(16)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    os.chmod(CONFIG_PATH, 0o600)
    return cfg


def load_state():
    try:
        with open(STATE_PATH) as f:
            saved = json.load(f)
        runtime["active_package"] = saved.get("active_package")
        runtime["buzzer_volume"] = saved.get("buzzer_volume")
    except (OSError, ValueError):
        pass


def save_state():
    os.makedirs(BASE_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "active_package": runtime["active_package"],
            "buzzer_volume": runtime["buzzer_volume"],
        }, f)
    os.replace(tmp, STATE_PATH)


def _new_id(suffix=""):
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}{suffix}"


def _existing_image(name):
    return name if name and os.path.exists(os.path.join(EVENTS_DIR, name)) else ""


def _set_snooze(minutes):
    runtime["snooze_until"] = time.time() + minutes * 60 if minutes > 0 else 0.0


# --- Camera + motion ---
class Watcher:
    def __init__(self, camera=None):
        if camera is None:
            from picamera2 import Picamera2

            camera = Picamera2()
            camera.configure(camera.create_video_configuration(
                main={"size": FRAME_SIZE, "format": "RGB888"}))
            if SHORT_EXPOSURE:
                try:
                    from libcamera import controls

                    camera.set_controls({"AeExposureMode": controls.AeExposureModeEnum.Short})
                except Exception as e:
                    print(f"Short exposure not available ({e})", flush=True)
            camera.start()
            time.sleep(2)
        self.cam = camera
        self.lock = threading.Lock()
        self.avg = None
        self.motion_count = 0

    def grab(self):
        with self.lock:
            return self.cam.capture_array()

    def burst(self, first_frame):
        frames = [first_frame]
        for _ in range(BURST_FRAMES - 1):
            time.sleep(BURST_INTERVAL)
            frames.append(self.grab())
        return frames

    def reset(self):
        self.avg = None
        self.motion_count = 0

    def check(self):
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
        return {"trigger": "motion", "motion": round(motion, 3), "frame": frame}


def sharpness(frame):
    gray = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def pick_sharpest(frames):
    return max(frames, key=sharpness)


def save_frame(frame, name):
    os.makedirs(EVENTS_DIR, exist_ok=True)
    path = os.path.join(EVENTS_DIR, f"{name}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


# --- Vision + package parsing ---
def first_sentences(text, n=DESCRIPTION_SENTENCES):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    complete = [p for p in parts if p.endswith((".", "!", "?"))]
    text = " ".join(complete[:n]) if complete else text.strip()
    for pattern in FILLER_PATTERNS:
        text = pattern.sub("", text, count=1)
    return text[:1].upper() + text[1:] if text else ""


def parse_package_response(text):
    """Return (package_present: bool, clean_description: str) from Moondream's reply."""
    raw = (text or "").strip()
    if not raw or raw.startswith("(vision failed"):
        return False, raw
    lower = raw.lower()
    if re.match(r"^\s*yes\b", lower):
        present = True
    elif re.match(r"^\s*no\b", lower):
        present = False
    elif NEGATED_PACKAGE_PATTERN.search(lower):
        present = False
    else:
        present = bool(PACKAGE_PATTERN.search(lower))
    cleaned = re.sub(r"^\s*(?:yes|no)[,.:!\s-]+", "", raw, flags=re.I).strip()
    cleaned = first_sentences(cleaned or raw)
    return present, cleaned


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
                timeout=(3, 600),
            )
            resp.raise_for_status()
            return first_sentences(resp.json()["response"])
        except requests.RequestException as e:
            last_error = e
    raise RuntimeError(f"no vision server worked ({last_error})")


# --- Notifications & cloud backup ---
def _header(text, limit=900):
    return " ".join(text.split()).encode("ascii", "ignore").decode()[:limit]


def send_ntfy(title, message, image_path=None, tags="", priority=3, buttons=True):
    if not topics:
        return
    url = f"{NTFY_SERVER}/{topics['alert_topic']}"
    cmd_url = f"{NTFY_SERVER}/{topics['cmd_topic']}"
    headers = {"Title": _header(title, 200), "Priority": str(priority)}
    if tags:
        headers["Tags"] = tags
    if buttons:
        headers["Actions"] = (
            f"http, Picked up, {cmd_url}, body=picked up, clear=true; "
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
    def _upload():
        try:
            import boto3

            boto3.client("s3", region_name=S3_REGION).upload_file(
                path, S3_BUCKET, key, ExtraArgs={"ContentType": "image/jpeg"})
        except Exception as e:
            print(f"S3 upload failed for {key}: {e}", flush=True)

    threading.Thread(target=_upload, daemon=True).start()


def publish_mqtt(event, subtopic=MQTT_EVENT_SUBTOPIC):
    try:
        from mqtt_publish import publish

        publish(subtopic, event)
    except Exception as e:
        print(f"MQTT publish failed: {e}", flush=True)


def listen_commands(out_queue):
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


def heartbeat(url):
    while True:
        try:
            requests.get(url, timeout=10)
        except requests.RequestException:
            pass
        time.sleep(HEARTBEAT_SECONDS)


# --- Temperature & SLM package advisor ---
def c_to_f(c):
    return round(c * 9 / 5 + 32, 1)


def temp_text(c):
    return f"{c_to_f(c)}F ({c}C)"


def _duration(seconds):
    hours, minutes = int(seconds // 3600), int(seconds % 3600 // 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def _db(path):
    return _dbs.setdefault(path, TinyDB(path))


def temp_rows(start=0.0, end=None):
    end = end or time.time()
    if not os.path.exists(TEMP_DB_PATH):
        return []
    with db_lock:
        rows = _db(TEMP_DB_PATH).all()
    return [r for r in rows if start <= r["time"] <= end]


def log_temperature(c, humidity):
    now = time.time()
    os.makedirs(BASE_DIR, exist_ok=True)
    with db_lock:
        db = _db(TEMP_DB_PATH)
        db.insert({"time": now, "c": c, "humidity": humidity})
        cutoff = now - TEMP_KEEP_DAYS * 86400
        db.remove(lambda r: r["time"] < cutoff)
    ts = datetime.now(timezone.utc)
    publish_mqtt({
        "filename": f"temp_{ts.strftime('%Y%m%dT%H%M%SZ')}.json",
        "reading_type": "temperature",
        "temperature_c": str(c),
        "temperature_f": str(c_to_f(c)),
        "humidity": str(humidity),
        "timestamp": ts.isoformat(),
    }, subtopic="temperature")


def _rule_advice(temp_c, humidity, notes="", has_package=True):
    """Deterministic fallback when Ollama is offline or slow."""
    text_notes = notes.lower()
    sensitive = bool(re.search(
        r"\b(chocolate|food|grocer\w*|frozen|ice|meat|dairy|flower|plant|wine|medicine|insulin|"
        r"medication|skincare|cosmetic|vinyl|candle|battery|electronic\w*|laptop|phone|liquid)\b",
        text_notes,
    ))
    sturdy = bool(re.search(
        r"\b(paper towel|toilet paper|book|clothes|apparel|plastic|hardware|tool|furniture)\b",
        text_notes,
    ))
    if temp_c is None:
        if sensitive:
            return "Bring it inside as soon as you can since it contains temperature-sensitive items (sensor unavailable)."
        return "Temperature sensor is unavailable; bring the package inside when convenient, or sooner if it is perishable."
    reading = f"{temp_text(temp_c)} and {humidity}% humidity"
    if temp_c <= TEMP_LOW_C:
        return f"Outside is cold ({reading}). Bring the package inside soon to avoid freezing or cold damage."
    if temp_c >= TEMP_HIGH_C:
        if sturdy and not sensitive:
            return f"Outside is warm ({reading}), which is hot for perishables, though {notes} should hold up for a short time."
        return f"Outside is warm/hot ({reading}). Bring the package inside now to avoid heat damage or spoilage."
    if humidity is not None and humidity >= HUMIDITY_HIGH:
        return f"Humidity is high ({humidity}% at {temp_text(temp_c)}). Bring cardboard boxes inside soon before they dampen."
    if sensitive:
        return f"Outside is {reading}; even though the temperature is mild, bring {notes} inside soon to be safe."
    if has_package:
        return f"Outside is {reading} — mild conditions, safe for most dry goods for a little while unless it holds perishables."
    return f"Outside is {reading} — mild conditions right now for standard deliveries."


def _package_facts(package=None, extra_info=""):
    lines = []
    cur = runtime["temp"]
    if cur:
        lines.append(f"Current outdoor temperature: {temp_text(cur['c'])}, {cur['humidity']}% humidity.")
    else:
        lines.append("Current outdoor temperature: unavailable (sensor not reporting).")

    if package:
        arrived = datetime.fromtimestamp(package["arrived_at"]).strftime("%I:%M %p").lstrip("0")
        outside_for = _duration(max(0, time.time() - package["arrived_at"]))
        lines.append(f"Package status: outside since {arrived} ({outside_for} ago).")
        if package.get("description"):
            lines.append(f"Camera description: {package['description']}")
        if package.get("notes"):
            lines.append(f"User-provided contents/info: {package['notes']}")
        rows = temp_rows(package["arrived_at"], time.time())
        if rows:
            cs = [r["c"] for r in rows]
            lines.append(f"Temperature since arrival: low {temp_text(min(cs))}, high {temp_text(max(cs))}.")
    else:
        lines.append("Package status: no package currently tracked outside.")

    if extra_info and (not package or extra_info != package.get("notes")):
        lines.append(f"Additional user info: {extra_info}")
    return "\n".join(lines)


def _ask_ollama(prompt, fallback, label="LLM"):
    try:
        reply = llm_answer.invoke(prompt).content.strip()
        if reply:
            return reply, True
    except Exception as e:
        print(f"{label} failed ({e}); using fallback", flush=True)
    return fallback, False


def advise_package(question="Should I bring the package inside?", package=None, extra_info=""):
    """Use the temperature sensor + package context as input to the SLM for bring-inside advice."""
    extra = (extra_info or "").strip()
    with state_lock:
        pkg = package if package is not None else runtime["active_package"]
        if pkg and extra:
            if not pkg.get("notes"):
                pkg["notes"] = extra[:240]
            elif extra.lower() not in pkg["notes"].lower():
                pkg["notes"] = f"{pkg['notes']}; {extra}"[:240]
            if runtime["active_package"] is pkg:
                save_state()
        facts = _package_facts(pkg, extra)
        notes = extra or (pkg.get("notes", "") if pkg else "")
        has_pkg = bool(pkg)

    cur = runtime["temp"] or {}
    fallback = _rule_advice(cur.get("c"), cur.get("humidity"), notes, has_package=has_pkg)
    prompt = (
        "You are Package Watcher, an assistant advising whether to bring a delivered package inside "
        "based on outdoor temperature, humidity, and any known package contents.\n"
        "Use ONLY the facts below. Never invent readings or times. Give temperatures in F.\n"
        "Reply in 1-3 concise, practical sentences: clearly state whether to bring it inside now or if it can wait, and why.\n\n"
        f"Facts:\n{facts}\n\nQuestion: {question}\nAdvice:"
    )
    reply, _ = _ask_ollama(prompt, fallback, "SLM advisor")
    return reply


def check_package_temp(c, humidity, now):
    """Alert if conditions become unsafe while a package is sitting outside (or extreme ambient heat)."""
    recent = runtime["temp_recent"]
    recent.append((now, c))
    window = [t for when, t in recent if now - when <= HEAT_JUMP_MINUTES * 60]

    with state_lock:
        pkg = runtime["active_package"]
        if not pkg:
            if c >= HEAT_LIMIT_C and now - runtime["last_temp_alert"] >= TEMP_ALERT_COOLDOWN:
                runtime["last_temp_alert"] = now
                send_ntfy(f"High outdoor temp ({_clock()})",
                          f"Sensor reads {temp_text(c)} ({humidity}% humidity).",
                          tags="thermometer", priority=4, buttons=False)
                if BUZZER_ON_TEMP_ALERT:
                    buzz(BUZZER_ON_TEMP_ALERT, "rule: extreme_temp")
            return
        if now < runtime["snooze_until"] or now - pkg.get("last_temp_alert", 0.0) < TEMP_ALERT_COOLDOWN:
            return

    reason = None
    if c <= TEMP_LOW_C:
        reason = f"Cold alert ({temp_text(c)})"
    elif c >= TEMP_HIGH_C:
        reason = f"Heat alert ({temp_text(c)})"
    elif humidity >= HUMIDITY_HIGH:
        reason = f"High humidity ({humidity}%)"
    elif window and c - min(window) >= HEAT_JUMP_C:
        reason = f"Temp rose +{round(c - min(window), 1)}C in {HEAT_JUMP_MINUTES}m"
    if not reason:
        return

    advice = advise_package(
        "The weather changed while the package is still outside. Should I bring it inside now?",
        package=pkg,
    )
    with state_lock:
        # Guard against package picked up while Ollama was running
        if runtime["active_package"] is not pkg:
            return
        pkg["last_temp_alert"] = now
        pkg["advice"] = advice
        save_state()
        notes = pkg.get("notes", "")
        img = _existing_image(pkg.get("image", ""))

    photo = os.path.join(EVENTS_DIR, img) if img else None
    send_ntfy(f"Package weather alert ({_clock()})", f"{reason}. {advice}",
              photo, "thermometer,warning", priority=5, buttons=True)
    if BUZZER_ON_TEMP_ALERT:
        buzz(BUZZER_ON_TEMP_ALERT, "rule: package_temp")

    record_event("temp_alert", "temp_alert", "temperature",
                 description=f"{reason} while package outside.", advice=advice,
                 notes=notes, package_present=True, alert=True, reason=reason,
                 image_name=img, event_id=_new_id("_temp"))
    print(f"Package weather alert: {reason} -> {advice[:90]}", flush=True)


def temperature_loop():
    try:
        from dht11 import DHT11

        sensor = DHT11(TEMP_GPIO)
    except Exception as e:
        print(f"Temperature sensor off ({e})", flush=True)
        return
    print(f"Temperature sensor on (DHT11, GPIO{TEMP_GPIO})", flush=True)
    last_log = 0.0
    while True:
        try:
            c, humidity = sensor.read()
            now = time.time()
            runtime["temp"] = {"c": c, "f": c_to_f(c), "humidity": humidity, "time": now}
            if now - last_log >= TEMP_LOG_SECONDS:
                last_log = now
                log_temperature(c, humidity)
            check_package_temp(c, humidity, now)
        except Exception as e:
            print(f"Temperature read failed: {e}", flush=True)
        time.sleep(TEMP_READ_SECONDS)


# --- Database & event logging ---
def db_all():
    if not os.path.exists(DB_PATH):
        return []
    with db_lock:
        return _db(DB_PATH).all()


def record_event(outcome, mode, trigger, description="", advice="", notes="", package_present=False,
                 alert=False, reason="", image_path=None, image_name="", event_id=None, **extra):
    os.makedirs(EVENTS_DIR, exist_ok=True)
    now = time.time()
    cur = runtime["temp"] or {}
    img = os.path.basename(image_path) if image_path else image_name
    event = {
        "id": event_id or _new_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "outcome": outcome,
        "trigger": trigger,
        "package_present": bool(package_present),
        "description": description,
        "advice": advice,
        "notes": notes,
        "temp_c": extra.get("temp_c", cur.get("c")),
        "temp_f": extra.get("temp_f", cur.get("f")),
        "humidity": extra.get("humidity", cur.get("humidity")),
        "alert": bool(alert),
        "reason": reason,
        "image": img,
        "expires_at": int(now) + RETENTION_DAYS * 86400,
    }
    if "motion" in extra:
        event["motion"] = extra["motion"]
    if "timing" in extra:
        event["timing"] = extra["timing"]
    if image_path and UPLOAD_TO_S3:
        s3_key = f"{S3_PREFIX}/{img}"
        event["s3_key"] = s3_key
        upload_photo(image_path, s3_key)
    else:
        event["s3_key"] = ""
    with db_lock:
        db = _db(DB_PATH)
        db.insert(event)
        db.remove(lambda r: r.get("expires_at") and r["expires_at"] < now)
    publish_mqtt(event)

    photos = sorted(f for f in os.listdir(EVENTS_DIR) if f.endswith(".jpg"))
    for old in photos[:-MAX_IMAGES]:
        os.remove(os.path.join(EVENTS_DIR, old))
        thumb = os.path.join(THUMBS_DIR, old)
        if os.path.exists(thumb):
            os.remove(thumb)
    return event


# --- LangGraph event pipeline ---
class WatcherState(TypedDict, total=False):
    event_id: str
    mode: str        # auto | snapshot | question
    trigger: str     # motion | manual
    motion: float
    question: str
    image_path: str
    description: str
    package_present: bool
    outcome: str     # delivered | picked_up | still_present | empty | snapshot | question
    advice: str
    temp_c: float
    temp_f: float
    humidity: float
    alert: bool
    reason: str
    timing: dict


def _clock():
    return datetime.now().strftime("%H:%M")


def _mark(state, name, **extra):
    timing = dict(state.get("timing") or {})
    if "t0" in timing:
        timing[name] = round(time.time() - timing["t0"], 2)
    timing.update(extra)
    return timing


def gate(state: WatcherState):
    if state["mode"] == "auto" and time.time() < runtime["snooze_until"]:
        return {"outcome": "snoozed", "alert": False, "reason": "Snoozed"}
    return {}


def after_gate(state: WatcherState):
    return "end" if state.get("outcome") == "snoozed" else "analyze"


def analyze(state: WatcherState):
    started = time.time()
    prompt = state["question"] if state["mode"] == "question" else PACKAGE_PROMPT
    try:
        raw = describe_image(state["image_path"], prompt)
    except Exception as e:
        raw = f"(vision failed: {e})"
    if state["mode"] == "question":
        return {"description": raw, "package_present": bool(runtime["active_package"]),
                "timing": _mark(state, "described", describe_s=round(time.time() - started, 2))}
    present, desc = parse_package_response(raw)
    return {"description": desc or raw, "package_present": present,
            "timing": _mark(state, "described", describe_s=round(time.time() - started, 2))}


def decide(state: WatcherState):
    mode = state["mode"]
    cur = runtime["temp"]
    temp_fields = ({"temp_c": cur["c"], "temp_f": cur["f"], "humidity": cur["humidity"]}
                   if cur else {})
    if mode != "auto":
        return {**temp_fields, "outcome": mode, "alert": False, "reason": mode.capitalize()}

    with state_lock:
        active = runtime["active_package"]
    present = state.get("package_present", False)

    if active is None and present:
        pkg = {
            "id": state["event_id"],
            "arrived_at": time.time(),
            "description": state.get("description", ""),
            "image": os.path.basename(state["image_path"]),
            "notes": "",
            "advice": "",
            "temp_c": cur["c"] if cur else None,
            "humidity": cur["humidity"] if cur else None,
            "last_temp_alert": 0.0,
        }
        advice = advise_package("A package just arrived. Should I bring it inside?", package=pkg)
        pkg["advice"] = advice
        with state_lock:
            runtime["active_package"] = pkg
            save_state()
        return {**temp_fields, "outcome": "delivered", "advice": advice, "alert": True,
                "reason": "Package dropped off"}

    if active is not None and not present:
        with state_lock:
            runtime["active_package"] = None
            save_state()
        return {**temp_fields, "outcome": "picked_up", "alert": True,
                "reason": "Package picked up"}

    if active is not None and present:
        return {**temp_fields, "outcome": "still_present", "alert": False,
                "reason": "Package still outside"}

    return {**temp_fields, "outcome": "empty", "alert": False, "reason": "No package"}


def after_decide(state: WatcherState):
    if state.get("outcome") in ("still_present", "empty"):
        return "end"
    return "notify"


def notify(state: WatcherState):
    outcome = state["outcome"]
    if outcome == "delivered":
        temp_line = (f"Outside: {temp_text(state['temp_c'])}, {state['humidity']}% humidity"
                     if "temp_c" in state else "Outside temperature unavailable")
        msg = " | ".join([state["description"], temp_line, f"Advice: {state.get('advice', '')}"])
        send_ntfy(f"Package dropped off ({_clock()})", msg, state["image_path"],
                  tags="package", priority=4, buttons=True)
        if BUZZER_ON_DELIVERY:
            buzz(BUZZER_ON_DELIVERY, "rule: delivery")
    elif outcome == "picked_up":
        send_ntfy(f"Package picked up ({_clock()})", "The package is no longer in view.",
                  state["image_path"], tags="white_check_mark", priority=2, buttons=False)
    elif outcome == "snapshot":
        send_ntfy(f"Snapshot ({_clock()})", state["description"],
                  state["image_path"], tags="camera", priority=3, buttons=False)
    elif outcome == "question":
        send_ntfy(f"Answer ({_clock()})", state["description"],
                  state["image_path"], tags="speech_balloon", priority=3, buttons=False)
    return {"timing": _mark(state, "alert")}


def log_event(state: WatcherState):
    timing = _mark(state, "total")
    extra = {"motion": state.get("motion", 0.0)}
    if "temp_c" in state:
        extra.update(temp_c=state["temp_c"], temp_f=state["temp_f"], humidity=state["humidity"])
    if "t0" in timing:
        extra["timing"] = {k: v for k, v in timing.items() if k != "t0"}
    event = record_event(
        state["outcome"], state["mode"], state["trigger"],
        description=state.get("description", ""), advice=state.get("advice", ""),
        package_present=bool(state.get("package_present")), alert=bool(state.get("alert")),
        reason=state.get("reason", ""), image_path=state["image_path"],
        event_id=state["event_id"], **extra,
    )
    print(f"[{event['id']}] {event['outcome']}: {(event['description'] or '-')[:80]}", flush=True)
    return {}


def build_graph():
    g = StateGraph(WatcherState)
    for name, fn in [("gate", gate), ("analyze", analyze), ("decide", decide),
                     ("notify", notify), ("log", log_event)]:
        g.add_node(name, fn)
    g.add_edge(START, "gate")
    g.add_conditional_edges("gate", after_gate, {"end": END, "analyze": "analyze"})
    g.add_edge("analyze", "decide")
    g.add_conditional_edges("decide", after_decide, {"end": END, "notify": "notify"})
    g.add_edge("notify", "log")
    g.add_edge("log", END)
    return g.compile()


graph = build_graph()


def run_event(mode, trigger, frame=None, motion=0.0, question=None, timing=None):
    event_id = _new_id()
    if frame is None:
        frame = watcher.grab()
    state = {"timing": dict(timing or {})}
    return graph.invoke({
        "event_id": event_id, "mode": mode, "trigger": trigger,
        "motion": motion, "question": question or "",
        "image_path": save_frame(frame, event_id), "timing": _mark(state, "queue"),
    })


def mark_package_picked_up(source="manual", notify_phone=True):
    """Manually mark the active package as brought inside / picked up."""
    with state_lock:
        pkg = runtime["active_package"]
        if not pkg:
            return False
        runtime["active_package"] = None
        save_state()
    record_event("picked_up", "manual", source, description=f"Marked picked up ({source}).",
                 notes=pkg.get("notes", ""), reason="Package picked up", event_id=_new_id("_pu"))
    if notify_phone:
        send_ntfy(f"Package picked up ({_clock()})", "Marked as brought inside.",
                  tags="white_check_mark", priority=2, buttons=False)
    print(f"Package marked picked up ({source})", flush=True)
    return True


def enqueue(**event):
    try:
        jobs.put_nowait(event)
    except queue.Full:
        print("AI busy: event dropped (queue full)", flush=True)


def ai_worker():
    while True:
        event = jobs.get()
        try:
            if event.pop("kind", None) == "ask":
                answer_on_phone(event["question"])
            else:
                run_event(**event)
        except Exception as e:
            print(f"Event failed: {e}", flush=True)


# --- Phone commands ---
def status_text():
    today = datetime.now().strftime("%Y-%m-%d")
    events = sorted((e for e in db_all() if _local(e).strftime("%Y-%m-%d") == today),
                    key=lambda e: e["timestamp"])
    deliveries = sum(1 for e in events if _kind(e) == "delivered")
    if time.time() < runtime["snooze_until"]:
        state = "Snoozed until " + datetime.fromtimestamp(runtime["snooze_until"]).strftime("%H:%M")
    elif runtime["active_package"]:
        outside_for = _duration(max(0, time.time() - runtime["active_package"]["arrived_at"]))
        state = f"Package outside ({outside_for})"
    else:
        state = "Watching (no package outside)"
    temp = f" Outside: {temp_text(runtime['temp']['c'])}, {runtime['temp']['humidity']}% humidity." if runtime["temp"] else ""
    return f"{state}. Deliveries today: {deliveries}.{temp}"


def handle_command(text):
    t = text.lower().strip()
    print(f"Command: {text}", flush=True)
    if t in ("snapshot", "photo", "pic", "take another"):
        frame = pick_sharpest(watcher.burst(watcher.grab()))
        enqueue(mode="snapshot", trigger="manual", frame=frame)
    elif t in ("picked up", "pickedup", "got it", "brought inside", "collected"):
        if not mark_package_picked_up("phone", notify_phone=False):
            send_ntfy("No active package", "There is no package currently marked outside.", tags="package", buttons=False)
        else:
            send_ntfy("Package picked up", "Marked as brought inside.", tags="white_check_mark", buttons=False)
    elif t.startswith("snooze"):
        digits = re.findall(r"\d+", t)
        minutes = min(int(digits[0]), 240) if digits else 60
        _set_snooze(minutes)
        send_ntfy("Snoozed", f"Alerts paused for {minutes} min. Send 'resume' to turn back on.",
                  tags="zzz", buttons=False)
    elif t in ("resume", "unsnooze"):
        _set_snooze(0)
        send_ntfy("Alerts resumed", "Package alerts are back on.", tags="package", buttons=False)
    elif t == "status":
        send_ntfy("Status", status_text(), tags="bar_chart", buttons=False)
    elif m := re.fullmatch(r"(?:buzzer )?volume (\d{1,3})%?", t):
        ok = buzz("volume", "phone", m.group(1))
        send_ntfy("Buzzer", f"Buzzer volume set to {int(m.group(1))}%." if ok
                  else "Volume must be 5 to 100, and a buzzer must be connected.", tags="bell", buttons=False)
    elif t in ("buzz", "beep", "alarm", "chirp", "buzzer off", "silence", "quiet", "stop"):
        action = {"buzz": "beep", "buzzer off": "off", "silence": "off", "quiet": "off", "stop": "off"}.get(t, t)
        ok = buzz(action, "phone")
        send_ntfy("Buzzer", f"Buzzer: {action}" if ok else "The buzzer isn't available.",
                  tags="bell", buttons=False)
    else:
        enqueue(kind="ask", question=text)


# --- Buzzer actuator (MQTT) ---
def parse_volume(value):
    try:
        v = float(str(value).strip().rstrip("%"))
    except ValueError:
        return None
    v = v / 100 if v > 1 else v
    return round(v, 2) if 0.05 <= v <= 1.0 else None


def buzzer_command(action, source, sent_at=None, volume=None):
    if buzzer is None:
        return {"device": "buzzer", "state": "unavailable", "action": action, "source": source}
    if action == "volume":
        runtime["buzzer_volume"] = round(buzzer.set_volume(volume), 2)
        save_state()
    elif action == "off":
        buzzer.off()
    else:
        buzzer.play(action)
    state = {"device": "buzzer", "state": buzzer.state, "action": action, "source": source,
             "volume": round(buzzer.volume, 2), "time": datetime.now(timezone.utc).isoformat()}
    if isinstance(sent_at, (int, float)):
        state["latency_ms"] = round((time.time() - sent_at) * 1000, 1)
    runtime["buzzer"] = state
    return state


def buzz(action, source, volume=None):
    if buzzer is None:
        return False
    if action == "volume":
        volume = parse_volume(volume)
        if volume is None:
            return False
    elif action not in BUZZER_ACTIONS:
        return False
    message = {"action": action, "source": source, "sent_at": time.time()}
    if action == "volume":
        message["volume"] = volume
    try:
        from mqtt_publish import publish

        publish(BUZZER_CMD_TOPIC.split("/", 1)[1], message)
    except Exception as e:
        print(f"MQTT unavailable ({e}); controlling buzzer directly", flush=True)
        buzzer_command(action, source, volume=volume)
    return True


def mqtt_actuator_loop():
    import paho.mqtt.client as mqtt

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="package-watcher-actuators")
    except AttributeError:
        client = mqtt.Client(client_id="package-watcher-actuators")

    def on_connect(c, userdata, flags, rc, properties=None):
        c.subscribe(BUZZER_CMD_TOPIC, qos=1)
        print(f"Buzzer listening on MQTT {BUZZER_CMD_TOPIC}", flush=True)

    def on_message(c, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except ValueError:
            data = msg.payload.decode(errors="ignore")
        if not isinstance(data, dict):
            data = {"action": str(data)}
        action = str(data.get("action", "")).strip().lower()
        volume = None
        if action == "volume":
            volume = parse_volume(data.get("volume"))
            if volume is None:
                return
        elif action not in BUZZER_ACTIONS:
            return
        state = buzzer_command(action, str(data.get("source", "mqtt"))[:40], data.get("sent_at"), volume)
        c.publish(BUZZER_STATE_TOPIC, json.dumps(state), qos=1, retain=True)

    client.on_connect, client.on_message = on_connect, on_message
    client.reconnect_delay_set(1, 30)
    client.connect_async(MQTT_HOST, 1883)
    client.loop_forever(retry_first_connection=True)


def start_buzzer():
    global buzzer
    try:
        from actuators import Buzzer

        buzzer = Buzzer(BUZZER_GPIO, BUZZER_PASSIVE, BUZZER_ACTIVE_LOW, tone_hz=BUZZER_TONE_HZ,
                        volume=runtime["buzzer_volume"] or BUZZER_VOLUME)
    except Exception as e:
        print(f"Buzzer off ({e})", flush=True)
        return
    threading.Thread(target=mqtt_actuator_loop, daemon=True).start()


# --- Ask & Tool-Calling Agent ---
def cpu_temp_c():
    try:
        with open(CPU_TEMP_PATH) as f:
            return round(int(f.read().strip()) / 1000, 1)
    except (OSError, ValueError):
        return None


def history_window(question):
    q, now = question.lower(), time.time()
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    if m := re.search(r"\b(?:last|past) (\d+) (minute|min|hour|hr|day)s?\b", q):
        n, unit = int(m.group(1)), m.group(2)
        seconds = n * {"minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400}[unit]
        return now - seconds, now, f"the last {n} {unit}s"
    if "yesterday" in q:
        return midnight - 86400, midnight, "yesterday"
    if re.search(r"\bmonth(ly)?\b", q):
        return now - 30 * 86400, now, "the last 30 days"
    if re.search(r"\bweek(ly)?\b", q):
        return now - 7 * 86400, now, "the last 7 days"
    if re.search(r"\b(today|tonight|this (morning|afternoon|evening))\b", q):
        return midnight, now, "today"
    return now - 86400, now, "the last 24 hours"


def _gather_packages(start, end, label):
    events = [e for e in sorted(db_all(), key=lambda e: e["timestamp"])
              if start <= _local(e).timestamp() <= end and _kind(e) in ("delivered", "picked_up", "temp_alert")]
    delivered = [e for e in events if _kind(e) == "delivered"]
    picked = [e for e in events if _kind(e) == "picked_up"]
    temp_alerts = [e for e in events if _kind(e) == "temp_alert"]
    lines = [
        f"Period: {label}. Deliveries: {len(delivered)}. Picked up: {len(picked)}. "
        f"Temperature warnings: {len(temp_alerts)}."
    ]
    if runtime["active_package"]:
        pkg = runtime["active_package"]
        lines.append(
            f"Currently outside: arrived at {datetime.fromtimestamp(pkg['arrived_at']).strftime('%H:%M')} "
            f"({_duration(max(0, time.time() - pkg['arrived_at']))} ago), {pkg.get('description', '')}"
        )
    else:
        lines.append("Currently outside: none.")
    if events:
        shown = events[-HISTORY_MAX_EVENTS:]
        lines.append("Events (oldest first):\n" + "\n".join(
            f"- {_local(e).strftime('%a %H:%M')}: {_kind(e)} - {(e.get('description') or e.get('reason', '')).rstrip('.')}"
            for e in shown))
        last = (delivered or events)[-1]
        plain = (f"{label.capitalize()}: {len(delivered)} delivery(s), {len(picked)} picked up. "
                 f"Last: {_local(last).strftime('%a %H:%M')}, {(last.get('description') or last.get('reason', '')).rstrip('.')}.")
    else:
        plain = f"No package deliveries recorded {'in ' if label.startswith('the ') else ''}{label}."
    related = [_web_event(e) for e in reversed((delivered or events)[-4:])]
    return "\n".join(lines), plain, related


def _gather_temperature(start, end, label):
    lines, plain = [], []
    cur = runtime["temp"]
    if cur:
        lines.append(f"Outdoor sensor now: {temp_text(cur['c'])}, {cur['humidity']}% humidity.")
        plain.append(f"It's {temp_text(cur['c'])} with {cur['humidity']}% humidity.")
    rows = temp_rows(start, end)
    if rows:
        cs = [r["c"] for r in rows]
        hot, cold = max(rows, key=lambda r: r["c"]), min(rows, key=lambda r: r["c"])
        hums = [r["humidity"] for r in rows]
        at = lambda r: datetime.fromtimestamp(r["time"]).strftime("%a %H:%M")
        lines.append(
            f"Temperature over {label}: low {temp_text(cold['c'])} ({at(cold)}), "
            f"high {temp_text(hot['c'])} ({at(hot)}), avg {temp_text(round(sum(cs) / len(cs), 1))}. "
            f"Humidity {min(hums)}-{max(hums)}%."
        )
        plain.append(f"{label.capitalize()}: low {temp_text(cold['c'])}, high {temp_text(hot['c'])}.")
    elif not cur:
        lines.append("Outdoor temperature: no readings (sensor not connected or not reading).")
        plain.append("No temperature readings yet.")
    cpu = cpu_temp_c()
    if cpu is not None:
        lines.append(f"Raspberry Pi CPU temperature now: {temp_text(cpu)}.")
    return " ".join(lines), " ".join(plain)


def compose_answer(question, facts, summary, plain):
    style = ("Write a short summary in 3-4 sentences covering deliveries first, then outdoor temperature."
             if summary else "Answer in 1-2 short sentences.")
    prompt = (
        "You are Package Watcher, an assistant for a Pi camera and temperature sensor watching for deliveries. "
        "Answer using ONLY the facts below. Never invent events, times, or numbers; if missing, say so. "
        "Give temperatures in F. " + style +
        "\n\nFacts:\n" + "\n\n".join(facts) +
        f"\n\nQuestion: {question}\nAnswer:"
    )
    reply, _ = _ask_ollama(prompt, " ".join(plain), "Ask LLM")
    return reply


def ask(question):
    """Rule-based fallback when ToolAgent picks no tool or is disabled."""
    started = time.time()
    q = question.lower()
    summary = bool(SUMMARY_PATTERN.search(q))
    start, end, label = history_window(q)
    about_temp = bool(TEMP_PATTERN.search(q))
    about_advice = bool(ADVICE_PATTERN.search(q))
    about_history = bool(HISTORY_PATTERN.search(q))

    if about_advice and not summary:
        r = tool_advise_bring_in({}, question)
        return {"answer": r["answer"], "period": "", "events": [], "image": r["image"],
                "used_llm": True, "seconds": round(time.time() - started, 1)}

    facts, plain, events = [], [], []
    if summary or (about_history and not about_temp):
        r = tool_get_packages({"period": label}, question)
        facts.append(r["facts"]); plain.append(r["plain"]); events = r["events"]
    if summary or about_temp:
        r = tool_get_temperature({"period": label}, question)
        facts.append(r["facts"]); plain.append(r["plain"])
    if not facts:
        r = tool_look_at_porch({"question": question}, question)
        return {"answer": r["answer"], "period": "", "events": [], "image": r["image"],
                "used_llm": False, "seconds": round(time.time() - started, 1)}

    answer = compose_answer(question, facts, summary, plain)
    image = next((e["image"] for e in events if e["image"]), "")
    return {"answer": answer, "period": label, "events": events, "image": image,
            "used_llm": answer != " ".join(plain), "seconds": round(time.time() - started, 1)}


def answer_on_phone(question):
    with ask_lock:
        r = ask_smart(question, session="phone")
    image = r["image"] or next((e["image"] for e in r["events"] if e["image"]), "")
    photo = os.path.join(EVENTS_DIR, image) if image else None
    period = f" ({r['period']})" if r["period"] else ""
    title = "Confirm?" if r.get("pending") else f"Answer{period}"
    send_ntfy(title, r["answer"], photo, "speech_balloon", 3, buttons=False)


# --- Tool definitions for ToolAgent ---
PERIODS = ["today", "yesterday", "last 24 hours", "this week", "this month"]
BUZZER_PATTERNS = ["beep", "double", "alarm", "chirp", "off"]


def tool_advise_bring_in(args, question):
    extra = (args.get("package_info") or "").strip()
    answer = advise_package(question, extra_info=extra)
    pkg = runtime["active_package"]
    return {"answer": answer, "image": _existing_image(pkg.get("image", "") if pkg else "")}


def tool_get_packages(args, question):
    start, end, label = history_window(str(args.get("period") or "last 24 hours"))
    f, p, ev = _gather_packages(start, end, label)
    return {"facts": f, "plain": p, "events": ev, "period": label}


def tool_get_temperature(args, question):
    start, end, label = history_window(str(args.get("period") or "last 24 hours"))
    f, p = _gather_temperature(start, end, label)
    return {"facts": f, "plain": p, "period": label}


def tool_get_status(args, question):
    s = web_status()
    parts = [status_text()]
    t = s["today"]
    parts.append(f"Today: {t['deliveries']} deliveries, {t['picked_up']} picked up, {t['temp_alerts']} temp alerts.")
    if s["cpu_c"] is not None:
        parts.append(f"Pi CPU {temp_text(s['cpu_c'])}.")
    if s["buzzer"]:
        parts.append(f"Buzzer {s['buzzer']['state']}.")
    return {"facts": " ".join(parts), "plain": status_text()}


def tool_look_at_porch(args, question):
    path = save_frame(watcher.grab(), _new_id("_ask"))
    q = (args.get("question") or question or "").strip()
    checking_box = bool(re.search(r"\b(package|box|parcel|delivery|porch|door|outside|there)\b", q, re.I))
    try:
        if checking_box or not q:
            raw = describe_image(path, PACKAGE_PROMPT)
            present, desc = parse_package_response(raw)
            text = f"Yes, I see a package: {desc}" if present else f"No package visible. {desc}"
        else:
            text = describe_image(path, q)
    except Exception as e:
        text = f"I couldn't look at the camera ({e})."
    return {"answer": text, "image": os.path.basename(path)}


def tool_mark_picked_up(args, question):
    if mark_package_picked_up("agent", notify_phone=False):
        return {"answer": "Marked the package as picked up."}
    return {"answer": "There is no package currently marked outside."}


def tool_sound_buzzer(args, question):
    if not buzz(args["pattern"], "agent"):
        raise RuntimeError("no buzzer is connected")
    return {"answer": "Buzzer off." if args["pattern"] == "off" else f"Buzzer: {args['pattern']}."}


def tool_snooze(args, question):
    minutes = args["minutes"]
    _set_snooze(minutes)
    return {"answer": f"Alerts snoozed for {minutes} minutes." if minutes else "Alerts resumed."}


def tool_snapshot(args, question):
    handle_command("snapshot")
    return {"answer": "Taking a snapshot; the photo and description will arrive on your phone."}


def agent_context():
    temp = runtime["temp"]
    pkg = runtime["active_package"]
    return {
        "time": datetime.now().strftime("%a %H:%M"),
        "package_outside": bool(pkg),
        "package_notes": pkg.get("notes", "") if pkg else "",
        "alerts_snoozed": time.time() < runtime["snooze_until"],
        "buzzer": buzzer.state if buzzer else "not connected",
        "outdoor_temp_f": temp["f"] if temp else None,
        "outdoor_humidity": temp["humidity"] if temp else None,
    }


AGENT_ROUTING = """Questions:
- whether to bring a package inside, safety for contents (food, chocolate, medicine, electronics, etc.), or user shares what is in the package -> advise_bring_in (pass any contents/details in package_info)
- past deliveries, pickups, counts, summaries -> get_packages (pick the period asked about)
- outdoor temperature or humidity -> get_temperature
- current status: whether a package is outside, snoozed, buzzer state, Pi CPU temp -> get_status
- what is at the door / porch right now -> look_at_porch
Actions (only when the user asks):
- mark a package as picked up / brought inside -> mark_picked_up
- make a sound, or silence the buzzer -> sound_buzzer
- pause alerts for N minutes (hours x 60), or resume with 0 -> snooze_alerts
- take a picture now -> take_snapshot"""


def build_tool_agent():
    from tool_agent import Tool, ToolAgent

    period = {"type": "string", "enum": PERIODS, "description": "time period"}
    tools = [
        Tool("advise_bring_in",
             "Advise whether to bring a package inside using outdoor temperature, humidity, and optional package info.",
             tool_advise_bring_in,
             {"package_info": {"type": "string", "description": "optional details on what is inside or how long it will sit"}}),
        Tool("get_packages", "Package delivery and pickup history, counts, and summaries over a period.",
             tool_get_packages, {"period": period}),
        Tool("get_temperature", "Outdoor temperature and humidity now or over a period, plus Pi CPU temperature.",
             tool_get_temperature, {"period": period}),
        Tool("get_status", "Current status: whether a package is outside, snoozed, buzzer state, Pi CPU temp.",
             tool_get_status),
        Tool("look_at_porch", "Take a live photo now to see what is at the door or porch right now.",
             tool_look_at_porch, {"question": {"type": "string", "description": "what to look for"}}),
        Tool("mark_picked_up", "Mark the active package as picked up / brought inside.",
             tool_mark_picked_up, action=True, label="mark package picked up"),
        Tool("sound_buzzer", "Sound the buzzer with a pattern, or turn it off.", tool_sound_buzzer,
             {"pattern": {"type": "string", "enum": BUZZER_PATTERNS}}, ("pattern",), action=True,
             describe=lambda a: "turn the buzzer off" if a.get("pattern") == "off"
             else f"sound the buzzer ({a.get('pattern')})", label="sound or stop the buzzer"),
        Tool("snooze_alerts", "Pause package/temperature alerts for some minutes, or resume with 0.", tool_snooze,
             {"minutes": {"type": "integer", "minimum": 0, "maximum": 240}}, ("minutes",), action=True,
             confirm=lambda a: a["minutes"] > 0,
             describe=lambda a: f"snooze alerts for {a.get('minutes')} minutes" if a.get("minutes")
             else "resume alerts", label="snooze or resume alerts (up to 4 hours)"),
        Tool("take_snapshot", "Take a photo now and send it to the phone.", tool_snapshot, action=True,
             label="take a snapshot"),
    ]
    return ToolAgent(tools, agent_context, compose_answer, server="http://localhost:11434", model=TOOL_MODEL,
                     num_thread=VISION_THREADS, log_path=AGENT_LOG_PATH, command_pattern=COMMAND_PATTERN,
                     routing=AGENT_ROUTING, app_name="Package Watcher")


_tool_agent = None


def get_tool_agent():
    global _tool_agent
    if _tool_agent is None:
        _tool_agent = build_tool_agent()
    return _tool_agent


def ask_smart(question, session="web", dry_run=False, model=None):
    started = time.time()
    reason = ""
    if AGENT_ENGINE == "tools":
        try:
            r = get_tool_agent().ask(question, session, dry_run, model)
        except Exception as e:
            r = {"engine": "none", "error": str(e)}
        if r["engine"] == "tools" and r.get("answer"):
            calls = ", ".join(f"{c['tool']}:{c['status']}" for c in r["tool_calls"]) or "no tools"
            print(f"Agent: {question!r} -> [{calls}] {r['answer'][:90]!r} ({r['timing']['total_s']}s)", flush=True)
            return {**r, "used_llm": True, "seconds": round(time.time() - started, 1)}
        reason = (r.get("error") or "no tool chosen")[:80]
    r = ask(question)
    if AGENT_ENGINE == "tools":
        get_tool_agent().remember(session, question, r["answer"])
    return {**r, "engine": "rules", "tool_calls": [], "pending": False, "fallback_reason": reason,
            "model": model or TOOL_MODEL, "timing": {"total_s": round(time.time() - started, 2)},
            "seconds": round(time.time() - started, 1)}


# --- Web timeline ---
SAFE_IMAGE = re.compile(r"^[\w.-]+\.jpg$")


def _local(e):
    return datetime.fromisoformat(e["timestamp"]).astimezone()


def _kind(e):
    outcome = e.get("outcome")
    if outcome in ("delivered", "picked_up"):
        return outcome
    if e.get("mode") == "temp_alert" or e.get("trigger") == "temperature":
        return "temp_alert"
    if e.get("mode") == "snapshot":
        return "snapshot"
    if e.get("mode") == "question":
        return "answer"
    return "delivered" if e.get("package_present") else "snapshot"


def _web_event(e, kind=None):
    return {
        "id": e["id"], "time": _local(e).isoformat(), "kind": kind or _kind(e),
        "reason": e.get("reason", ""), "description": e.get("description", ""),
        "advice": e.get("advice", ""), "notes": e.get("notes", ""),
        "temp_c": e.get("temp_c"), "temp_f": e.get("temp_f"), "humidity": e.get("humidity"),
        "alert": bool(e.get("alert")),
        "image": _existing_image(e.get("image", "")),
    }


def web_events(filter_="all", day=None, limit=200):
    events = sorted(db_all(), key=lambda e: e["timestamp"])
    if day:
        events = [e for e in events if _local(e).strftime("%Y-%m-%d") == day]
    out = []
    for e in reversed(events):
        kind = _kind(e)
        if filter_ != "all" and kind != filter_:
            continue
        out.append(_web_event(e, kind))
        if len(out) >= limit:
            break
    return out


def web_status():
    today = datetime.now().strftime("%Y-%m-%d")
    events = sorted((e for e in db_all() if _local(e).strftime("%Y-%m-%d") == today),
                    key=lambda e: e["timestamp"])
    snoozed = time.time() < runtime["snooze_until"]
    with state_lock:
        pkg = dict(runtime["active_package"]) if runtime["active_package"] else None
    active_view = None
    if pkg:
        active_view = {
            "id": pkg["id"],
            "arrived_at": datetime.fromtimestamp(pkg["arrived_at"]).isoformat(),
            "outside_minutes": int(max(0, time.time() - pkg["arrived_at"]) // 60),
            "description": pkg.get("description", ""),
            "notes": pkg.get("notes", ""),
            "advice": pkg.get("advice", ""),
            "temp_c": pkg.get("temp_c"),
            "temp_f": c_to_f(pkg["temp_c"]) if pkg.get("temp_c") is not None else None,
            "humidity": pkg.get("humidity"),
            "image": _existing_image(pkg.get("image", "")),
        }
    return {
        "active_package": active_view,
        "outside_now": 1 if pkg else 0,
        "snoozed_until": datetime.fromtimestamp(runtime["snooze_until"]).strftime("%H:%M") if snoozed else None,
        "queue": jobs.qsize(),
        "today": {
            "deliveries": sum(1 for e in events if _kind(e) == "delivered"),
            "picked_up": sum(1 for e in events if _kind(e) == "picked_up"),
            "temp_alerts": sum(1 for e in events if _kind(e) == "temp_alert"),
        },
        "last_event": _local(events[-1]).strftime("%H:%M") if events else None,
        "temperature": ({**runtime["temp"], "age_s": int(time.time() - runtime["temp"]["time"])}
                        if runtime["temp"] else None),
        "cpu_c": cpu_temp_c(),
        "buzzer": ({**runtime.get("buzzer", {}), "state": buzzer.state, "volume": round(buzzer.volume, 2)}
                   if buzzer else None),
    }


def load_labels():
    try:
        with open(LABELS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def label_queue():
    events = sorted((e for e in db_all() if e.get("mode") == "auto"), key=lambda e: e["timestamp"])
    out = []
    for e in events:
        image = e.get("image", "")
        if image and os.path.exists(os.path.join(EVENTS_DIR, image)):
            out.append({"id": e["id"], "time": _local(e).isoformat(), "image": image})
    return {"events": out, "labels": load_labels()}


def save_label(event_id, value):
    with db_lock:
        labels = load_labels()
        if value in ("package", "none", "unclear"):
            labels[event_id] = value
        else:
            labels.pop(event_id, None)
        os.makedirs(BASE_DIR, exist_ok=True)
        tmp = LABELS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(labels, f, indent=0, sort_keys=True)
        os.replace(tmp, LABELS_PATH)
    return labels


def thumbnail(name):
    os.makedirs(THUMBS_DIR, exist_ok=True)
    thumb = os.path.join(THUMBS_DIR, name)
    if not os.path.exists(thumb):
        img = cv2.imread(os.path.join(EVENTS_DIR, name))
        if img is None:
            return None
        h, w = img.shape[:2]
        cv2.imwrite(thumb, cv2.resize(img, (360, int(h * 360 / w))), [cv2.IMWRITE_JPEG_QUALITY, 75])
    return thumb


def make_web_handler(web_key):
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import parse_qs, urlparse

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _authed(self, query):
            cookie = self.headers.get("Cookie", "")
            return f"pw_key={web_key}" in cookie or f"rg_key={web_key}" in cookie or query.get("key", [""])[0] == web_key

        def _file(self, path, ctype):
            with open(path, "rb") as f:
                self._send(200, f.read(), ctype)

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if not self._authed(query):
                return self._send(401, "<h3>Package Watcher: add ?key=... from config.json</h3>", "text/html")
            if url.path == "/" and "key" in query:
                return self._send(302, "", "text/plain", {
                    "Set-Cookie": f"pw_key={web_key}; Max-Age=31536000; HttpOnly; SameSite=Strict; Path=/",
                    "Location": "/"})
            if url.path in ("/", "/index.html"):
                return self._file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
            if url.path in ("/label", "/label.html"):
                return self._file(os.path.join(WEB_DIR, "label.html"), "text/html; charset=utf-8")
            if url.path == "/api/label/queue":
                return self._send(200, label_queue())
            if url.path == "/api/status":
                return self._send(200, web_status())
            if url.path == "/api/temperature":
                hours = min(float(query.get("hours", ["24"])[0] or 24), 24 * TEMP_KEEP_DAYS)
                rows = temp_rows(time.time() - hours * 3600)
                return self._send(200, [{"t": int(r["time"]), "c": r["c"], "h": r["humidity"]} for r in rows])
            if url.path == "/api/events":
                return self._send(200, web_events(query.get("filter", ["all"])[0],
                                                  query.get("day", [None])[0] or None))
            if url.path == "/api/live.jpg":
                frame = cv2.resize(watcher.grab(), (960, 720))
                return self._send(200, cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes(),
                                  "image/jpeg")
            for prefix in ("/photos/", "/thumbs/"):
                if url.path.startswith(prefix):
                    name = url.path[len(prefix):]
                    if not SAFE_IMAGE.match(name):
                        return self._send(404, {"error": "not found"})
                    path = os.path.join(EVENTS_DIR, name) if prefix == "/photos/" else thumbnail(name)
                    if path and os.path.exists(path):
                        return self._file(path, "image/jpeg")
                    return self._send(404, {"error": "not found"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            url = urlparse(self.path)
            if not self._authed(parse_qs(url.query)):
                return self._send(401, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                body = {}
            if url.path == "/api/snapshot":
                handle_command("snapshot")
                return self._send(200, {"ok": True, "message": "Snapshot queued"})
            if url.path == "/api/picked_up":
                ok = mark_package_picked_up("dashboard", notify_phone=False)
                return self._send(200, {"ok": ok, **web_status()})
            if url.path == "/api/buzzer":
                if not buzz(str(body.get("action", "")), "dashboard", body.get("volume")):
                    return self._send(400, {"error": "Buzzer unavailable, or that action or volume isn't allowed"})
                time.sleep(0.2)
                return self._send(200, web_status())
            if url.path == "/api/label":
                event_id = str(body.get("id", ""))
                if not re.fullmatch(r"[\w-]+", event_id):
                    return self._send(400, {"error": "bad id"})
                labels = save_label(event_id, body.get("value"))
                return self._send(200, {"ok": True, "count": len(labels)})
            if url.path == "/api/ask":
                question = str(body.get("question", "")).strip()[:300]
                if not question:
                    return self._send(400, {"error": "Type a question"})
                dry_run = bool(body.get("dry_run"))
                model = str(body.get("model") or "") if dry_run else ""
                if model and not re.fullmatch(r"[\w.:-]{1,60}", model):
                    return self._send(400, {"error": "bad model name"})
                session = re.sub(r"[^\w-]", "", str(body.get("session") or "web"))[:40] or "web"
                if not ask_lock.acquire(blocking=False):
                    return self._send(429, {"error": "Still answering the previous question"})
                try:
                    return self._send(200, ask_smart(question, session, dry_run, model or None))
                except Exception as e:
                    print(f"Ask failed: {e}", flush=True)
                    return self._send(500, {"error": f"Couldn't answer: {e}"})
                finally:
                    ask_lock.release()
            if url.path == "/api/snooze":
                try:
                    minutes = int(body.get("minutes", 0))
                except (TypeError, ValueError):
                    return self._send(400, {"error": "minutes must be an integer"})
                if not 0 <= minutes <= 240:
                    return self._send(400, {"error": "minutes must be between 0 and 240"})
                _set_snooze(minutes)
                return self._send(200, web_status())
            self._send(404, {"error": "not found"})

    return Handler


def tailscale_ip():
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5).stdout
        return out.split()[0] if out.split() else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def run_web(web_key):
    from http.server import ThreadingHTTPServer

    for _ in range(30):
        host = tailscale_ip()
        if host:
            break
        time.sleep(5)
    else:
        print("Web dashboard off: no Tailscale address", flush=True)
        return
    server = ThreadingHTTPServer((host, WEB_PORT), make_web_handler(web_key))
    print(f"Web dashboard: http://{host}:{WEB_PORT}/?key=<web_key in config.json>", flush=True)
    server.serve_forever()


# --- Main loop ---
def main():
    global watcher, topics
    topics = load_topics()
    load_state()
    print("Package Watcher starting...", flush=True)
    print("Subscribe to these in the ntfy app:", flush=True)
    print(f"  Alerts:   {topics['alert_topic']}", flush=True)
    print(f"  Commands: {topics['cmd_topic']}", flush=True)

    watcher = Watcher()
    threading.Thread(target=ai_worker, daemon=True).start()
    if topics.get("healthcheck_url"):
        threading.Thread(target=heartbeat, args=(topics["healthcheck_url"],), daemon=True).start()
        print("Heartbeat on", flush=True)
    if TEMP_ENABLED:
        threading.Thread(target=temperature_loop, daemon=True).start()
    if BUZZER_ENABLED:
        start_buzzer()
    if WEB_ENABLED:
        threading.Thread(target=run_web, args=(topics["web_key"],), daemon=True).start()

    commands = queue.Queue()
    threading.Thread(target=listen_commands, args=(commands,), daemon=True).start()
    send_ntfy("Package Watcher is on", "Watching for package deliveries. Send 'status' anytime.",
              tags="package", buttons=False)

    last_event = 0.0
    while True:
        try:
            while not commands.empty():
                handle_command(commands.get_nowait())
                watcher.reset()

            hit = watcher.check()
            if hit and time.time() - last_event >= EVENT_COOLDOWN:
                last_event = time.time()
                frame = pick_sharpest(watcher.burst(hit["frame"]))
                print(f"Motion {hit['motion'] * 100:.0f}% of frame; checking for package...", flush=True)
                enqueue(mode="auto", trigger=hit["trigger"], frame=frame, motion=hit["motion"],
                        timing={"t0": last_event, "detect": round(time.time() - last_event, 2)})
                watcher.reset()
        except Exception as e:
            print(f"Error: {e}", flush=True)
            time.sleep(5)
        time.sleep(1 / CHECK_FPS)


if __name__ == "__main__":
    main()
