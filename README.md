# Room Guardian

A LangGraph agent on a Raspberry Pi 5 that watches a room with the camera and sends alerts to your phone.

- **Fast:** cheap OpenCV motion detection runs all the time. When something moves, the photo goes to your phone in about a second.
- **Recognizes what it sees:** Moondream (a local vision model running in Ollama) describes the scene in one sentence.
- **Only loud when it matters:** a person passing by or something suspicious (mask, weapon, smoke, fire, broken glass) triggers a loud alert. Anything else arrives quietly as "Nothing important".
- **Two-way phone control** through [ntfy](https://ntfy.sh), with no account needed: tap **Take another** or **Snooze 1h**, or send `snapshot`, `status`, `snooze 30`, `resume`, or a question like *"is my laptop on the desk?"*
- **MQTT:** every event is published to `fuzail-pi/guardian/event`, for Node-RED, AWS IoT, or any other subscriber.
- Runs entirely on the Pi. No cloud AI is needed.

## How it works

Optional modes, set at the top of `room_guardian.py`:
- `USE_YOLO`: a YOLO person/animal detector decides alerts, and Moondream is skipped when nothing is found.
- `USE_LLM_ASSESS`: Llama 3.2 1B decides alerts instead of keyword rules (slower, because Ollama swaps models).
- `VISION_SERVERS`: run Moondream on another machine (e.g. a laptop over Tailscale) and fall back to the Pi.

## Hardware

- Raspberry Pi 5 (4 GB or more)
- Pi camera (tested with an OV5647)
- Official 27W USB-C power supply. Running AI models on a weaker supply causes under-voltage reboots.

## Setup

```bash
# System packages (camera + OpenCV come from apt, not pip)
sudo apt install -y python3-picamera2 python3-opencv opencv-data

# Ollama + models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull moondream
ollama pull llama3.2:1b        # only needed for USE_LLM_ASSESS

# Python packages
pip install -r requirements.txt --break-system-packages
#   or use a venv that can see the apt packages:
#   python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt

# Optional: a local MQTT broker for the event feed
sudo apt install -y mosquitto
```

## Run

```bash
LIBCAMERA_LOG_LEVELS=*:ERROR python3 room_guardian.py
```

On the first run, it creates two random ntfy topics and saves them in `~/guardian/config.json`:
- **Alerts topic:** subscribe to it in the ntfy app (iOS or Android).
- **Command topic:** send commands from `https://ntfy.sh/<command-topic>` in a browser.

Treat the topic names like passwords. Anyone who knows them can see your photos and send commands.

### Start at boot

```bash
sudo cp room-guardian.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now room-guardian
journalctl -u room-guardian -f
```

## Commands (send to the command topic)

| Command | What it does |
|---|---|
| `snapshot` | Takes a photo and describes it |
| `status` | Armed/snoozed state, today's event count, last event |
| `snooze 30` | Pauses alerts for 30 minutes (default 60) |
| `resume` | Turns alerts back on |
| any other text | Takes a photo and answers it as a question |

## Data

- Photos: `~/guardian/events/*.jpg` (only the newest 500 are kept)
- Event log: `~/guardian/events.json`
- Neither is part of this repository.

## Tuning

| Setting | Default | Meaning |
|---|---|---|
| `MOTION_THRESHOLD` | `0.02` | Fraction of the frame that must change. Raise it if shadows cause events |
| `EVENT_COOLDOWN` | `60` | Seconds between automatic events |
| `VISION_THREADS` | `2` | CPU threads for Moondream (4 is faster but needs the 27W supply) |
| `SUSPICIOUS_PATTERN` | — | Words that mark an event as suspicious |
