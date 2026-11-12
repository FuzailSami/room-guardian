#!/usr/bin/env python3
"""
Buzzer actuator, controlled over MQTT.

Wiring (GPIO27 = physical pin 13):
  3-pin module:  VCC -> pin 17 (3.3V),  I/O or S -> pin 13,  GND -> pin 14
  2-pin buzzer:  +   -> pin 13,                              -  -> pin 14

Buzzer types:
  active  - beeps by itself when powered (has a sealed back / sticker on top)
  passive - needs a tone signal (PWM); silent when just powered
Some modules are "low-level trigger": they sound when the pin is LOW (active_low=True).

Test which settings make yours sound:
  python3 actuators.py            # tries active, active-low, passive, passive-low (2 s pause each)
  python3 actuators.py alarm      # plays a pattern with the settings in room_guardian.py
  python3 actuators.py volume     # beeps at 100%, 60%, 40%, 25%, 15% to pick BUZZER_VOLUME
  python3 actuators.py tones      # passive buzzer: beeps at 1000-5000 Hz to find its loudest tone

A passive buzzer is loudest at its resonant frequency (often 2-4 kHz): set BUZZER_TONE_HZ to it.

Volume: the buzzer is switched on and off very fast (PWM at VOLUME_PWM_HZ); a lower duty cycle
gives it less power, so it sounds quieter. It can be changed while running with set_volume().
"""

import sys
import threading
import time

import lgpio

VOLUME_PWM_HZ = 10000  # fast enough to blend into a steady (quieter) tone
PATTERNS = {  # (on seconds, off seconds) pairs
    "beep": [(0.15, 0)],
    "double": [(0.12, 0.1), (0.12, 0)],
    "alarm": [(0.4, 0.2)] * 8,  # about 5 seconds
    "chirp": [(0.05, 0.05)] * 3,
}


def open_rp1_chip():
    """Header GPIOs are on the RP1 chip: gpiochip0 on newer kernels, gpiochip4 on older ones."""
    for n in (0, 4):
        try:
            handle = lgpio.gpiochip_open(n)
        except lgpio.error:
            continue
        if "rp1" in str(lgpio.gpio_get_chip_info(handle)[-1]).lower():
            return handle
        lgpio.gpiochip_close(handle)
    return lgpio.gpiochip_open(0)


class Buzzer:
    def __init__(self, gpio=27, passive=False, active_low=False, tone_hz=2000, max_seconds=30, volume=1.0):
        self.gpio, self.passive, self.active_low = gpio, passive, active_low
        self.tone_hz, self.max_seconds = tone_hz, max_seconds
        self.volume = min(max(float(volume), 0.05), 1.0)
        self.chip = open_rp1_chip()
        lgpio.gpio_claim_output(self.chip, gpio, self._level(False))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.state = "off"  # off | on | <pattern name>

    def _level(self, on):
        return int(on) ^ int(self.active_low)

    def _set(self, on, hz=None):
        if self.passive:  # square wave at the tone; 50% duty is loudest, a narrower pulse is quieter
            lgpio.tx_pwm(self.chip, self.gpio, hz or self.tone_hz, 50 * self.volume if on else 0)
        else:  # fast PWM: the buzzer only gets part of the power (100% = full volume)
            # Always PWM, even at full volume, so changing the volume never mixes PWM and plain writes
            duty = 100 * self.volume if on else 0
            lgpio.tx_pwm(self.chip, self.gpio, VOLUME_PWM_HZ, 100 - duty if self.active_low else duty)

    def _run(self, steps, name):
        try:
            deadline = time.time() + self.max_seconds
            for i, (on_s, off_s) in enumerate(steps):
                if self.stop_event.is_set() or time.time() > deadline:
                    break
                # passive buzzer: the alarm alternates two tones, like a siren
                hz = self.tone_hz * (0.75 if (self.passive and name == "alarm" and i % 2) else 1)
                self._set(True, hz)
                if self.stop_event.wait(on_s):
                    break
                self._set(False)
                if off_s and self.stop_event.wait(off_s):
                    break
        finally:
            self._set(False)
            with self.lock:
                if self.state == name:
                    self.state = "off"

    def play(self, name="beep", seconds=None):
        """Play a pattern (beep, double, alarm, chirp) or 'on' for a steady tone, in the background."""
        if name == "on":
            steps = [(min(seconds or 3, self.max_seconds), 0)]
        elif name in PATTERNS:
            steps = PATTERNS[name]
        else:
            raise ValueError(f"unknown pattern {name!r}")
        self.off()
        with self.lock:
            self.stop_event = threading.Event()
            self.state = name
            self.thread = threading.Thread(target=self._run, args=(steps, name), daemon=True)
            self.thread.start()

    def set_volume(self, volume):
        """0.05-1.0; applies from the next beep (a pattern that is playing picks it up at its next step)."""
        self.volume = min(max(float(volume), 0.05), 1.0)
        return self.volume

    def off(self):
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=1)
        self._set(False)
        self.state = "off"

    def close(self):
        self.off()
        lgpio.gpio_free(self.chip, self.gpio)
        lgpio.gpiochip_close(self.chip)


if __name__ == "__main__":
    gpio = 27
    if len(sys.argv) > 1:  # use the guardian's settings
        sys.path.insert(0, ".")
        from room_guardian import BUZZER_ACTIVE_LOW, BUZZER_GPIO, BUZZER_PASSIVE, BUZZER_TONE_HZ, BUZZER_VOLUME

        if sys.argv[1] == "tones":
            for hz in range(1000, 5001, 500):
                print(f"{hz} Hz ...", flush=True)
                b = Buzzer(BUZZER_GPIO, True, BUZZER_ACTIVE_LOW, tone_hz=hz, volume=1.0)
                b.play("on", seconds=0.6)
                b.thread.join()
                b.close()
                time.sleep(0.8)
            print("\nSet BUZZER_TONE_HZ in room_guardian.py to the loudest one (and BUZZER_PASSIVE = True).")
            sys.exit()

        if sys.argv[1] == "volume":
            for v in (1.0, 0.6, 0.4, 0.25, 0.15):
                print(f"Volume {v} ...", flush=True)
                b = Buzzer(BUZZER_GPIO, BUZZER_PASSIVE, BUZZER_ACTIVE_LOW, tone_hz=BUZZER_TONE_HZ, volume=v)
                b.play("double")
                b.thread.join()
                b.close()
                time.sleep(1.5)
            print("\nSet BUZZER_VOLUME in room_guardian.py to the level you liked.")
            sys.exit()
        b = Buzzer(BUZZER_GPIO, BUZZER_PASSIVE, BUZZER_ACTIVE_LOW, tone_hz=BUZZER_TONE_HZ, volume=BUZZER_VOLUME)
        b.play(sys.argv[1])
        b.thread.join()
        b.close()
        sys.exit()
    for k, (passive, low) in enumerate([(False, False), (False, True), (True, False), (True, True)], 1):
        label = f"{'passive' if passive else 'active'}{', low-trigger' if low else ''}"
        print(f"Test {k}/4: {label} ... ", end="", flush=True)
        b = Buzzer(gpio, passive, low)
        b.play("double")
        b.thread.join()
        b.close()
        print("done. Did it beep twice?")
        time.sleep(2)
    print("\nSet BUZZER_PASSIVE / BUZZER_ACTIVE_LOW in room_guardian.py to the first one that beeped.")
