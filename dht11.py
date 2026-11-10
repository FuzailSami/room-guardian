#!/usr/bin/env python3
"""
DHT11 temperature/humidity reader for the Raspberry Pi 5.

Uses lgpio edge alerts: the kernel timestamps every edge on the data pin, so the
~26-70 microsecond bit timing is measured exactly even though Python is slow.
(The dht11 kernel overlay often times out on the Pi 5; remove it before using this.)

Test:  python3 dht11.py          # GPIO17 (physical pin 11)
       python3 dht11.py 27       # another GPIO
"""

import sys
import time

import lgpio


class DHT11:
    def __init__(self, gpio=17):
        self.gpio = gpio
        self.chip = self._open_chip()

    @staticmethod
    def _open_chip():
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

    def read_once(self):
        """One measurement. Returns (temperature_c, humidity_percent) or raises ValueError."""
        edges = []
        cb = lgpio.callback(self.chip, self.gpio, lgpio.BOTH_EDGES,
                            lambda chip, gpio, level, tick: edges.append((level, tick)))
        try:
            lgpio.gpio_claim_output(self.chip, self.gpio, 0)  # start signal: hold the line low
            time.sleep(0.02)
            lgpio.gpio_free(self.chip, self.gpio)
            lgpio.gpio_claim_alert(self.chip, self.gpio, lgpio.BOTH_EDGES)  # release + listen
            time.sleep(0.05)  # the full 40-bit reply takes about 5 ms
        finally:
            cb.cancel()
            try:
                lgpio.gpio_free(self.chip, self.gpio)
            except lgpio.error:
                pass
        return self.decode(edges)

    @staticmethod
    def decode(edges):
        """Each bit is a high pulse: ~26 us = 0, ~70 us = 1. Ticks are in nanoseconds."""
        widths, rise = [], None
        for level, tick in edges:
            if level == 1:
                rise = tick
            elif level == 0 and rise is not None:
                widths.append((tick - rise) / 1000)
                rise = None
        if len(widths) < 40:
            raise ValueError(f"only {len(widths)}/40 bits received (check wiring)")
        bits = [1 if w > 50 else 0 for w in widths[-40:]]
        data = [int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, 40, 8)]
        if sum(data[:4]) & 0xFF != data[4]:
            raise ValueError("checksum mismatch (noisy signal)")
        humidity = data[0] + data[1] / 10
        temp = data[2] + (data[3] & 0x7F) / 10
        if data[3] & 0x80:
            temp = -temp
        if not (-20 <= temp <= 60 and 5 <= humidity <= 100):
            raise ValueError(f"implausible reading {temp}C / {humidity}%")
        return round(temp, 1), round(humidity, 1)

    def read(self, retries=5, delay=2.0):
        """Retry a few times: the DHT11 drops readings now and then. Needs >= 1 s between reads."""
        error = None
        for _ in range(retries):
            try:
                return self.read_once()
            except ValueError as e:
                error = e
                time.sleep(delay)
        raise ValueError(f"no valid reading after {retries} tries: {error}")

    def close(self):
        lgpio.gpiochip_close(self.chip)


if __name__ == "__main__":
    pin = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    sensor = DHT11(pin)
    print(f"Reading DHT11 on GPIO{pin} (Ctrl+C to stop)")
    try:
        while True:
            try:
                c, h = sensor.read_once()
                print(f"{c}°C  {round(c * 9 / 5 + 32, 1)}°F  {h}% humidity")
            except ValueError as e:
                print(f"read failed: {e}")
            time.sleep(2)
    except KeyboardInterrupt:
        sensor.close()
