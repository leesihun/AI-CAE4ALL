"""Click two screen positions at a scheduled local time.

Install once:
    python -m pip install pyautogui

Run:
    python auto_click.py

Move the mouse to any screen corner to abort while the script is running.
"""

from datetime import datetime
import time

import pyautogui


# Change these three settings before running the script.
TARGET_TIME = "2026-08-10 00:31:00"  # Local time: YYYY-MM-DD HH:MM:SS
# TARGET_TIME = "2026-08-09 22:06:00"  # Local time: YYYY-MM-DD HH:MM:SS
CLICK_POINTS = [
    (1152, 734),   # First button: (x, y)
    (1542, 734),  # Second button: (x, y)
]
DELAY_BETWEEN_CLICKS = 0.5  # Seconds


def wait_until(target: datetime) -> None:
    while True:
        seconds_left = (target - datetime.now()).total_seconds()
        if seconds_left <= 0:
            return
        time.sleep(min(seconds_left, 0.1))


def main() -> None:
    target = datetime.strptime(TARGET_TIME, "%Y-%m-%d %H:%M:%S")
    if target <= datetime.now():
        raise SystemExit(f"Target time has already passed: {TARGET_TIME}")

    pyautogui.FAILSAFE = True
    print(f"Waiting until {target} to click {len(CLICK_POINTS)} places...")
    print("Move the mouse to a screen corner to abort.")

    wait_until(target)

    for click_number, (x, y) in enumerate(CLICK_POINTS, start=1):
        pyautogui.click(x=x, y=y)
        print(f"Clicked place {click_number} at ({x}, {y})")
        if click_number < len(CLICK_POINTS):
            time.sleep(DELAY_BETWEEN_CLICKS)


if __name__ == "__main__":
    main()
