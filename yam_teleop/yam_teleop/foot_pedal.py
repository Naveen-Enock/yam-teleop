"""Multiplexed foot pedal reader using evdev.

Grabs all PCsensor FootSwitch (3553:b001) keyboards so keypresses
don't reach the desktop or other applications.

Supports multiple pedals dispatched by key code:
  - KEY_AUDIO (Right Shift): single-button audio pedal
  - KEY_FAILURE (Left Bracket): 2-button pedal, left = failure
  - KEY_SUCCESS (Right Bracket): 2-button pedal, right = success
"""

import selectors
import threading

import evdev
from evdev import ecodes

VENDOR_ID = 0x3553
PRODUCT_ID = 0xB001

# Well-known key assignments
KEY_AUDIO = ecodes.KEY_RIGHTSHIFT
KEY_FAILURE = ecodes.KEY_LEFTBRACE
KEY_SUCCESS = ecodes.KEY_RIGHTBRACE

_TRACKED_KEYS = frozenset({KEY_AUDIO, KEY_FAILURE, KEY_SUCCESS})


def _find_all_devices():
    """Find all FootSwitch keyboard interfaces by vendor:product ID."""
    devices = []
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        if (
            dev.info.vendor == VENDOR_ID
            and dev.info.product == PRODUCT_ID
            and "Keyboard" in dev.name
        ):
            devices.append(dev)
        else:
            dev.close()
    if not devices:
        raise RuntimeError(
            "No FootSwitch keyboard devices found. "
            "Is it plugged in? Is user in the 'input' group?"
        )
    return devices


class FootPedalHub:
    """Multiplexed reader for all PCsensor FootSwitch pedals.

    Grabs all devices so keypresses don't reach the desktop.
    Dispatches events by key code.
    """

    def __init__(self):
        self._devices = _find_all_devices()
        for dev in self._devices:
            dev.grab()
        print(f"[Pedals] Grabbed {len(self._devices)} FootSwitch device(s)")

        self._closed = False

        # Per-key blocking state
        self._pressed = {}
        self._press_events = {}
        self._release_events = {}
        for code in _TRACKED_KEYS:
            self._pressed[code] = False
            self._press_events[code] = threading.Event()
            self._release_events[code] = threading.Event()

        # Polling: latched key-down codes consumed by poll_press()
        self._pending = set()
        self._pending_lock = threading.Lock()

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        sel = selectors.DefaultSelector()
        for dev in self._devices:
            sel.register(dev, selectors.EVENT_READ)
        try:
            while not self._closed:
                for key, _ in sel.select(timeout=0.1):
                    dev = key.fileobj
                    for event in dev.read():
                        if event.type != ecodes.EV_KEY:
                            continue
                        if event.code not in self._press_events:
                            continue
                        if event.value == 1:  # key down
                            self._pressed[event.code] = True
                            self._press_events[event.code].set()
                            self._release_events[event.code].clear()
                            with self._pending_lock:
                                self._pending.add(event.code)
                        elif event.value == 0:  # key up
                            self._pressed[event.code] = False
                            self._release_events[event.code].set()
                            self._press_events[event.code].clear()
        except OSError:
            pass
        finally:
            sel.close()

    # --- Blocking API (for AudioRecorder) ---

    def wait_for_press(self, key_code, timeout=None):
        """Block until key is pressed. Returns False on timeout."""
        ev = self._press_events.get(key_code)
        if ev is None:
            return False
        ev.clear()
        if self._pressed.get(key_code, False):
            return True
        return ev.wait(timeout=timeout)

    def wait_for_release(self, key_code, timeout=None):
        """Block until key is released. Returns False on timeout."""
        ev = self._release_events.get(key_code)
        if ev is None:
            return False
        ev.clear()
        if not self._pressed.get(key_code, False):
            return True
        return ev.wait(timeout=timeout)

    def is_pressed(self, key_code):
        """Current state of a key."""
        return self._pressed.get(key_code, False)

    # --- Polling API (for teleop loop) ---

    def poll_press(self, key_codes=None):
        """Return a key code if any was pressed since last poll, else None.

        Args:
            key_codes: set of codes to check, or None for all tracked keys.
        """
        with self._pending_lock:
            if key_codes is not None:
                match = self._pending & set(key_codes)
            else:
                match = set(self._pending)
            if match:
                code = match.pop()
                self._pending.discard(code)
                return code
            return None

    def clear_pending(self):
        """Discard any latched presses (call before starting a new phase)."""
        with self._pending_lock:
            self._pending.clear()

    # --- Channel API (drop-in for old FootPedal) ---

    def channel(self, key_code):
        """Return a PedalChannel for a single key code."""
        return PedalChannel(self, key_code)

    # --- Lifecycle ---

    def close(self):
        """Release grabs and close all devices."""
        if self._closed:
            return
        self._closed = True
        for dev in self._devices:
            try:
                dev.ungrab()
            except OSError:
                pass
            dev.close()
        # Unblock all waiters
        for ev in self._press_events.values():
            ev.set()
        for ev in self._release_events.values():
            ev.set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PedalChannel:
    """Single-key view of a FootPedalHub.

    Drop-in replacement for the old FootPedal class.
    """

    def __init__(self, hub, key_code):
        self._hub = hub
        self._key_code = key_code

    def wait_for_press(self, timeout=None):
        """Block until pedal is pressed. Returns False on timeout."""
        return self._hub.wait_for_press(self._key_code, timeout)

    def wait_for_release(self, timeout=None):
        """Block until pedal is released. Returns False on timeout."""
        return self._hub.wait_for_release(self._key_code, timeout)

    def poll_press(self):
        """Edge-triggered: True if pedal was pressed since last poll."""
        return self._hub.poll_press({self._key_code}) is not None

    @property
    def is_pressed(self):
        """Current pedal state."""
        return self._hub.is_pressed(self._key_code)

    def close(self):
        pass  # hub owns the lifecycle

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
