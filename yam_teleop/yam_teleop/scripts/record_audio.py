"""Continuous audio recorder with sub-task marker pedal.

Records audio for the full duration of each episode (one WAV per episode).
Foot-pedal presses during the episode are saved as sub-task transition
markers (step + wall_time) in the audio metadata JSON, not as start/stop
triggers. The PyAudio input stream stays open for the recorder's lifetime;
frames are only appended to the episode buffer while an episode is active.

Standalone usage (microphone test):
    python -m yam_teleop.scripts.record_audio
"""

import collections
import json
import os
import threading
import time
import wave
from datetime import datetime

import numpy as np
import pyaudio

from yam_teleop.foot_pedal import FootPedalHub, KEY_AUDIO

SAMPLE_RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024

# Silence detection — used to alert the operator when the wireless mic
# dies / disconnects / is muted. The USB receiver keeps producing samples
# even with no transmitter signal (~noise floor or DC). Any peak below
# `SILENCE_PEAK_THRESHOLD` (normalized 0–1 against int16 full scale) for
# `SILENCE_WINDOW_S` seconds counts as "no signal."
SILENCE_PEAK_THRESHOLD = 0.002    # ≈ 65 raw int16 LSB
SILENCE_WINDOW_S = 5.0


class AudioRecorder:
    """Continuous per-episode audio recorder.

    Reads ``state['step']`` to detect episode start (transition from -1 to
    >=0) and end (transition back to -1). One WAV + JSON is saved per
    episode into ``state['episode_dir']``. Foot-pedal presses while an
    episode is active append ``{step, wall_time}`` to the episode's markers
    list.
    """

    def __init__(self, state: dict, pedal=None):
        """
        Args:
            state: shared dict with keys:
                - 'step': current step count, or -1 between episodes.
                - 'episode': current episode number.
                - 'episode_dir': directory to save audio for current episode.
            pedal: PedalChannel for the marker pedal. If None, the recorder
                creates its own FootPedalHub (standalone mode).
        """
        self._state = state
        self._pedal = pedal
        self._running = False
        self._thread = None
        # Audio capture buffer. The PyAudio callback (PortAudio thread)
        # appends iff _record_active is True; the run loop reads/clears
        # under _frames_lock when finalizing an episode.
        self._frames = []
        self._frames_lock = threading.Lock()
        self._record_active = False
        # Rolling per-chunk peak history for silence detection. Always-on
        # so we can warn at episode start if the mic is already dead.
        self._level_lock = threading.Lock()
        self._chunk_peaks: collections.deque = collections.deque()
        # Sticky "mic has gone silent at some point this session" flag.
        # Set when _maybe_warn_silence or wait_for_audio_check observes
        # silence. The collector reads it via is_audio_dead() to refuse to
        # start a new episode; restart the script to clear it.
        self._audio_dead = False
        # Callback liveness tracking. Distinguishes "device not producing
        # samples" (USB receiver unplugged → PortAudio never invokes us)
        # from "device producing zeros" (transmitter dead but receiver up).
        self._callback_fired = False
        self._last_callback_t = 0.0

    def start(self):
        """Start the background recording thread (idempotent for restart)."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background recording thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def is_audio_dead(self) -> bool:
        """Sticky: True if silence has been observed at any point this session.

        Cleared only by restarting the process. The collector uses this to
        block starting a new episode after the mic dies.
        """
        return self._audio_dead

    def wait_for_audio_check(self, timeout: float = 6.0) -> bool:
        """Block until ``SILENCE_WINDOW_S`` of audio is observed; verdict.

        Returns ``True`` if the rolling-window peak exceeds the silence
        threshold (mic alive). Returns ``False`` if no audio callbacks
        fire (input stream open failed or USB device unplugged), the
        window fills and the peak is below threshold (mic dead, e.g.
        receiver up but transmitter off), or ``timeout`` elapses
        inconclusively. On any False verdict, ``_audio_dead`` is set so
        subsequent ``is_audio_dead()`` calls also block.
        """
        # Fast-fail: PortAudio should invoke our callback within a few
        # chunks (~25ms each). If nothing fires within 1s the input
        # stream isn't producing samples — receiver unplugged, stream
        # open failed silently, or device disappeared.
        time.sleep(1.0)
        if not self._callback_fired:
            print("[Audio] No audio callbacks fired within 1s — input "
                  "stream is not producing samples. Is the USB receiver "
                  "plugged in? Did the stream fail to open?")
            self._audio_dead = True
            return False

        deadline = time.monotonic() + (timeout - 1.0)
        while time.monotonic() < deadline:
            peak = self._recent_peak()
            if peak >= SILENCE_PEAK_THRESHOLD:
                # Print observed peak so the operator can eyeball headroom
                # over the silence threshold — useful for tuning if a
                # wireless receiver gates aggressively or the room is dead.
                print(f"[Audio] Rolling-window peak: {peak:.4f} "
                      f"(threshold: {SILENCE_PEAK_THRESHOLD:.4f}, "
                      f"headroom: {peak / SILENCE_PEAK_THRESHOLD:.1f}x)")
                return True
            if 0.0 <= peak < SILENCE_PEAK_THRESHOLD:
                self._audio_dead = True
                return False
            # peak < 0 → window not yet full
            time.sleep(0.1)
        print(f"[Audio] WARNING: audio level inconclusive within {timeout}s")
        self._audio_dead = True
        return False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        now = time.monotonic()
        self._callback_fired = True
        self._last_callback_t = now
        if self._record_active:
            with self._frames_lock:
                self._frames.append(in_data)
        # Always sample loudness so we can warn at episode start if the
        # mic was already dead. Peak (not RMS) — easier to threshold and
        # robust to short bursts.
        samples = np.frombuffer(in_data, dtype=np.int16)
        if samples.size:
            peak = float(np.max(np.abs(samples))) / 32768.0
            with self._level_lock:
                self._chunk_peaks.append((now, peak))
                cutoff = now - (SILENCE_WINDOW_S + 1.0)
                while self._chunk_peaks and self._chunk_peaks[0][0] < cutoff:
                    self._chunk_peaks.popleft()
        return (None, pyaudio.paContinue)

    def _recent_peak(self) -> float:
        """Max chunk peak within the last SILENCE_WINDOW_S seconds.

        Returns 0.0 if we don't yet have a window's worth of audio.
        """
        now = time.monotonic()
        with self._level_lock:
            recent = [p for t, p in self._chunk_peaks
                      if t >= now - SILENCE_WINDOW_S]
        if len(recent) < 10:  # need a real window before judging
            return -1.0
        return max(recent)

    def _run(self):
        own_hub = None
        if self._pedal is not None:
            pedal = self._pedal
        else:
            own_hub = FootPedalHub()
            pedal = own_hub.channel(KEY_AUDIO)

        pa = pyaudio.PyAudio()
        try:
            try:
                dev = pa.get_default_input_device_info()
                print(f"[Audio] Input device: {dev['name']} "
                      f"(index {dev['index']}, "
                      f"{int(dev['defaultSampleRate'])} Hz default)")
            except Exception:
                pass
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self._audio_callback,
            )
        except Exception as e:
            print(f"[Audio] Failed to open input stream: {e}")
            self._audio_dead = True
            if own_hub is not None:
                own_hub.close()
            pa.terminate()
            return

        try:
            print("[Audio] Ready. Recording per-episode; pedal = sub-task marker.")
            episode_info = None        # dict while an episode is being recorded
            last_active_step = -1      # last step seen with step >= 0

            while self._running:
                step = self._state.get("step", -1)
                episode_dir = self._state.get("episode_dir")
                episode_idx = self._state.get("episode", 0)
                active = step >= 0 and episode_dir is not None

                if active and episode_info is None:
                    # Episode just started — fresh buffer, open the gate.
                    with self._frames_lock:
                        self._frames.clear()
                    self._record_active = True
                    episode_info = {
                        "episode_dir": episode_dir,
                        "episode_idx": episode_idx,
                        "start_step": step,
                        "start_wall": time.time(),
                        "markers": [],
                        "silence_warned": False,
                    }
                    print(f"[Audio] Recording started "
                          f"(episode {episode_idx}, start_step {step})")
                    # If the mic was already dead before the episode started,
                    # the rolling history is silent — warn immediately.
                    self._maybe_warn_silence(episode_info)

                if active:
                    last_active_step = step
                    # Edge-triggered pedal poll → append a sub-task marker.
                    if pedal.poll_press():
                        marker = {"step": step, "wall_time": time.time()}
                        episode_info["markers"].append(marker)
                        print(f"[Audio] Marker @ step={step} "
                              f"(total: {len(episode_info['markers'])})")
                    self._maybe_warn_silence(episode_info)

                if not active and episode_info is not None:
                    # Episode just ended — close the gate, save WAV + JSON.
                    self._record_active = False
                    end_wall = time.time()
                    with self._frames_lock:
                        captured = list(self._frames)
                        self._frames.clear()
                    self._save_episode(pa, episode_info, captured,
                                       last_active_step, end_wall)
                    episode_info = None

                time.sleep(0.05)
        except Exception as e:
            print(f"[Audio] Error: {e}")
        finally:
            self._record_active = False
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            if own_hub is not None:
                own_hub.close()
            pa.terminate()

    def _maybe_warn_silence(self, episode_info):
        """Print a one-shot warning if audio is unhealthy.

        Two failure modes:
          1. Callbacks stopped firing → USB device disappeared (receiver
             unplugged mid-session, kernel dropped the device, etc.).
             Detected by stale ``_last_callback_t``.
          2. Callbacks firing but the rolling window is silent → mic
             battery dead / muted / unplugged at the transmitter.

        The episode-level ``silence_warned`` flag controls *print*
        frequency. The recorder-level ``_audio_dead`` flag is sticky and
        gates the next episode start regardless of subsequent recovery —
        a script restart is the only way to clear it.
        """
        now = time.monotonic()
        if self._callback_fired and (now - self._last_callback_t) > 1.0:
            stale_for = now - self._last_callback_t
            self._audio_dead = True
            if not episode_info.get("silence_warned"):
                print(f"[Audio] !!! WARNING: no audio callbacks for "
                      f"{stale_for:.1f}s — USB receiver unplugged? "
                      f"(this episode will finish; next episode is "
                      f"BLOCKED — relaunch)")
                episode_info["silence_warned"] = True
            return

        peak = self._recent_peak()
        if peak < 0:
            return  # window not full yet
        if peak >= SILENCE_PEAK_THRESHOLD:
            episode_info["silence_warned"] = False
            return
        self._audio_dead = True
        if episode_info.get("silence_warned"):
            return
        print(f"[Audio] !!! WARNING: no audio signal for "
              f"{SILENCE_WINDOW_S:.0f}s (peak={peak:.4f}, "
              f"threshold={SILENCE_PEAK_THRESHOLD:.4f}). "
              f"Mic battery? Mic muted? Mic unplugged? "
              f"(this episode will finish; next episode is BLOCKED — relaunch)")
        episode_info["silence_warned"] = True

    def _save_episode(self, pa, episode_info, frames, end_step, end_wall):
        episode_dir = episode_info["episode_dir"]
        if not os.path.isdir(episode_dir):
            print(f"[Audio] Episode dir gone, skipping save: {episode_dir}")
            return
        if not frames:
            print(f"[Audio] No frames captured for episode "
                  f"{episode_info['episode_idx']}, skipping save")
            return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_filename = f"audio_{timestamp_str}.wav"
        wav_path = os.path.join(episode_dir, wav_filename)
        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pa.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(frames))
        except Exception as e:
            print(f"[Audio] WAV save failed: {e}")
            return

        # Whole-episode peak so downstream tooling (transcribe, future UI)
        # can flag dead-mic episodes without rescanning the WAV.
        all_samples = np.frombuffer(b"".join(frames), dtype=np.int16)
        episode_peak = (float(np.max(np.abs(all_samples))) / 32768.0
                        if all_samples.size else 0.0)
        is_silent = episode_peak < SILENCE_PEAK_THRESHOLD

        metadata = {
            "audio_file": wav_filename,
            "episode": episode_info["episode_idx"],
            "start": {
                "step": episode_info["start_step"],
                "wall_time": episode_info["start_wall"],
            },
            "end": {
                "step": end_step,
                "wall_time": end_wall,
            },
            "duration_s": end_wall - episode_info["start_wall"],
            "sample_rate": SAMPLE_RATE,
            "markers": episode_info["markers"],
            "audio_peak": episode_peak,
            "audio_silent": is_silent,
        }
        json_path = os.path.join(episode_dir, f"audio_{timestamp_str}.json")
        try:
            with open(json_path, "w") as jf:
                json.dump(metadata, jf, indent=2)
        except Exception as e:
            print(f"[Audio] JSON save failed: {e}")
            return

        wav_seconds = len(frames) * CHUNK / SAMPLE_RATE
        suffix = " (SILENT — check mic!)" if is_silent else ""
        print(f"[Audio] Saved {wav_filename} ({wav_seconds:.1f}s, "
              f"{len(metadata['markers'])} marker(s), "
              f"peak={episode_peak:.4f}){suffix}")


# --- Standalone microphone test ---

def main():
    """Continuous mic test: records continuously to one file. Pedal = mark."""
    output_dir = "data/audio"
    os.makedirs(output_dir, exist_ok=True)

    pa = pyaudio.PyAudio()
    frames = []

    def callback(in_data, frame_count, time_info, status):
        frames.append(in_data)
        return (None, pyaudio.paContinue)

    print("Microphone Test (continuous)")
    print("  Recording starts immediately. Press pedal to register a marker.")
    print("  Ctrl+C to stop and save.\n")

    hub = FootPedalHub()
    pedal = hub.channel(KEY_AUDIO)
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=callback,
    )
    start_wall = time.time()
    markers = []
    try:
        while True:
            if pedal.poll_press():
                marker_t = time.time() - start_wall
                markers.append(marker_t)
                print(f">> Marker @ {marker_t:.2f}s "
                      f"(total: {len(markers)})")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(output_dir, f"mic_test_{timestamp_str}.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        duration = len(frames) * CHUNK / SAMPLE_RATE
        print(f"\n>> Saved: {wav_path} ({duration:.1f}s, "
              f"{len(markers)} marker(s))")
        if markers:
            json_path = os.path.join(output_dir, f"mic_test_{timestamp_str}.json")
            with open(json_path, "w") as jf:
                json.dump({"audio_file": os.path.basename(wav_path),
                           "duration_s": duration,
                           "markers_s": markers}, jf, indent=2)
            print(f">> Markers: {json_path}")
        hub.close()
        pa.terminate()


if __name__ == "__main__":
    main()
