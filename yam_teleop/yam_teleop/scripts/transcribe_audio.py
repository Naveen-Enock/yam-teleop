"""Transcribe per-episode audio using Whisper, splitting at sub-task markers.

Each recorded episode now has a single audio_*.wav with a paired
audio_*.json that lists ``markers`` (sub-task transitions, captured by
foot-pedal presses during teleop). This script transcribes the WAV with
word-level timestamps and slices the resulting word sequence at each
marker to produce one annotation per sub-task segment.

Foot-pedal markers are treated as hard ground-truth boundaries — they
are NOT moved to align with silences or punctuation. The only marker
mutation is collapsing rapid repeats (pedal bounces / accidental
multi-presses within a short window) to the first press, since real
sub-tasks are expected to be at least one second apart. Residual word
boundary errors (a word landing on the wrong side of a press because
of Whisper word-stamp slop) are fixed in post-hoc review.

Usage:
    python -m yam_teleop.scripts.transcribe_audio data/my_dataset
    python -m yam_teleop.scripts.transcribe_audio data/my_dataset --model large-v3
"""

import argparse
import glob
import json
import os
import sys

from faster_whisper import WhisperModel


def _coalesce_rapid_repeats(markers, min_gap_s: float = 1.0):
    """Drop markers that fire within ``min_gap_s`` of the previously kept one.

    Pedal bounces and operator multi-taps register as several presses
    inside a fraction of a second; we keep only the first press of each
    rapid burst. Comparison is against the most recently *kept* marker
    (not the immediately previous one), so a burst of N rapid presses
    collapses to a single boundary regardless of how the individual
    presses are spaced inside the window.

    Returns ``(kept_markers, num_dropped)``.
    """
    if not markers:
        return list(markers), 0
    out = [markers[0]]
    dropped = 0
    for m in markers[1:]:
        if m["wall_time"] - out[-1]["wall_time"] >= min_gap_s:
            out.append(m)
        else:
            dropped += 1
    return out, dropped


def _segment_boundaries(meta: dict, min_marker_gap_s: float = 1.0):
    """Build segment boundaries from episode start/end + foot-pedal markers.

    Markers are coalesced (rapid repeats collapsed to the first press),
    then used as hard ground-truth boundaries — no snapping, no
    movement. Returns ``(boundaries, num_dropped)``.
    """
    raw_markers = meta.get("markers", [])
    coalesced, dropped = _coalesce_rapid_repeats(raw_markers, min_marker_gap_s)

    start_wall = meta["start"]["wall_time"]
    # Always start the first segment at step 0, even if audio recording started a
    # tick later (audio thread startup race can put meta["start"]["step"] at 1).
    # Downstream consumers assume segments cover the full episode range.
    boundaries = [{"step": 0, "audio_t": 0.0}]
    for m in coalesced:
        boundaries.append({
            "step": m["step"],
            "audio_t": max(0.0, m["wall_time"] - start_wall),
        })
    boundaries.append({
        "step": meta["end"]["step"],
        "audio_t": max(0.0, meta["end"]["wall_time"] - start_wall),
    })
    return boundaries, dropped


def _words_in_range(words, t0: float, t1: float):
    """Yield words whose center time falls in [t0, t1)."""
    for w in words:
        center = 0.5 * (w.start + w.end)
        if t0 <= center < t1:
            yield w


def transcribe_episode(model: WhisperModel, episode_dir: str,
                       min_marker_gap_s: float = 1.0) -> list:
    """Transcribe each audio file in one episode dir; split by markers.

    Returns annotations sorted by start_step. Each annotation covers one
    sub-task segment (between two boundaries). Empty segments (e.g.
    silent leading/trailing windows, or operator double-presses) are
    merged into the adjacent non-empty segment so every step in
    ``[start_step, end_step]`` is covered.
    """
    json_files = sorted(glob.glob(os.path.join(episode_dir, "audio_*.json")))
    if not json_files:
        return []

    annotations = []
    for json_path in json_files:
        with open(json_path) as f:
            meta = json.load(f)

        wav_path = os.path.join(episode_dir, meta["audio_file"])
        if not os.path.exists(wav_path):
            print(f"  Warning: {meta['audio_file']} not found, skipping")
            continue

        # Skip dead-mic episodes — Whisper hallucinates "you" / "thanks
        # for watching" out of noise floor, which is worse than no annotation.
        if meta.get("audio_silent"):
            print(f"    [{meta['audio_file']}] flagged audio_silent "
                  f"(peak={meta.get('audio_peak', 0):.4f}); skipping transcription")
            continue

        # Anti-hallucination flags. Whisper (esp. large-v3) confabulates
        # phrases on silent tails and self-primes repetition loops when the
        # same instruction recurs. VAD strips silence before ASR sees it;
        # disabling prev-text conditioning breaks the loop; the thresholds
        # drop low-confidence and repetition-collapsed segments.
        segments_iter, _ = model.transcribe(
            wav_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
        )
        words = []
        for seg in segments_iter:
            if seg.words is None:
                continue
            words.extend(seg.words)
        if not words:
            print(f"  Warning: empty transcription for {meta['audio_file']}")
            continue

        boundaries, dropped = _segment_boundaries(
            meta, min_marker_gap_s=min_marker_gap_s,
        )
        if dropped:
            print(f"    [{meta['audio_file']}] dropped {dropped} rapid-repeat marker(s)")

        # First build raw segments (one per boundary pair), then merge
        # empties into neighbors. Leading empties extend the next segment
        # backward; middle/trailing empties extend the previous segment
        # forward. This guarantees full step-range coverage even when the
        # operator is silent at episode start, double-presses, or finishes
        # speaking before episode end.
        raw = []
        for i in range(len(boundaries) - 1):
            b0, b1 = boundaries[i], boundaries[i + 1]
            seg_words = list(_words_in_range(words, b0["audio_t"], b1["audio_t"]))
            text = "".join(w.word for w in seg_words).strip()
            raw.append({"b0": b0, "b1": b1, "text": text})

        merged = []
        pending_b0 = None  # leading-empty start, consumed by next non-empty
        for r in raw:
            if r["text"]:
                b0 = pending_b0 if pending_b0 is not None else r["b0"]
                merged.append({"b0": b0, "b1": r["b1"], "text": r["text"]})
                pending_b0 = None
            else:
                if not merged:
                    if pending_b0 is None:
                        pending_b0 = r["b0"]
                else:
                    merged[-1]["b1"] = r["b1"]

        for m in merged:
            annotations.append({
                "text": m["text"],
                "start_step": m["b0"]["step"],
                "end_step": m["b1"]["step"],
                "start_time": meta["start"]["wall_time"] + m["b0"]["audio_t"],
                "end_time": meta["start"]["wall_time"] + m["b1"]["audio_t"],
                "audio_file": meta["audio_file"],
            })

    annotations.sort(key=lambda a: a["start_step"])
    return annotations


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio annotations for recorded episodes")
    parser.add_argument("dataset_dir", help="Path to dataset directory")
    parser.add_argument("--model", default="large-v3",
                        help="Whisper model size (default: large-v3)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--compute-type", default="float16",
                        help="Compute type (default: float16)")
    parser.add_argument("--min-marker-gap", type=float, default=1.0,
                        help="Seconds — drop a marker if it fires within "
                             "this much of the previous kept one. Filters "
                             "pedal bounces / accidental multi-presses; "
                             "real sub-tasks are assumed to be at least "
                             "this far apart (default: 1.0).")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Error: {args.dataset_dir} is not a directory")
        sys.exit(1)

    episode_dirs = sorted(
        os.path.dirname(p)
        for p in glob.glob(os.path.join(args.dataset_dir, "**", "episode.hdf5"),
                           recursive=True)
    )
    if not episode_dirs:
        print(f"No episodes found in {args.dataset_dir}")
        sys.exit(1)

    print(f"Found {len(episode_dirs)} episode(s)")
    print(f"Loading Whisper model '{args.model}' on {args.device}...")
    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type)

    total_annotations = 0
    for episode_dir in episode_dirs:
        episode_name = os.path.basename(episode_dir)
        annotations = transcribe_episode(
            model, episode_dir,
            min_marker_gap_s=args.min_marker_gap,
        )

        if not annotations:
            print(f"  {episode_name}: no audio files")
            continue

        out_path = os.path.join(episode_dir, "language_annotations.json")
        with open(out_path, "w") as f:
            json.dump(annotations, f, indent=2)

        total_annotations += len(annotations)
        print(f"  {episode_name}: {len(annotations)} annotation(s)")
        for ann in annotations:
            print(f"    [{ann['start_step']:>5d}-{ann['end_step']:>5d}] \"{ann['text']}\"")

    print(f"\nDone. {total_annotations} annotation(s) across {len(episode_dirs)} episode(s).")


if __name__ == "__main__":
    main()
