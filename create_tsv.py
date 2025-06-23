import os
from enum import Enum
import audioread
import chardet

# ------------- constants -------------
WINDOW_DURATION_MS             = 30_000    # 30-s analysis window
PRE_BREAK_GAP_MS               = 5_000     # silence before each break
BREAK_DURATION_MS              = 30_000    # silence between clips
FIRST_CLIP_END_TRIM_MS         = 7_000     # trim 7 s off end of first clip

MISSING_HEAD_SECONDS           = 10        # seconds lost if first clip is cut
START_DELAY_SECONDS_IF_INTACT  = 7         # run begins 7 s late if nothing is cut

class NarrationType(Enum):
    TRAUMATIC = "Traumatic"
    SAD       = "Sad"
    NEUTRAL   = "Neutral"

# ------------- helpers -------------
def get_audio_length_ms(file_path: str) -> int | None:
    """Return clip duration in milliseconds, or None on error."""
    try:
        with audioread.audio_open(file_path) as handle:
            return int(handle.duration * 1000)
    except Exception as err:
        print(f"Error reading {file_path}: {err}")
        return None

def read_single_text_file(directory: str) -> str:
    """Return contents of the single .txt file in a directory tree."""
    text_path = None
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".txt"):
                if text_path is not None:
                    raise ValueError("More than one text file found.")
                text_path = os.path.join(root, file_name)
    if text_path is None:
        raise FileNotFoundError("No .txt file found.")

    try:
        with open(text_path, "r", encoding="utf-16") as fh:
            return fh.read()
    except UnicodeError:
        with open(text_path, "rb") as fh:
            raw_bytes = fh.read()
        encoding = chardet.detect(raw_bytes)["encoding"]
        with open(text_path, "r", encoding=encoding) as fh:
            return fh.read()

# ------------- core logic -------------
def build_window_mapping(
    audio_directory: str,
    text_directory: str,
    peak_window_offset_sec: int,
    first_clip_is_cut: bool = True,
):
    """
    Build a mapping:
        {NarrationType: [(window_start_ms, window_end_ms), …]}
    and return it together with the presentation order list.
    """
    # 1 ─ collect clip lengths
    clip_lengths_ms = {nt: None for nt in NarrationType}
    for root, _, files in os.walk(audio_directory):
        for file_name in files:
            if file_name.lower().endswith((".wav", ".mp3")):
                length_ms = get_audio_length_ms(os.path.join(root, file_name))
                if length_ms is None:
                    continue
                lower = file_name.lower()
                if "neutral" in lower:
                    clip_lengths_ms[NarrationType.NEUTRAL]  = length_ms
                elif "sad" in lower:
                    clip_lengths_ms[NarrationType.SAD]      = length_ms
                elif any(k in lower for k in ["trauma", "traumatic", "trumatic"]):
                    clip_lengths_ms[NarrationType.TRAUMATIC] = length_ms

    missing = [nt.value for nt, v in clip_lengths_ms.items() if v is None]
    if missing:
        raise FileNotFoundError("Missing audio clips: " + ", ".join(missing))

    # 2 ─ determine presentation order from LogFrame
    clip_order: list[NarrationType] = []
    log_text = read_single_text_file(text_directory)
    inside_frame = False
    for line in log_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("*** LogFrame Start ***"):
            inside_frame = True
        elif stripped.startswith("*** LogFrame End ***"):
            inside_frame = False
        elif inside_frame and stripped.startswith("SoundFile:"):
            ref = stripped.split(":", 1)[1].strip().lower()
            if any(k in ref for k in ["trauma", "trumatic"]):
                clip_order.append(NarrationType.TRAUMATIC)
            elif "sad" in ref:
                clip_order.append(NarrationType.SAD)
            elif "neutral" in ref:
                clip_order.append(NarrationType.NEUTRAL)
            else:
                raise ValueError(f"Unrecognized SoundFile entry: {ref}")

    # 3 ─ decide timing offsets
    generic_offset_ms      = peak_window_offset_sec * 1000
    first_clip_offset_ms   = generic_offset_ms
    global_timeline_shift  = 0

    if first_clip_is_cut:
        first_clip_offset_ms = max(
            0, (peak_window_offset_sec - MISSING_HEAD_SECONDS) * 1000
        )
    else:
        global_timeline_shift = START_DELAY_SECONDS_IF_INTACT * 1000

    # 4 ─ build windows
    window_mapping = {nt: [] for nt in NarrationType}
    current_timeline_ms = global_timeline_shift
    is_first_clip = True

    for narration_type in clip_order:
        clip_length_ms = clip_lengths_ms[narration_type]
        if is_first_clip:
            clip_length_ms -= FIRST_CLIP_END_TRIM_MS

        clip_start_ms = current_timeline_ms
        clip_end_ms   = clip_start_ms + clip_length_ms

        offset_ms = first_clip_offset_ms if is_first_clip else generic_offset_ms
        window_start_ms = clip_start_ms + offset_ms

        # keep window inside clip bounds
        if window_start_ms + WINDOW_DURATION_MS > clip_end_ms:
            window_start_ms = max(clip_end_ms - WINDOW_DURATION_MS, clip_start_ms)
        window_end_ms = window_start_ms + WINDOW_DURATION_MS

        window_mapping[narration_type].append((window_start_ms, window_end_ms))

        # advance overall timeline
        current_timeline_ms = (
            clip_end_ms + BREAK_DURATION_MS + PRE_BREAK_GAP_MS
        )
        is_first_clip = False

    return window_mapping, clip_order

def write_events_tsv(
    audio_directory: str,
    text_directory: str,
    peak_window_offset_sec: int,
    first_clip_is_cut: bool = True,
):
    """Write a BIDS-style *_events.tsv file for the session."""
    window_mapping, clip_order = build_window_mapping(
        audio_directory,
        text_directory,
        peak_window_offset_sec,
        first_clip_is_cut,
    )

    counters = {nt: 0 for nt in NarrationType}
    tsv_lines = ["onset\tduration\ttrial_type"]

    for narration_type in clip_order:
        idx = counters[narration_type]
        onset_ms, end_ms = window_mapping[narration_type][idx]
        counters[narration_type] += 1

        onset_sec      = onset_ms / 1000.0
        duration_sec   = (end_ms - onset_ms) / 1000.0
        trial_type_tag = (
            "trauma" if narration_type is NarrationType.TRAUMATIC
            else narration_type.value.lower()
        )
        tsv_lines.append(f"{onset_sec:.3f}\t{duration_sec:.3f}\t{trial_type_tag}")

    subject_name = os.path.basename(audio_directory).replace("Subject_", "sub-")
    output_name  = f"sub-{subject_name}_ses-1.tsv"

    with open(output_name, "w", encoding="utf-8") as fh:
        fh.write("\n".join(tsv_lines))
    print(f"Wrote {output_name}")

# ------------- example usage -------------
if __name__ == "__main__":
    root_directory          = r"C:\Users\User\Downloads\subjects"
    participant_ids         = ["672"]
    peak_offset_sec         = 42        # intended offset if nothing is missing
    first_clip_is_cut_flag  = True      # set False when the first 10 s are intact

    for pid in participant_ids:
        participant_path = os.path.join(root_directory, pid)
        write_events_tsv(
            participant_path,
            participant_path,
            peak_offset_sec,
            first_clip_is_cut_flag,
        )
