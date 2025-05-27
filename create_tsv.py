import os
from datetime import datetime
from enum import Enum
import audioread
import chardet

class NarrationType(Enum):
    TRAUMATIC = 'Traumatic'
    SAD = 'Sad'
    NEUTRAL = 'Neutral'

def get_audio_length(file_path):
    try:
        with audioread.audio_open(file_path) as f:
            duration_sec = f.duration
            return duration_sec * 1000
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def find_and_read_text_file(directory):
    text_file_path = None

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                if text_file_path is not None:
                    print(os.path.join(root, file))
                    raise ValueError("More than one text file found.")
                text_file_path = os.path.join(root, file)
                break

    if text_file_path is None:
        raise FileNotFoundError("No text file found.")

    try:
        with open(text_file_path, 'r', encoding='utf-16') as file:
            return file.read()
    except UnicodeError:
        # Detect encoding
        with open(text_file_path, 'rb') as f:
            raw = f.read()
            detected = chardet.detect(raw)
            encoding = detected['encoding']

        print(f"Detected encoding: {encoding}")  # Optional: for debugging

        with open(text_file_path, 'r', encoding=encoding) as file:
            return file.read()

    return content

def get_narration_type_to_time_frames_mapping(audio_directory_path, peak_truma_start_time, txt_path):
    audio_lengths = {NarrationType.TRAUMATIC: None, NarrationType.SAD: None, NarrationType.NEUTRAL: None}

    for root, dirs, files in os.walk(audio_directory_path):
        for file_name in files:
            if file_name.lower().endswith(('.wav', '.mp3')):
                file_path = os.path.join(root, file_name)
                audio_length = get_audio_length(file_path)
                if audio_length:
                    fn_lower = file_name.lower()
                    if "neutral" in fn_lower:
                        audio_lengths[NarrationType.NEUTRAL] = audio_length
                    elif "sad" in fn_lower:
                        audio_lengths[NarrationType.SAD] = audio_length
                    elif "trauma" in fn_lower or "traumatic" in fn_lower or "trumatic" in fn_lower:
                        audio_lengths[NarrationType.TRAUMATIC] = audio_length

    missing_files = [nt for nt, length in audio_lengths.items() if length is None]
    if missing_files:
        raise FileNotFoundError("Missing audio files: " + ", ".join(nt.value for nt in missing_files))

    file_content = find_and_read_text_file(txt_path)

    mapped_narration_type = []
    in_log_frame = False

    for line in file_content.splitlines():
        line = line.strip()
        if line.startswith("*** LogFrame Start ***"):
            in_log_frame = True
        elif line.startswith("*** LogFrame End ***"):
            in_log_frame = False
        elif in_log_frame and line.startswith("SoundFile:"):
            sound_file = line.split(":", 1)[1].strip().lower()
            if 'trauma' in sound_file or 'trumatic' in sound_file:
                mapped_narration_type.append(NarrationType.TRAUMATIC)
            elif 'sad' in sound_file:
                mapped_narration_type.append(NarrationType.SAD)
            elif 'neutral' in sound_file:
                mapped_narration_type.append(NarrationType.NEUTRAL)
            else:
                raise ValueError(f"Invalid SoundFile format: {sound_file}")

    start_time, before_break, break_time = 0, 5000, 30000
##add another type - truma_peak-truma_regular
    narration_type_to_time_frames_mapping = {NarrationType.TRAUMATIC: [], NarrationType.SAD: [], NarrationType.NEUTRAL: []}

    def add_time_frame(narr_type, current_ms, is_first_frame):
        end_offset = 7000 if is_first_frame else 0
        start_ms = current_ms
        end_ms = start_ms + audio_lengths[narr_type] - end_offset
        narration_type_to_time_frames_mapping[narr_type].append((start_ms, end_ms))
        return end_ms + break_time + before_break

    current_ms = start_time
    is_first_frame = True
    for ntype in mapped_narration_type:
        current_ms = add_time_frame(ntype, current_ms, is_first_frame)
        is_first_frame = False

    return narration_type_to_time_frames_mapping, mapped_narration_type

def print_tsv_time_frames(audio_directory, txt_path,peak_truma_start_time, output_tsv=None):
    print(audio_directory)
    narration_mapping, mapped_narration_type = get_narration_type_to_time_frames_mapping(audio_directory,peak_truma_start_time, txt_path)
    index_counters = {NarrationType.TRAUMATIC: 0, NarrationType.SAD: 0, NarrationType.NEUTRAL: 0}

    lines = ["onset\tduration\ttrial_type"]

    for ntype in mapped_narration_type:
        idx = index_counters[ntype]
        start_ms, end_ms = narration_mapping[ntype][idx]
        onset_sec = start_ms / 1000.0
        trial_type_str = ntype.value.lower()
        if trial_type_str == "traumatic":
            trial_type_str = "trauma"
        duration_sec = (end_ms - start_ms) / 1000.0
        lines.append(f"{onset_sec}\t{duration_sec}\t{trial_type_str}")
        index_counters[ntype] += 1

    subject_id = os.path.basename(audio_directory).split('\\')[0]
    subject_id = subject_id.replace("Subject_","sub-")
    print("subject_id")
    print(audio_directory)
    print(subject_id)
    output_csv = f"{subject_id}_ses-1.csv"

    with open(output_csv, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"TSV written to {output_csv}")


# Define the root directory
root_dir = r'C:\Users\User\Downloads\subjects'
subjects = ["672"]
for subject in subjects:
    print_tsv_time_frames(
       'C:\\Users\\User\\Downloads\\subjects\\' + subject,
        'C:\\Users\\User\\Downloads\\subjects\\' + subject,
    )
# Walk through all subdirectories without printing the root directory itself
#for subdir, dirs, files in os.walk(root_dir):
#    if subdir == root_dir:
#        continue  # skip the root directory
#    print_tsv_time_frames(
#     subdir,
#     subdir,
#    )