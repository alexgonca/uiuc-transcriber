import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore", message=r"\ntorchcodec is not installed")
warnings.filterwarnings("ignore", message=r"TensorFloat-32")
warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0")
class _SuppressFilter(logging.Filter):
    _patterns = ["automatically upgraded your loaded checkpoint"]
    def filter(self, record):
        return not any(p in record.getMessage() for p in self._patterns)
logging.getLogger().addFilter(_SuppressFilter())
print(f"memory = {int(os.environ.get('MEM_LIMIT'))/(1024**3)}GB")
print(f"cores  = {os.environ.get('CPU_LIMIT')}")
import torch
print("PyTorch Version:", torch.__version__)
print("GPU Available:", torch.cuda.is_available())

import yaml
import whisperx
from whisperx.diarize import DiarizationPipeline
import gc
import torch

# --- 1. CONFIGURATION ---
if len(sys.argv) < 2:
    print("Usage: transcribe.py <audio-folder>")
    sys.exit(1)

audio_folder = sys.argv[1]
config_path = os.path.join(audio_folder, "session.yml")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

original_audio = os.path.join(audio_folder, cfg["audio-file"])
_tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".local", "tmp")
os.makedirs(_tmp_dir, exist_ok=True)
audio_file = os.path.join(_tmp_dir, os.path.splitext(os.path.basename(original_audio))[0] + ".wav")
language_code = cfg["language"]
num_speakers = len(cfg["participants"])
my_prompt = cfg["prompt"]
transcript_path = os.path.join(audio_folder, "transcript.txt")

MIN_FREE_VRAM_GB = 4.0

if torch.cuda.is_available():
    best_idx, best_free_gb, best_total_gb = 0, 0.0, 0.0
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        print(f"gpu    = {i} ({total_gb:.1f}GB total, {free_gb:.1f}GB free)")
        if free_gb > best_free_gb:
            best_idx, best_free_gb, best_total_gb = i, free_gb, total_gb
    if best_free_gb < MIN_FREE_VRAM_GB:
        print(f"ERROR: No GPU has {MIN_FREE_VRAM_GB}GB free VRAM. Best available: GPU {best_idx} with {best_free_gb:.1f}GB. Try again later.")
        sys.exit(1)
    device_idx = best_idx
    torch.cuda.set_device(device_idx)
    batch_size = max(4, min(32, int(best_free_gb / 2)))
    print(f"Using GPU {device_idx} with {best_free_gb:.1f}GB free — batch_size = {batch_size}")
else:
    device_idx = None
    batch_size = 4
    print(f"batch_size = {batch_size}")

device = "cuda" if device_idx is not None else "cpu"
compute_type = "float16"
settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.yml")
with open(settings_path, "r") as f:
    settings = yaml.safe_load(f)
hf_token = settings["hf_token"]

# --- 1.5 AUTO-CONVERT TO WAV ---
if not os.path.exists(audio_file):
    print("Converting audio to WAV format...")
    os.system(
        f'ffmpeg -y -i "{original_audio}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"'
    )
    print("Conversion complete!")

# --- 2. MODEL SETUP ---
my_asr_options = {"initial_prompt": my_prompt}

# --- 3. TRANSCRIPTION ---
print("Loading transcription model on GPU...")
model = whisperx.load_model(
    "large-v3",
    device,
    compute_type=compute_type,
    language=language_code,
    asr_options=my_asr_options
)

print("Loading audio...")
audio = whisperx.load_audio(audio_file)

print("Transcribing...")
result = model.transcribe(audio, batch_size=batch_size)
print("Transcription complete!")

del model
gc.collect()
torch.cuda.empty_cache()

# --- 4. ALIGNMENT ---
print("Loading alignment model...")
model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)

print("Aligning...")
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

del model_a
gc.collect()
torch.cuda.empty_cache()

# --- 5. DIARIZATION ---
print("Loading diarization model...")
diarize_model = DiarizationPipeline(token=hf_token, device=device)

print("Diarizing (this may take a moment)...")
diarize_segments = diarize_model(audio, min_speakers=num_speakers, max_speakers=num_speakers)

print("Assigning speakers to text...")
result = whisperx.assign_word_speakers(diarize_segments, result)

# --- 6. SAVE RESULTS ---
speakers_found = sorted({seg.get("speaker", "UNKNOWN") for seg in result["segments"]})
participants = cfg["participants"]

# YAML front matter with speaker mapping
front_matter_lines = ["---", "speakers:"]
for i, speaker in enumerate(speakers_found):
    guess = participants[i] if i < len(participants) else "?"
    front_matter_lines.append(f"  {speaker}: {guess}?")
front_matter_lines.append("---")

# Merge consecutive segments from the same speaker into paragraphs
paragraphs = []
for segment in result["segments"]:
    speaker = segment.get("speaker", "UNKNOWN")
    text = segment.get("text", "").strip()
    end = segment.get("end", 0.0)
    if paragraphs and paragraphs[-1]["speaker"] == speaker:
        paragraphs[-1]["text"] += " " + text
        paragraphs[-1]["end"] = end
    else:
        paragraphs.append({"speaker": speaker, "text": text, "end": end})


def _fmt_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:04.1f}"


body_lines = []
for para in paragraphs:
    ts = _fmt_ts(para["end"])
    body_lines.append(f"**{para['speaker']}:** {para['text']} [{ts}]")

transcript = "\n".join(front_matter_lines) + "\n\n" + "\n\n".join(body_lines)
print(transcript)

with open(transcript_path, "w", encoding="utf-8") as f:
    f.write(transcript + "\n")

print(f"\nTranscript saved to {transcript_path}")

os.remove(audio_file)
print("Temporary WAV file removed.")
