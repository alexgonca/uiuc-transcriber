import os
import sys
import json
import re
import difflib
import subprocess
import logging
import time
import warnings
warnings.filterwarnings("ignore", message=r"\ntorchcodec is not installed")
warnings.filterwarnings("ignore", message=r"TensorFloat-32")
warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0")
class _SuppressFilter(logging.Filter):
    _patterns = ["automatically upgraded your loaded checkpoint"]
    def filter(self, record):
        return not any(p in record.getMessage() for p in self._patterns)
logging.getLogger().addFilter(_SuppressFilter())
t_start = time.time()

_mem_bytes = int(os.environ.get('MEM_LIMIT', 0))
_cpu_cores = os.environ.get('CPU_LIMIT', '?')
print(f"memory = {_mem_bytes/(1024**3):.1f}GB")
print(f"cores  = {_cpu_cores}")
# Make torch's device indices match nvidia-smi's (PCI bus order) so the GPU
# we pick via NVML below is the one torch.cuda.set_device() actually selects.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import torch
print("PyTorch Version:", torch.__version__)
print("GPU Available:", torch.cuda.is_available())

import yaml
import pandas as pd
import pypandoc
import whisperx
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
transcript_path = os.path.join(audio_folder, os.path.basename(os.path.abspath(audio_folder)) + ".md")

# Minimum run of consecutive words shared with the prompt to treat as leakage.
# Whisper regurgitates the initial_prompt on low-confidence chunks; real speech
# almost never reproduces this many consecutive words from the prompt verbatim.
PROMPT_LEAK_MIN_RUN = int(cfg.get("prompt-leak-min-run", 5))

MIN_FREE_VRAM_GB = 4.0


def _query_gpu_memory():
    """Return [(index, free_gb, total_gb), ...] read from NVML via nvidia-smi.

    We deliberately avoid torch.cuda.mem_get_info() here: the first CUDA call
    touching a device initializes a context on it, which needs a few hundred MB
    of VRAM. On a shared node a fully-occupied GPU makes that context creation
    OOM, and the exception aborts the whole probe before we can skip that GPU
    and find a free one. nvidia-smi reads the driver's stats without ever
    creating a CUDA context, so a full GPU is just another row we can reject.
    """
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = []
    for line in out.strip().splitlines():
        idx, free_mib, total_mib = (p.strip() for p in line.split(","))
        gpus.append((int(idx), int(free_mib) / 1024, int(total_mib) / 1024))
    return gpus


if torch.cuda.is_available():
    best_idx, best_free_gb, best_total_gb = None, 0.0, 0.0
    for idx, free_gb, total_gb in _query_gpu_memory():
        print(f"gpu    = {idx} ({total_gb:.1f}GB total, {free_gb:.1f}GB free)")
        if free_gb > best_free_gb:
            best_idx, best_free_gb, best_total_gb = idx, free_gb, total_gb
    if best_idx is None or best_free_gb < MIN_FREE_VRAM_GB:
        best_desc = (
            f"GPU {best_idx} with {best_free_gb:.1f}GB"
            if best_idx is not None else "none"
        )
        print(f"ERROR: No GPU has {MIN_FREE_VRAM_GB}GB free VRAM. Best available: {best_desc}. Try again later.")
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

# --- 1.5 AUTO-CONVERT TO WAV ---
if not os.path.exists(audio_file):
    print("Converting audio to WAV format...")
    os.system(
        f'ffmpeg -y -i "{original_audio}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"'
    )
    print("Conversion complete!")

# --- 2. MODEL SETUP ---
my_asr_options = {"initial_prompt": my_prompt, "beam_size": 10}

_TOKEN_RE = re.compile(r"\S+")


def _normalize_token(tok):
    # Lowercase and drop punctuation; keeps accented letters (Unicode \w).
    return re.sub(r"[^\w]", "", tok.lower())


def _normalize_prompt(prompt):
    return [t for t in (_normalize_token(w) for w in _TOKEN_RE.findall(prompt)) if t]


def _strip_prompt_leak(text, prompt_words, min_run):
    """Remove runs of >= min_run consecutive words that match the prompt.

    Handles leaks that appear on their own or embedded inside otherwise-real
    speech, by excising only the matched span rather than dropping the segment.
    """
    if not prompt_words or not text.strip():
        return text
    tokens = _TOKEN_RE.findall(text)
    norm = [_normalize_token(t) for t in tokens]
    matcher = difflib.SequenceMatcher(a=norm, b=prompt_words, autojunk=False)
    drop = set()
    for a, _b, size in matcher.get_matching_blocks():
        if size >= min_run:
            drop.update(range(a, a + size))
    if not drop:
        return text
    return " ".join(tokens[i] for i in range(len(tokens)) if i not in drop)

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
_SAMPLE_RATE = 16000
audio_duration_s = len(audio) / _SAMPLE_RATE

print("Transcribing...")
t_transcribe_start = time.time()
result = model.transcribe(audio, batch_size=batch_size)
t_transcribe_end = time.time()
print("Transcription complete!")

# --- 3.5 STRIP INITIAL-PROMPT LEAKAGE ---
# WhisperX prepends initial_prompt to every VAD chunk in batched mode, so
# low-confidence chunks sometimes echo the prompt verbatim. Remove those runs
# before alignment/diarization so downstream stages only see real speech.
_prompt_words = _normalize_prompt(my_prompt)
_cleaned = 0
for _seg in result["segments"]:
    _orig = _seg.get("text", "")
    _new = _strip_prompt_leak(_orig, _prompt_words, PROMPT_LEAK_MIN_RUN)
    if _new != _orig:
        _cleaned += 1
        _seg["text"] = _new
result["segments"] = [s for s in result["segments"] if s.get("text", "").strip()]
if _cleaned:
    print(f"Removed prompt-leak text from {_cleaned} segment(s).")

del model
gc.collect()
torch.cuda.empty_cache()

# --- 4. ALIGNMENT ---
print("Loading alignment model...")
model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)

print("Aligning...")
t_align_start = time.time()
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
t_align_end = time.time()

del model_a
gc.collect()
torch.cuda.empty_cache()

# --- 5. DIARIZATION ---
print("Diarizing with DiariZen (this may take a moment)...")
_diarize_json = os.path.join(_tmp_dir, "diarization.json")
_base = os.path.dirname(os.path.abspath(__file__))
_diarizen_python = os.path.join(_base, ".local", "diarizen-venv", "bin", "python")
_worker = os.path.join(_base, "diarize_worker.py")
t_diarize_start = time.time()
# The worker is a separate process, so it doesn't inherit the GPU the main
# process selected. Pin it to that GPU via CUDA_VISIBLE_DEVICES: the chosen
# device becomes the worker's only visible GPU (index 0), so DiariZen runs on
# the right card and the worker's own VRAM probe hits a known-free device
# instead of OOM-ing on a full GPU 0. Empty string => CPU (matches main).
_worker_env = os.environ.copy()
_worker_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
_worker_env["CUDA_VISIBLE_DEVICES"] = (
    str(device_idx) if device_idx is not None else ""
)
subprocess.run(
    [_diarizen_python, _worker, audio_file, str(num_speakers), _diarize_json],
    check=True,
    env=_worker_env,
)
t_diarize_end = time.time()
with open(_diarize_json) as f:
    diarize_segments = pd.DataFrame(json.load(f))
os.remove(_diarize_json)

print("Assigning speakers to text...")
result = whisperx.assign_word_speakers(diarize_segments, result)

# --- 6. SAVE RESULTS ---
for seg in result["segments"]:
    if not isinstance(seg.get("speaker"), str):
        seg["speaker"] = "UNKNOWN"
speakers_found = sorted({seg.get("speaker", "UNKNOWN") for seg in result["segments"]})
participants = cfg["participants"]

def _fmt_dur(seconds):
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


t_end = time.time()
dur_transcribe = t_transcribe_end - t_transcribe_start
dur_align = t_align_end - t_align_start
dur_diarize = t_diarize_end - t_diarize_start
dur_total = t_end - t_start

_mem_gb = f"{_mem_bytes / (1024**3):.1f}GB" if _mem_bytes else "?"

# YAML front matter with speaker mapping and processing metadata
front_matter_lines = [
    "---",
    "speakers:",
]
for i, speaker in enumerate(speakers_found):
    guess = participants[i] if i < len(participants) else "?"
    front_matter_lines.append(f"  {speaker}: {guess}?")
front_matter_lines += [
    f"audio_duration: \"{_fmt_dur(audio_duration_s)}\"",
    "processing:",
    f"  transcription: \"{_fmt_dur(dur_transcribe)}\"",
    f"  alignment: \"{_fmt_dur(dur_align)}\"",
    f"  diarization: \"{_fmt_dur(dur_diarize)}\"",
    f"  total: \"{_fmt_dur(dur_total)}\"",
    "resources:",
    "  transcription_and_alignment:",
    f"    memory: {_mem_gb}",
    f"    cpu_cores: {_cpu_cores}",
    "  diarization:",
    f"    memory: {_mem_gb}",
    f"    cpu_cores: {_cpu_cores}",
    "---",
]

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

with open(transcript_path, "w", encoding="utf-8") as f:
    f.write(transcript + "\n")

docx_path = os.path.splitext(transcript_path)[0] + ".docx"
pypandoc.convert_file(transcript_path, "docx", outputfile=docx_path)
print(f"\nTranscript saved successfully:\n  {transcript_path}\n  {docx_path}")

os.remove(audio_file)
print("Temporary WAV file removed.")
