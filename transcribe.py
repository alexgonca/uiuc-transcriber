import os
import logging
import warnings
warnings.filterwarnings("ignore", message=r"\ntorchcodec is not installed")
warnings.filterwarnings("ignore", message=r"TensorFloat-32")
warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0")
logging.getLogger("lightning.pytorch.utilities.upgrade_checkpoint").setLevel(logging.ERROR)
print(f"memory = {int(os.environ.get('MEM_LIMIT'))/(1024**3)}GB")
print(f"cores  = {os.environ.get('CPU_LIMIT')}")
import torch
print("PyTorch Version:", torch.__version__)
print("GPU Available:", torch.cuda.is_available())


import configparser
import whisperx
from whisperx.diarize import DiarizationPipeline
import gc
import torch

# --- 1. CONFIGURATION ---
original_audio = "audio1421884911.m4a"
audio_file = original_audio.replace(".m4a", ".wav")
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
        exit(1)
    device_idx = best_idx
    torch.cuda.set_device(device_idx)
    batch_size = max(4, min(32, int(best_free_gb / 2)))
    print(f"Using GPU {device_idx} with {best_free_gb:.1f}GB free — batch_size = {batch_size}")
else:
    device_idx = None
    batch_size = 4
    print(f"batch_size = {batch_size}")
language_code = "pt"
device = "cuda" if device_idx is not None else "cpu"
compute_type = "float16"
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".local", "config.ini"))
hf_token = config["credentials"]["hf_token"]

# --- 1.5 AUTO-CONVERT TO WAV ---
# Converts .m4a to 16kHz mono .wav, bypassing torchcodec
if not os.path.exists(audio_file):
    print("Converting audio to WAV format...")
    os.system(
        f"ffmpeg -y -i {original_audio} -ar 16000 -ac 1 -c:a pcm_s16le {audio_file}"
    )
    print("Conversion complete!")

# --- 2. PROMPT & MODEL SETUP ---
my_prompt = (
    "Uma conversa entre duas brasileiras chamadas Renata e Julia Leitão. "
    "Julia é nutricionista e influenciadora. Ela tem falado na internet sobre remédios para perda de peso como Ozempic, Mounjaro e Wegovy. "
    "Renata é doutoranda em comunicação na University of Illinois Urbana-Champaign. "
    "Elas conversaram sobre credibilidade de criadores de conteúdo ao falar sobre remédios para perda de peso na internet."
)

my_asr_options = {
    "initial_prompt": my_prompt
}

# --- 3. TRANSCRIPTION ---
print("Loading transcription model on GPU...")
model = whisperx.load_model(
    "large-v3",
    device,
    compute_type=compute_type,
    language=language_code, # Now it won't waste time guessing!
    asr_options=my_asr_options
)

print("Loading audio...")
audio = whisperx.load_audio(audio_file)

print("Transcribing...")
result = model.transcribe(audio, batch_size=batch_size)
print("Transcription complete!")

# Free up GPU memory
del model
gc.collect()
torch.cuda.empty_cache()

# --- 4. ALIGNMENT (Maps words to exact millisecond timestamps) ---
print("Loading alignment model...")
model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)

print("Aligning...")
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# Free up GPU memory
del model_a
gc.collect()
torch.cuda.empty_cache()

# --- 5. DIARIZATION (Who is speaking when) ---
print("Loading diarization model...")
diarize_model = DiarizationPipeline(token=hf_token, device=device)

print("Diarizing (this may take a moment)...")
# Note: min_speakers=2 and max_speakers=2 forces the model to look for exactly two people!
diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)

print("Assigning speakers to text...")
result = whisperx.assign_word_speakers(diarize_segments, result)

# --- 6. PRINT RESULTS ---
for segment in result["segments"]:
    speaker = segment.get("speaker", "UNKNOWN")
    text = segment.get("text", "")
    print(f"[{speaker}]: {text}")