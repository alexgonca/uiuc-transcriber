import sys
import json
import torch
import torchaudio
import soundfile as sf
import numpy as np

# PyTorch 2.6+ defaults weights_only=True, which breaks DiariZen's pre-2.6
# checkpoints. Patch torch.load before any imports trigger checkpoint loading.
_orig_load = torch.load
def _load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _load

# System torchaudio defaults to torchcodec which is not installed.
# Patch torchaudio.load to use soundfile instead.
# Must honour frame_offset/num_frames: pyannote-audio calls torchaudio.load
# with chunk boundaries during segmentation, so ignoring those args feeds the
# full audio to every chunk and corrupts the diarization output.
def _sf_load(uri, frame_offset=0, num_frames=-1, *args, **kwargs):
    stop = None if num_frames < 0 else frame_offset + num_frames
    data, sample_rate = sf.read(uri, dtype="float32", always_2d=True,
                                start=frame_offset, stop=stop)
    return torch.from_numpy(data.T), sample_rate
torchaudio.load = _sf_load

from diarizen.pipelines.inference import DiariZenPipeline

audio_file = sys.argv[1]
num_speakers = int(sys.argv[2])
output_path = sys.argv[3]

pipeline = DiariZenPipeline.from_pretrained(
    "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
)
pipeline.min_speakers = num_speakers
pipeline.max_speakers = num_speakers

if torch.cuda.is_available():
    free_gb = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
    batch_size = max(1, min(32, int(free_gb * 2)))
    pipeline._segmentation.batch_size = batch_size
    print(f"DiariZen batch_size set to {batch_size} ({free_gb:.1f}GB free VRAM)")

diar_results = pipeline(audio_file)

segments = [
    {"start": turn.start, "end": turn.end, "speaker": f"SPEAKER_{speaker:02d}"}
    for turn, _, speaker in diar_results.itertracks(yield_label=True)
]

with open(output_path, "w") as f:
    json.dump(segments, f)
