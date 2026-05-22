import sys
import json
import torch

# PyTorch 2.6+ defaults weights_only=True, which breaks DiariZen's pre-2.6
# checkpoints. Patch torch.load before any imports trigger checkpoint loading.
_orig_load = torch.load
def _load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _load

from diarizen.pipelines.inference import DiariZenPipeline

audio_file = sys.argv[1]
num_speakers = int(sys.argv[2])
output_path = sys.argv[3]

pipeline = DiariZenPipeline.from_pretrained(
    "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
)
pipeline.min_speakers = num_speakers
pipeline.max_speakers = num_speakers

diar_results = pipeline(audio_file)

segments = [
    {"start": turn.start, "end": turn.end, "speaker": speaker}
    for turn, _, speaker in diar_results.itertracks(yield_label=True)
]

with open(output_path, "w") as f:
    json.dump(segments, f)
