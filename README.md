This repository uses [Illinois Computes Research Notebooks](https://jupyter.ncsa.illinois.edu/), OpenAI's [Whisper](https://openai.com/index/whisper/) for transcription, and [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) for speaker diarization.

# Starting a JupyterLab Session

Do this every time before installing or running a transcription:

1. Visit [https://jupyter.ncsa.illinois.edu/](https://jupyter.ncsa.illinois.edu/).
2. Click **Sign in with CILogon**.
3. For **Environment**, choose **Jupyter - PyTorch**.
4. For **Resource**, choose **H200 141GB VRAM GPU, 10CPU/32GB**.
5. Click **Start**.

# Installation

1. Open a terminal on JupyterLab and clone the repository:

```bash
git clone https://github.com/alexgonca/uiuc-transcriber.git
git config --global --add safe.directory ~/uiuc-transcriber
```

2. Run the setup script:

```bash
cd uiuc-transcriber
./setup.sh
```

This will create two virtual environments (`.local/venv/` for WhisperX and `.local/diarizen-venv/` for DiariZen), install all dependencies, and download a static FFmpeg binary. No accounts or API tokens are required.

# Usage

1. Create a folder for your recording (e.g. `my-interview/`).

2. Place your audio file inside it and create a `session.yml` file with the following structure:

```yaml
audio-file: recording.mp4
language: pt
participants:
  - Alice
  - Bob
prompt: > 
  "Entrevista sobre..."
```

- `audio-file`: filename of your recording (any format FFmpeg can read)
- `language`: [ISO 639-1 language code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) (e.g. `pt` for Portuguese, `en` for English)
- `participants`: list of speaker names in the expected order
- `prompt`: short description of the recording in the target language — helps Whisper stay in the correct language and vocabulary

3. Run the transcription:

```bash
./transcribe.sh my-interview
```

4. The transcript will be saved as `my-interview/my-interview.md` and `my-interview/my-interview.docx`. The Markdown file includes a YAML front matter with speaker mappings that you can adjust manually.

# Models

| Task | Model |
|---|---|
| Transcription | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) |
| Alignment | [jonatasgrosman/wav2vec2-large-xlsr-53-portuguese](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-portuguese) (for `pt`) |
| Diarization | [BUT-FIT/diarizen-wavlm-large-s80-md-v2](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2) |

All models are downloaded automatically on first use. None require authentication.
