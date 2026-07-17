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
interview-date: 
participants:
  - Alice
  - Bob
prompt: >
  Falantes: Alice, Bob.
  Termos: Universidade de São Paulo, hegemonia, manuscrito.
```

- `audio-file`: filename of your recording (any format FFmpeg can read)
- `language`: [ISO 639-1 language code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) (e.g. `pt` for Portuguese, `en` for English)
- `participants`: list of speaker names in the expected order
- `prompt`: short vocabulary/spelling hints in the target language — speaker names, proper nouns, and jargon. **Keep it to keywords, not narrative sentences** (see [Avoiding prompt hallucination](#avoiding-prompt-hallucination) below).
- `prompt-leak-min-run` *(optional, default `5`)*: minimum run of consecutive words shared with the prompt that is treated as leaked text and removed. Lower it (e.g. `4`) if short leaks slip through.

## Avoiding prompt hallucination

Whisper uses `prompt` to bias its vocabulary, but it is prone to **echoing the prompt verbatim** into the transcript on low-confidence audio (silence, hesitation, crosstalk). The transcriber automatically strips any run of `prompt-leak-min-run` or more consecutive words that matches the prompt, but you can prevent most leakage at the source:

- **Use keywords, not sentences.** Prefer `Falantes: Alice, Bob. Termos: hegemonia, manuscrito.` over a paragraph describing the recording. Narrative prompts with incidental facts are what get regurgitated.
- **Don't rely on the prompt for language.** The `language:` field already pins the language; the prompt only needs vocabulary and spellings.
- **The prompt is optional.** For clean audio you can leave it nearly empty — an empty prompt cannot leak.

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
