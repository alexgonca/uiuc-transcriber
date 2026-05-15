This repository uses [Illinois Computes Research Notebooks](https://jupyter.ncsa.illinois.edu/) and OpenAI's [Whisper](https://openai.com/index/whisper/) to transcribe texts in different languages.

# Installation

1. Open a terminal on JupyterLab and clone the repository with these commands:

```bash
git clone https://github.com/alexgonca/uiuc-transcriber.git
git config --global --add safe.directory ~/uiuc-transcriber
```

2. Create Hugging Face access token:
   1. Visit [huggingface.co]. If you don't have a user on the platform, create one. It is free.
   2. Click on your profile (top right) -> Settings -> Access Tokens -> Create new token.
   3. Choose "Fine-grained" (default).
   4. Fill out Token name: 'Diarization'.
   5. Select "Read access to contents of all public gated repos you can access".
   6. Scroll down and click on "Create Token".
   7. Make sure you "copy" the access token. We will need it in the next step of this tutorial.

3. Go back to the JupyterLab terminal. Execute this command: ```./setup.sh``` When it requests the Hugging Face access token, paste the value you copied in the previous step.

4. 

# Advanced

If you want to add the transcription virtual environment (venv) to Jupyter Notebook, open a terminal on JupyterLab, navigate to the folder where the venv is installed (in this example below, `whisperx_env`), and execute the following command:

```python
./whisperx_env/bin/python -m ipykernel install --user --name whisperx_env --display-name "WhisperX (System PyTorch)"
```
