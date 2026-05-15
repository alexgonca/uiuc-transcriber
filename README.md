This repository uses [Illinois Computes Research Notebooks](https://jupyter.ncsa.illinois.edu/) and OpenAI's [Whisper](https://openai.com/index/whisper/) to transcribe texts in different languages.

# Installation

1. Open a terminal on JupyterLab and clone the repository:

```bash
git clone https://github.com/alexgonca/uiuc-transcriber.git
```



# Advanced

If you want to add the transcription virtual environment (venv) to Jupyter Notebook, open a terminal on JupyterLab, navigate to the folder where the venv is installed (in this example below, `whisperx_env`), and execute the following command:

```python
./whisperx_env/bin/python -m ipykernel install --user --name whisperx_env --display-name "WhisperX (System PyTorch)"
```
