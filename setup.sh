#!/bin/bash

mkdir -p .local/

if [ -f settings.yml ]; then
    echo "WARNING: settings.yml already exists — skipping creation. Edit it manually if you need to update your token."
else
    cat > settings.yml <<EOF
hf_token: paste-your-hugging-face-token-here
EOF
    chmod 600 settings.yml
    echo "settings.yml created. Open it and replace the placeholder with your Hugging Face token before running transcriptions."
fi

echo "1. Creating virtual environment..."
python3 -m venv .local/venv/ --system-site-packages

echo "2. Installing WhisperX and dependencies..."
.local/venv/bin/pip install whisperx pyyaml pypandoc

echo "3. Removing duplicate PyTorch packages to use system versions..."
.local/venv/bin/pip uninstall -y torch torchvision torchaudio triton nvidia-nccl-cu12 torchcodec

echo "4. Installing FFMPEG..."

mkdir -p .local/bin

if [ -f .local/bin/ffmpeg ]; then
    echo "FFmpeg already installed — skipping."
else
    wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    tar -xf ffmpeg-release-amd64-static.tar.xz
    cp ffmpeg-*-amd64-static/ffmpeg .local/bin/
    cp ffmpeg-*-amd64-static/ffprobe .local/bin/
    rm -rf ffmpeg-release-amd64-static.tar.xz ffmpeg-*-amd64-static/
fi

echo "Done!"
