#!/bin/bash

mkdir -p .local/

echo "Enter your Hugging Face token:"
read -rs hf_token
echo
cat > .local/config.ini <<EOF
[credentials]
hf_token = $hf_token
EOF
echo "Token saved to .local/config.ini"

echo "1. Creating virtual environment..."
python3 -m venv .local/venv/ --system-site-packages

echo "2. Installing WhisperX..."
.local/venv/bin/pip install whisperx

echo "3. Removing duplicate PyTorch packages to use system versions..."
.local/venv/bin/pip uninstall -y torch torchvision torchaudio triton nvidia-nccl-cu12 torchcodec

echo "4. Installing FFMPEG..."

# 1. Create a local bin directory
mkdir -p .local/bin

# 2. Download the official static build of FFmpeg for Linux x86_64
wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz

# 3. Extract the downloaded archive
tar -xf ffmpeg-release-amd64-static.tar.xz

# 4. Move the executable files to your local bin directory
cp ffmpeg-*-amd64-static/ffmpeg .local/bin/
cp ffmpeg-*-amd64-static/ffprobe .local/bin/

# 5. Clean up the downloaded files to save space
rm -rf ffmpeg-release-amd64-static.tar.xz ffmpeg-*-amd64-static/

echo "Done!"