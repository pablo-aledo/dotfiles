# Install dependencies (Python 3.10+)
uv sync

# Download the dataset
wget https://github.com/nathan-barry/tiny-diffusion/releases/download/v2.0.0/data.txt

# Download the trained model weights (if you don't want to train it from scratch)
mkdir -p weights && wget -P weights https://github.com/nathan-barry/tiny-diffusion/releases/download/v2.0.0/{gpt,diffusion}.pt

uv run diffusion.py --train
uv run gpt.py --train

uv run diffusion.py
uv run gpt.py
