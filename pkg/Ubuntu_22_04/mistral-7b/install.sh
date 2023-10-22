mkd ~/mistral-7b
wget https://github.com/mistralai/mistral-src/archive/refs/heads/main.zip
unzip main.zip
cd mistral-src-main
pip install -r requirements.txt
wget https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar
tar -xf mistral-7B-v0.1.tar
python -m main demo mistral-7B-v0.1/
