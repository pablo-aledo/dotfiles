cd
git clone https://github.com/Lightning-AI/lit-llama
cd lit-llama
pip install -r requirements.txt

python scripts/download.py --repo_id openlm-research/open_llama_7b --local_dir checkpoints/open-llama/7B
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/open-llama/7B --model_size 7B

python generate.py --prompt "Hello, my name is"
