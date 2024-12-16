mkdir ~/nano-llama31
cd    ~/nano-llama31

#Download the official `llama-models` repo, e.g. inside this project's directory is ok:
git clone https://github.com/meta-llama/llama-models.git

#Download a model, e.g. the Llama 3.1 8B (base) model:
cd llama-models/models/llama3_1
chmod u+x download.sh
./download.sh
#You'll have to enter a "URL from the email". For this you have to request access to Llama 3.1 [here](https://llama.meta.com/llama-downloads/). Then when it asks which model, let's enter `meta-llama-3.1-8b`, and then again one more time `meta-llama-3.1-8b` to indicate the base model instead of the instruct model. This will download about 16GB of data into `./Meta-Llama-3.1-8B` - 16GB because we have ~8B params in 2 bytes/param (bfloat16).

#Now we set up our environment, best to create a new conda env, e.g.:
conda create -n llama31 python=3.10
conda activate llama31

#This will install the `llama-models` package which we can use to load the model:
cd ../../
pip install -r requirements.txt
pip install -e .

#run the generation script
cd ../
pip install fire
torchrun --nnodes 1 --nproc_per_node 1 reference.py \
    --ckpt_dir llama-models/models/llama3_1/Meta-Llama-3.1-8B \
    --tokenizer_path llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model
