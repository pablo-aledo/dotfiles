pip install -r requirements.txt
python preprocess.py prepare-dataset --vocab-size 4096
python train.py
tensorboard --logdir=/out/logs
python sample.py \
    --prompt "Once upon a time" \
    --num_samples 3 \
    --temperature 0.7 \
    --max_new_tokens 500
