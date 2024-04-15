python prepro_tinyshakespeare.py
python train_gpt2.py
make train_gpt2
OMP_NUM_THREADS=8 ./train_gpt2

make test_gpt2cu
./test_gpt2cu
make train_gpt2cu
./train_gpt2cu
python train_gpt2.py --inference_only 1 --write_tensors 0 --sequence_length 1024 --batch_size 4
python train_gpt2.py --inference_only 1 --write_tensors 0 --sequence_length 1024 --batch_size 4 --compile 1
python train_gpt2.py --inference_only 1 --write_tensors 0 --sequence_length 1024 --batch_size 4 --compile 1 --tensorcores 1
