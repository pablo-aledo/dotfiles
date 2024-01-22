docker build --build-arg GPU_ENABLED=true -t llama-cpp-python-docker:12.1.1 -f .\docker\llama-cpp-python-docker\Dockerfile .
docker volume create gguf_models
docker run --rm -v gguf_models:/vol -v C:\lmstudio\models\TheBloke\dolphin-2.6-mistral-7B-GGUF:/src alpine cp /src/dolphin-2.6-mistral-7b.Q5_0.gguf /vol/
docker build --build-arg GPU_ENABLED=true -f docker/test-app/Dockerfile -t test_app .
docker run -d --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 -e MODEL=/var/model/dolphin-2.6-mistral-7b.Q5_0.gguf -v gguf_models:/var/model -t test_app
