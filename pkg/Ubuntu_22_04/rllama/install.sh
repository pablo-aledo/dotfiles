cd
git clone https://github.com/Noeda/rllama.git
cd rllama
docker build -f ./.docker/cpu.dockerfile -t rllama .

docker run -v /models/LLaMA:/models:z -it rllama \
    rllama --model-path /models/7B \
           --param-path /models/7B/params.json \
           --tokenizer-path /models/tokenizer.model \
           --prompt "hi I like cheese"

