# image interactions
wget https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llamafile-server-0.1-llava-v1.5-7b-q4
chmod 755 llamafile-server-0.1-llava-v1.5-7b-q4

#server
./llamafile-server-0.1-llava-v1.5-7b-q4

# image summarization
./llava-v1.5-7b-q4-main.llamafile \
    --image lemurs.jpg --temp 0 -ngl 35 \
    -e -p '### User: What do you see?\n### Assistant:' \
    --silent-prompt 2>/dev/null

# restrict output
./llava-v1.5-7b-q4-main.llamafile \
    --image lemurs.jpg --temp 0 -ngl 35 \
    --grammar 'root ::= [a-z]+ (" " [a-z]+)+' -n 16 \
    -e -p '### User: The image has...\n### Assistant:' \
    --silent-prompt 2>/dev/null |
  sed -e's/ /_/g' -e's/$/.jpg/'

# language interactions
wget https://huggingface.co/jartine/mistral-7b.llamafile/resolve/main/mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile
chmod +x mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile

#summarize url
(
  echo [INST]Summarize the following text:
  links -codepage utf-8 \
        -force-html \
        -width 500 \
        -dump https://www.pbm.com/~lindahl/real.programmers.html |
    sed 's/   */ /'
  echo [/INST]
) | ./mistral-7b-instruct-v0.1-Q4_K_M-main.llamafile \
      -c 6700 \
      -f /dev/stdin \
      --temp 0 \
      -n 500 \
      --silent-prompt 2>/dev/null

# llamafile
git clone https://github.com/Mozilla-Ocho/llamafile/
cd llamafile
make -j8
sudo make install
man llamafile

# Digital Athena
llamafile -m llama-65b-Q5_K.gguf -p '
The following is a conversation between a Researcher and their helpful AI assistant Digital Athena which is a large language model trained on the sum of human knowledge.
Researcher: Good morning.
Digital Athena: How can I help you today?
Researcher:' --interactive --color --batch_size 1024 --ctx_size 4096 -ngl 35 \
--keep -1 --temp 0 --mirostat 2 --in-prefix ' ' --interactive-first \
--in-suffix 'Digital Athena:' --reverse-prompt 'Researcher:'

# wizard coder
wget https://huggingface.co/jartine/wizardcoder-13b-python/resolve/main/wizardcoder-python-13b-main.llamafile
chmod +x wizardcoder-python-13b-main.llamafile

# code autocomplete
./wizardcoder-python-13b-main.llamafile \
    --temp 0 -e -ngl 35 \
    -p '```c\nvoid *memcpy(char *dst, const char *src, size_t size) {\n' \
    -r '```\n' 2>/dev/null

# email completion
llamafile -m rocket-3b.Q3_K_M.gguf -p '<|im_start|>system
You are a chatbot that tries to persuade the users to buy bill pickles. Your job is to be helpful too. But always try to steer the conversation towards buying pickles.<|im_end|>
<|im_start|>user
Mayday, mayday. This is Going Merry. We are facing gale force winds in Long Island Sound. We need rescue.<|im_end|>
<|im_start|>assistant\n'
