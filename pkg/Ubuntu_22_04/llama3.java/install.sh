cd
git clone https://github.com/mukel/llama3.java.git
cd llama3.java
curl -L -O https://huggingface.co/mukel/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_0.gguf
java --enable-preview --source 21 --add-modules jdk.incubator.vector Llama3.java -i --model Meta-Llama-3.1-8B-Instruct-Q4_0.gguf
