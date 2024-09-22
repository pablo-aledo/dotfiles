cd
git clone https://github.com/mukel/llama2.java.git
cd llama2.java
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
javac --enable-preview -source 21 --add-modules=jdk.incubator.vector Llama2.java
java --enable-preview --add-modules=jdk.incubator.vector Llama2 stories15M.bin
