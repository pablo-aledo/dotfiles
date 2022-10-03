cd /usr/local/lib
sudo curl -O https://www.antlr.org/download/antlr-4.10.1-complete.jar
export CLASSPATH="./usr/local/lib/antlr-4.10.1-complete.jar:$CLASSPATH"

sudo apt install antlr4

pip3 install antlr4-python3-runtime 
pip install antlr4-python3-runtime==4.7.2
