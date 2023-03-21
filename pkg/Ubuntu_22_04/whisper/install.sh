cd ~/tmp
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
bash ./models/download-ggml-model.sh base.en
make
./main -f samples/jfk.wav
