sudo apt-get install libportaudio2

cd
git clone https://github.com/nazdridoy/kokoro-tts.git
cd kokoro-tts
pip install soundfile sounddevice kokoro_onnx ebooklib beautifulsoup4
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx

echo 'hello world' > input.txt
./kokoro-tts input.txt output.wav --speed 1.2 --lang en-us --voice af_sarah
