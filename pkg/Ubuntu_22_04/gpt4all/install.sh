cd ~/tmp
git clone https://github.com/nomic-ai/gpt4all.git
aria2c 'magnet:?xt=urn:btih:1F11A9691EE06C18F0040E359361DCA0479BCB5A&dn=gpt4all-lora-quantized.bin&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=udp%3A%2F%2Fopentracker.i2p.rocks%3A6969%2Fannounce'
mv gpt4all-lora-quantized.bin gpt4all/chat
