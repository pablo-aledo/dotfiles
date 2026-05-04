sudo wget https://s3.amazonaws.com/releases.lmstudio.ai/prerelease/LM+Studio-0.2.8-beta-v1.AppImage -O /usr/bin/lmstudio
sudo chmod +x /usr/bin/lmstudio

# curl -fsSL https://lmstudio.ai/install.sh | bash
# lms daemon up
# lms --help
# vim ~/.lmstudio/settings.json
# lms get google/gemma-4-e2b
# lms ls
# lms load
# lms server start --port 4000
# socat TCP4-LISTEN:4001,fork,reuseaddr TCP4:127.0.0.1:4000
# curl http://localhost:4000/v1/models
# lms server stop
