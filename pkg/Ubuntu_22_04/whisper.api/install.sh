cd
git clone https://github.com/innovatorved/whisper.api.git
cd whisper.api

# Install ffmpeg for Audio Processing
sudo apt install ffmpeg

# Install Python Package
pip install -r requirements.txt
uvicorn app.main:app --reload

## Get Your token
curl -X 'POST' \
  'https://innovatorved-whisper-api.hf.space/api/v1/users/get_token' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "email": "example@domain.com",
  "password": "password"
}'

# Modify the token and audioFilePath
curl -X 'POST' \
  'http://localhost:8000/api/v1/transcribe/?model=tiny.en.q5' \
  -H 'accept: application/json' \
  -H 'Authentication: e9b7658aa93342c492fa64153849c68b8md9uBmaqCwKq4VcgkuBD0G54FmsE8JT' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@audioFilePath.wav;type=audio/wav'
