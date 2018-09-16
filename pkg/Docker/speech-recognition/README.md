# speech-recognition

This is a Centos7 based container for running:

https://pypi.python.org/pypi/SpeechRecognition/

which is a Python speech recognition library.

To manually build:

  docker build -t centos7-speech
  
Example usage:

  docker run -it -v /dev/snd:/dev/snd --privileged centos7-speech /bin/bash
 
and run:

  python3 -m speech_recognition
  
to start the test script taking input from your mic
