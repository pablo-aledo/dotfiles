FROM centos:centos7

MAINTAINER Andy Wong <pslandywong@hotmail.com>

# python3 -m speech_recognition

RUN \
yum install epel-release -y &&\
yum clean all && yum update -y

RUN \
yum install portaudio-devel python34-pip mlocate bash-completion git wget sox libtool autoconf bison swig python34-devel python34-numpy python34-scipy python-devel doxygen alsa-lib-devel pulseaudio-libs-devel make -y

#RUN \
#pip3 install --upgrade pip

RUN \
pip3 install pyaudio &&\
pip3 install SpeechRecognition

RUN \
updatedb
