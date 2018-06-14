FROM kaixhin/cuda-torch:latest

RUN sudo apt-get -y install python2.7-dev && \
    sudo apt-get -y install libhdf5-dev && \
    sudo apt-get -y install python-pip && \
    sudo pip install virtualenv

RUN luarocks install torch && \
    luarocks install nn && \
    luarocks install image && \
    luarocks install lua-cjson && \
    luarocks install cutorch && \
    luarocks install cunn && \
    luarocks install cudnn && \
    luarocks install https://raw.githubusercontent.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec


COPY . /fast_neural_style

RUN bash -c "cd /fast_neural_style && models/download_style_transfer_models.sh && models/download_vgg16.sh && virtualenv .env && source .env/bin/activate && pip install -r requirements.txt && deactivate"

WORKDIR /fast_neural_style
