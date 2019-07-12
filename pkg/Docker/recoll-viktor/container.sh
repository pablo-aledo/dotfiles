#!/bin/bash
DEFPATH="/home/victorash/Documents/pdf-recoll/"

if [[ $# -eq 0 ]] ; then
    echo 'USAGE: ./container-build.sh start|build|bash []'
    echo ' ... build  [/full/path/to/your/data/collection]'
    echo ' ... build without extra args sets the default path'
    echo " ... default path is $DEFPATH"
        
elif [[ $1 == start ]] ; then
    if [[ $# -eq 2 ]] ; then
        FULL_PATH=$(readlink -f $2)
        echo "The path for the collection is " $FULL_PATH
    else
        FULL_PATH=$(readlink -f $DEFPATH)
        echo "The default path for the collection will be" $FULL_PATH
    fi
    #The passed parameter to this script should be the TOP path to your local collection of data. 
    #This will pe passed on to docker, and the volume will be connected to the local mount-point
    

    contID=$(docker run -d --name docker-recoll-webui-pdfocr --mount src="$FULL_PATH",target=/home/docker/data,type=bind docker-recoll-webui-pdfocr)
    echo "ID of newly created container $contID"
    contIP=$(docker inspect -f "{{ .NetworkSettings.IPAddress }}" "$contID")
    echo "IP of container is $contIP"
    echo "Recoll is here: http://$contIP:8080"
    
elif [[ $1 == build ]] ; then
    docker kill docker-recoll-webui-pdfocr
    docker rm docker-recoll-webui-pdfocr
    docker build --rm=false \
            --no-cache \
            --label docker-recoll-webui-pdfocr \
            --tag docker-recoll-webui-pdfocr:latest .
elif [[ $1 == kill ]] ; then
    docker kill docker-recoll-webui-pdfocr
    docker rm docker-recoll-webui-pdfocr
elif [[ $1 == bash ]] ; then
    docker exec -it -u 0 docker-recoll-webui-pdfocr bash
fi


#docker build --label docker-recoll-webui --tag docker-recoll-webui https://github.com/viktor-c/docker-recoll-webui.git
#docker build --no-cache --label docker-recoll-webui --tag docker-recoll-webui .



