# Cloud9 v3 Dockerfile

This repository contains Dockerfile of Cloud9 IDE for Docker's automated build published to the public Docker Hub Registry.

## Base docker image

- [hub.docker.com](https://hub.docker.com/r/eeacms/cloud9)

## Features

- Custom container workspace directory by ```C9_WORKSPACE``` var (make it easier to link with VOLUME_FROM other container, not just host directory mapping).
- Define extra params to cloud9 by ```C9_EXTRA``` var. For [details](#extra-params)

## Installation

1. Install [Docker](https://www.docker.com/).
2. Install [Docker Compose](https://docs.docker.com/compose/).

## Usage

    docker run -it -d -p 8080:8080 eeacms/cloud9
    
You can add a workspace as a volume directory with the argument *-v /your-path/workspace/:/cloud9/workspace/* like this :

    docker run -it -d -p 8080:8080 -v /your-path/workspace/:/cloud9/workspace/ eeacms/cloud9
    
## Build and run with custom config directory

Get the latest version from github

    git clone https://github.com/eea/eea.docker.cloud9
    cd cloud9/

Build it

    docker build --force-rm=true -t "$USER/cloud9:latest" .
    
And run

    docker run -d -p 8080:8080 -v /your-path/workspace/:/cloud9/workspace/ $USER/cloud9:latest

## Extra params

By ```C9_EXTRA``` it is possible define extra params to cloud9

    --readonly          Run in read only mode
    --auth              Basic Auth username:password
    --collab            Whether to enable collab.

example:

    C9_EXTRA=--collab -a username:password

## Advance Usage

### Run the latest cloud9 sdk version

Get the latest version from github

    git clone https://github.com/eea/eea.docker.cloud9
    cd cloud9/

Run with docker compose:

    docker-compose up -d
    
Example docker-compose.yml:

    ide:
      build: .
      volumes_from:
        - data
      ports:
        - 8081:8080
      environment:
	- C9_WORKSPACE=/data/workspace
    data:
      image: busybox
      volumes:
        - /data/workspace


### Add cloud9 to edit your app files
    
    webapp:
      image: nginx
      volumes_from:
        - data
    ide:
      image: eeacms/cloud9
      volumes_from:
        - data
      ports:
        - 8081:8080
      environment:
        - C9_WORKSPACE=/var/www/httpd
    data:
      image: busybox
      volumes:
        - /var/www/httpd

It will set the parameters to:

- Workspace directory at `/data/workspace` linked to VOLUME_FROM `data` container
