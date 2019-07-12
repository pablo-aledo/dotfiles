#!/bin/bash
/bin/bash /usr/local/bin/bgindex.sh &
/usr/bin/python2 /home/docker/recoll-webui/webui-standalone.py -a 0.0.0.0
