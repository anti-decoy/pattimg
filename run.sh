#!/bin/bash
docker run -it -p 2222:22/tcp -w /home/$USER -h debian-docker -v /var/log:/var/log -v ~/Downloads:/downloads --privileged -v /dev/video0:/dev/video0 -v /dev/video1:/dev/video1 -v /dev/snd:/dev/snd -v $XDG_RUNTIME_DIR/pulse:$XDG_RUNTIME_DIR/pulse --user $USER --name pattimg pattimg:latest
