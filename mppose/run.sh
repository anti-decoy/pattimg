#!/bin/bash
curdir=$(dirname "$0")
cd "$curdir"

. /home/dec0y/venv/bin/activate

while true; do 
	while true; do
		pid=$(ps ax | grep detect.py | grep -v grep | awk '{print $1}')
		if [ ! -z "$pid" ]; then
			kill "$pid"
			sleep 5s
		else
			break
		fi
	done
	sleep 5s
	while [ $(date +%H) -lt "06" ]; do
		sleep 1m
	done
	python ./detect.py &
	sleep 2h
done
