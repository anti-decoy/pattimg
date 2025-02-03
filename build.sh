#!/bin/bash
docker build --build-arg all_proxy="http://192.168.1.67:8080" --build-arg new_user=$USER --build-arg user_id=$(id -u) --build-arg group_id=$(id -g) --build-arg gid_pulse=$(getent group pulse | awk -F: '{print $3}') --build-arg gid_pulse_access=$(getent group pulse-access | awk -F: '{print $3}') -t pattimg .
