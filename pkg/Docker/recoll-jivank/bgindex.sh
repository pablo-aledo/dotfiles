#!/bin/bash
while :
do
	/usr/bin/recollindex
	startTime=$(date +%s)
	endTime=$(date -d "tomorrow 0:00" +%s)
	timeToWait=$(($endTime- $startTime))
	/bin/sleep $timeToWait
done