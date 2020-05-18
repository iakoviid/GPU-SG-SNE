#!/bin/bash

echo "Starting program at $(date)"

echo "Running program $0 with $# arguments with pid $$"

for file in "$@"; do
	grep double "$file" > /dev/null 2> /dev/null
	#when pattern is not found grep has exit status 1
	#we redirect stdout and stderr to a null register

	if [ $? -ne 0 ]
       	then
		echo "File $file does not have any double, "
		echo "# doubleless" >> "$file"
	fi
done

