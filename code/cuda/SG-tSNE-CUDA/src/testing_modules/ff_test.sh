#!/bin/bash
for ng in 10 75 100 150 200
do
                ./yoff $ng 1 20 >> ff1Dg.txt

done

echo All ff1Dg

for ng in 10 75 100 150 200
do
                ./yoff $ng 2 20 >> ff2Dg.txt

done

echo All ff2Dg

for ng in 10 75 100 150 200
do
                ./yoff $ng 3 20 >> ff3Dg.txt

done

echo All ff3Dg
