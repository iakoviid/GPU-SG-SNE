#!/bin/bash
for n in 13 14 15 16 17 18 19 20 21 22 23 24 25
do
                ./grid $n 2 100 20 >> grid2Dwarp64100g.txt

done

echo All grid2Dwarp64100g


for n in 13 14 15 16 17 18 19 20 21 22 23 24 25
do
                ./grid $n 2 150 20 >> grid2Dwarp64150g.txt

done

echo All grid2Dwarp64150g


for n in 13 14 15 16 17 18 19 20 21 22 23 24 25
do
                ./grid $n 2 10 20 >> grid2Dwarp6410g.txt

done

echo All grid2Dwarp6410g

for n in 13 14 15 16 17 18 19 20 21 22 23 24 25
do
                ./grid $n 2 75 20 >> grid2Dwarp6475g.txt

done

echo All grid2Dwarp6475g

for n in 13 14 15 16 17 18 19 20 21 22 23 24 25
do
                ./grid $n 2 200 20 >> grid2Dwarp64200g.txt

done

echo All grid2Dwarp6475g
