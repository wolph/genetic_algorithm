#!/bin/sh

cities=15
args="$cities -g 1000 -s 25 --csv 5 --print-interval 0"

mkdir -p csv images

echo "Warning, processing takes a long time."
echo "On my 3.7GHz Intel Core I7 it took over 2 hours."
echo
echo "Do you want to continue? (ctrl+c to exit)"
read

for elitism in 0 1; do
    for population in 10 20 50 100; do
        for mutation in 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6; do
            for crossover in 0.1 0.3 0.5 0.7 0.9; do
                file="elitism_${elitism}_population_${population}_mutation_${mutation}_crossover_${crossover}"
                echo "Generating $file"
                ./ga.py $args \
                    -e $elitism \
                    -p $population \
                    -m $mutation \
                    -c $crossover \
                    --samples 10 \
                    --csv-file csv/$file.csv

                gnuplot << EOF
                set zlabel 'Path length'
                set xlabel 'Generation'
                set ylabel 'Sample'
                set datafile separator ';'
                set title 'Elitism $elitism, population: $population, mutation: $mutation, crossover: $crossover'
                set terminal png
                set output 'images/$file.png'
                splot 'csv/$file.csv' with pm3d
EOF

            done
        done
    done
done

