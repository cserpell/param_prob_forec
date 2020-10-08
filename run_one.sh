#!/bin/bash
# Run one experiment.
# set -e
set -x

# Reseed the random number generator using script process ID.
RANDOM=$$

OUT_NAME=out-$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 32)
echo "Random generated name: ${OUT_NAME}"

# python main.py --store_data --run_directory "${OUT_NAME}" "$@" > $OUT_NAME.out 2> $OUT_NAME.err
python main.py --run_directory "${OUT_NAME}" "$@" > $OUT_NAME.out 2> $OUT_NAME.err

bzip2 -f $OUT_NAME.err
bzip2 -f $OUT_NAME.out
