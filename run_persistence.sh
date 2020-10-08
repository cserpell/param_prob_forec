#!/bin/bash
# Script to run all experiments with persistence model
set -e
set -x

NUM_SPLITS=5
SPLIT_OVERLAP=0.2
TEST_SIZE=1000
VALIDATION_PERCENT=0.1
INPUT_STEPS=48
INPUT_LAGS=24
OUTPUT_STEPS=24
SAMPLES=200
for SPLIT_PART in 1 2 3 4 5; do
    for INPUT_SEED in 1 2 3 4 5 6 7 8 9 10; do
        export PYTHONHASHSEED=$INPUT_SEED
        ./run_one.sh \
            --seed $INPUT_SEED \
            --file_name datasets/inf475/wind_d08.csv \
            --input_series 'Velocidad de viento en 20.0 metros [mean,m/s]' \
            --fillna_method repeat_daily \
            --resolution H \
            --resolution_method mean \
            --number_splits $NUM_SPLITS \
            --split_position $SPLIT_PART \
            --split_overlap $SPLIT_OVERLAP \
            --test_size $TEST_SIZE \
            --validation_percentage $VALIDATION_PERCENT \
            --input_steps $INPUT_STEPS \
            --input_lags $INPUT_LAGS \
            --output_steps $OUTPUT_STEPS \
            --evaluation_sampler_number $SAMPLES \
            --model stochasticpersistence \
            --preprocess standard
        ./run_one.sh \
            --seed $INPUT_SEED \
            --file_name datasets/inf475/consumption_gefcom2014_load.csv \
            --input_series LOAD \
            --fillna_method repeat_daily \
            --resolution H \
            --resolution_method sum \
            --number_splits $NUM_SPLITS \
            --split_position $SPLIT_PART \
            --split_overlap $SPLIT_OVERLAP \
            --test_size $TEST_SIZE \
            --validation_percentage $VALIDATION_PERCENT \
            --input_steps $INPUT_STEPS \
            --input_lags $INPUT_LAGS \
            --output_steps $OUTPUT_STEPS \
            --evaluation_sampler_number $SAMPLES \
            --model stochasticpersistence \
            --preprocess standard
        ./run_one.sh \
            --seed $INPUT_SEED \
            --file_name datasets/inf475/consumption_uci.csv \
            --input_series Global_active_power \
            --fillna_method repeat_daily \
            --resolution H \
            --resolution_method sum \
            --number_splits $NUM_SPLITS \
            --split_position $SPLIT_PART \
            --split_overlap $SPLIT_OVERLAP \
            --test_size $TEST_SIZE \
            --validation_percentage $VALIDATION_PERCENT \
            --input_steps $INPUT_STEPS \
            --input_lags $INPUT_LAGS \
            --output_steps $OUTPUT_STEPS \
            --evaluation_sampler_number $SAMPLES \
            --model stochasticpersistence \
            --preprocess standard
        ./run_one.sh \
            --seed $INPUT_SEED \
            --file_name datasets/inf475/wind_b08.csv \
            --input_series 'Velocidad de viento en 20.0 metros [mean,m/s]' \
            --fillna_method repeat_daily \
            --resolution H \
            --resolution_method mean \
            --number_splits $NUM_SPLITS \
            --split_position $SPLIT_PART \
            --split_overlap $SPLIT_OVERLAP \
            --test_size $TEST_SIZE \
            --validation_percentage $VALIDATION_PERCENT \
            --input_steps $INPUT_STEPS \
            --input_lags $INPUT_LAGS \
            --output_steps $OUTPUT_STEPS \
            --evaluation_sampler_number $SAMPLES \
            --model stochasticpersistence \
            --preprocess standard
    done
done
