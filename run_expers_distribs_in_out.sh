#!/bin/bash
# Script to run all experiments with random parameters
# set -e
set -x

DATASET_FILES=('datasets/inf475/consumption_gefcom2014_load.csv' 'datasets/inf475/consumption_uci.csv' 'datasets/inf475/wind_b08.csv' 'datasets/inf475/wind_d08.csv' 'datasets/power/eolicas/canela_1.csv')
DATASET_SERIES=('LOAD' 'Global_active_power' 'Velocidad_de_viento_en_20.0_metros_[mean,m/s]' 'Velocidad_de_viento_en_20.0_metros_[mean,m/s]' 'GenerationMW_Canela_1_Eolica')
DATASET_RESOLMETHOD=('sum' 'sum' 'mean' 'mean' 'mean')
NUM_SPLITS=5
SPLIT_OVERLAP=0.2
TEST_SIZE=1000
VALIDATION_PERCENT=0.1
INPUT_STEPS=48
INPUT_LAGS=4
OPTIMIZER='Adam'
BATCH_SIZE=128
SAMPLES=200
# Reseed the random number generator using script process ID.
RANDOM=$$
# Random integer number between 1 and 5 (included)
SPLIT_PART=$((1 + RANDOM % 5))
# 24 steps ahead
OUTPUT_STEPS=24
# Random output distribution
DISTRIBS=(normal weibull log_normal gamma)
DISTRIB=${DISTRIBS[RANDOM % 4]}
# Random input preprocess
PREPROCESS='minmax'
if (( RANDOM % 2 == 0 )); then PREPROCESS='log_minmax'; fi
# Models: all steps, or autoinject
MODELS=(paperlstmsampled autoinjectlstm)
# SEQS=(-1 -1)
SELECTION=$((RANDOM % 2))
MODEL_NAME=${MODELS[SELECTION]}
SEQ_NUM=-1
# SEQ_NUM=${SEQS[SELECTION]}
STATEFUL=
# STATEFUL=--lstm_stateful
# if (( SELECTION != 0 )); then STATEFUL=; fi
# Random integer number between 1 and 3 (included)
LAYERS=$((1 + RANDOM % 3))
# Random integer number between 10 and 200 (included)
UNITS=$((10 + RANDOM % 191))
# Using patience, so use a big number to avoid stopping early
PATIENCE=50
MAX_EPOCHS=100000
# Random float number between 0.00001 and 0.1 (log scale)
LRATE=$(echo | awk '{ srand(); print (10 ^ (-4 * rand() - 1)); }')
# Random float number between 0.0001 and 0.1 (log scale)
DECAY=$(echo | awk '{ srand(); print (10 ^ (-3 * rand() - 1)); }')
if (( RANDOM % 4 == 0 )); then DECAY=-1.0; fi
# Random float number between 0.1 and 0.5 (linear scale)
DROPOUT=$(echo | awk '{ srand(); print 0.4 * rand() + 0.1; }')
if (( RANDOM % 8 == 0 )); then DROPOUT=-1.0; fi
DROPOUT_MC=''
DATASET_SELECTION=$((RANDOM % 5))
# DATASET_SELECTION=$((RANDOM % 8))
DATASET_FILE_NAME=${DATASET_FILES[DATASET_SELECTION]}
DATASET_SERIES_NAME=${DATASET_SERIES[DATASET_SELECTION]}
DATASET_RESOLMETHOD_NAME=${DATASET_RESOLMETHOD[DATASET_SELECTION]}
if (( RANDOM % 2 == 0 )); then DROPOUT_MC='--nn_dropout_no_mc'; fi
for rep in 1 2 3 4 5; do
    # 5 different random seeds
    # INPUT_SEED=$((1 + RANDOM % 10))
    INPUT_SEED=$rep
    export PYTHONHASHSEED=$INPUT_SEED
    ./run_one.sh \
        --seed $INPUT_SEED \
        --file_name $DATASET_FILE_NAME \
        --input_series $DATASET_SERIES_NAME \
        --fillna_method repeat_daily \
        --resolution H \
        --resolution_method $DATASET_RESOLMETHOD_NAME \
        --number_splits $NUM_SPLITS \
        --split_position $SPLIT_PART \
        --split_overlap $SPLIT_OVERLAP \
        --test_size $TEST_SIZE \
        --validation_percentage $VALIDATION_PERCENT \
        --input_steps $INPUT_STEPS \
        --input_lags $INPUT_LAGS \
        --output_steps $OUTPUT_STEPS \
        --evaluation_sampler_number $SAMPLES \
        --model $MODEL_NAME $STATEFUL $DROPOUT_MC \
        --preprocess $PREPROCESS \
        --nn_batch_size $BATCH_SIZE \
        --nn_epochs $MAX_EPOCHS \
        --nn_patience $PATIENCE \
        --nn_learning_rate $LRATE \
        --nn_optimizer $OPTIMIZER \
        --nn_dropout_output $DROPOUT \
        --nn_dropout_recurrence $DROPOUT \
        --nn_l2_regularizer $DECAY \
        --sequential_mini_step $SEQ_NUM \
        --lstm_layers $LAYERS \
        --lstm_nodes $UNITS \
        --nn_output_distribution $DISTRIB
done
