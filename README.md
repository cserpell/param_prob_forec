This is the code related to the paper "Multi-step probabilistic forecasting model using deep learning parametrized distributions".

# Installation

Create a `virtualenv` in order to not clash with local python libraries, and
then install requirements into it:

```bash
python3 -m venv p3
. p3/bin/activate
pip install -r requirements.txt
```

Otherwise, newest version of libraries can be installed using

```bash
pip install pandas sklearn tensorflow tensorflow_probability
```

# Running

The script has many options that can be explored through the `help` option:

```bash
python main.py --help
```

An example run line would be:

```bash
python main.py \
    --run_directory test2 \
    --seed 1 \
    --ensemble_number_models 1 \
    --file_name datasets/inf475/substation_load.csv \
    --fillna_method repeat_daily \
    --input_lags 3 \
    --input_series A_DE_CORDOVA__013 APOQUINDO_____013 LA_REINA______013 \
    --input_steps 48 \
    --output_steps 24 \
    --output_series A_DE_CORDOVA__013 LA_REINA______013 \
    --resolution H \
    --resolution_method sum \
    --train_percentage 0.3 \
    --test_size 1000 \
    --validation_percentage 0.1 \
    --number_splits 4 \
    --split_overlap 0.2 \
    --split_position 2 \
    --evaluation_sampler_number 200 \
    --model paperlstmsampled \
    --preprocess minmax \
    --sequential_mini_step -1 \
    --nn_batch_size 128 \
    --nn_dropout_output -1.0 \
    --nn_dropout_recurrence -1.0 \
    --nn_epochs 10000 \
    --nn_l2_regularizer 0.000541369 \
    --nn_learning_rate 9.50585e-05 \
    --nn_optimizer Adam \
    --nn_output_distribution normal \
    --nn_patience 50 \
    --lstm_layers 1 \
    --lstm_nodes 96
```

Don't hesitate to ask me further help!
