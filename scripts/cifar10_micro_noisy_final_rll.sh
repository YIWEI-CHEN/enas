#!/bin/bash

export PYTHONPATH="$(pwd)"
# All clean
ALL_CLEAN_1="0 1 0 1 0 1 0 0 0 1 1 3 1 1 1 1 1 0 0 2"
ALL_CLEAN_1="$ALL_CLEAN_1 1 0 1 0 0 0 0 2 0 0 0 3 0 1 1 1 0 0 1 0"

ALL_CLEAN_1989="0 1 1 1 0 0 1 3 0 3 1 0 0 0 0 1 0 3 0 1"
ALL_CLEAN_1989="$ALL_CLEAN_1989 0 1 1 0 0 3 1 0 1 0 1 0 1 0 0 1 4 0 1 1"

ALL_CLEAN_2019="1 3 1 0 0 0 0 1 0 0 0 1 0 0 0 2 0 0 0 3"
ALL_CLEAN_2019="$ALL_CLEAN_2019 0 4 1 4 0 1 1 3 1 0 1 4 1 1 0 0 4 2 1 0"

# ALL NOISY
ALL_NOISY_1="0 1 0 0 1 3 1 2 0 1 0 0 0 4 2 0 0 1 1 0"
ALL_NOISY_1="$ALL_NOISY_1 1 4 1 1 1 0 1 1 0 0 1 4 1 1 1 1 3 0 1 1"

ALL_NOISY_1989="1 4 1 1 1 1 1 1 1 0 1 0 1 0 1 2 0 3 5 2"
ALL_NOISY_1989="$ALL_NOISY_1989 0 0 1 4 1 0 1 0 1 0 0 1 3 1 0 0 1 0 0 1"

ALL_NOISY_2019="1 0 0 1 0 3 0 0 1 1 2 4 0 2 1 0 1 3 1 1"
ALL_NOISY_2019="$ALL_NOISY_2019 1 1 0 1 0 4 1 2 0 0 0 2 4 3 2 0 0 4 0 2"

# CLEAN VALIDATION
CLEAN_VALID_1="1 1 0 0 1 3 0 1 0 4 0 0 1 3 1 2 1 0 0 1"
CLEAN_VALID_1="$CLEAN_VALID_1 0 3 0 0 0 2 1 3 0 4 0 3 1 3 0 4 0 1 1 0"

CLEAN_VALID_1989="0 0 1 1 0 4 1 1 0 4 0 1 4 0 1 0 0 3 1 3"
CLEAN_VALID_1989="$CLEAN_VALID_1989 1 2 1 1 0 1 1 1 0 1 0 3 1 2 1 2 0 4 0 4"

CLEAN_VALID_2019="0 0 0 0 0 2 1 3 0 3 1 1 1 0 0 1 1 0 0 4"
CLEAN_VALID_2019="$CLEAN_VALID_2019 1 4 1 2 1 2 0 1 0 0 1 1 1 3 1 4 0 1 0 0"

# CLEAN TRAIN
CLEAN_TRAIN_1="0 1 1 2 0 4 0 1 1 0 1 0 0 4 1 1 0 1 0 3"
CLEAN_TRAIN_1="$CLEAN_TRAIN_1 1 1 0 4 1 0 0 2 3 4 0 0 1 3 0 1 1 3 2 4"

CLEAN_TRAIN_1989="0 0 0 1 1 0 0 0 3 2 3 1 1 1 0 0 0 3 0 3"
CLEAN_TRAIN_1989="$CLEAN_TRAIN_1989 0 4 0 2 0 2 0 1 1 3 1 0 1 3 1 3 1 1 0 1"

CLEAN_TRAIN_2019="1 0 1 4 0 2 0 0 1 1 1 2 2 0 1 0 0 3 0 4"
CLEAN_TRAIN_2019="$CLEAN_TRAIN_2019 0 3 0 0 0 3 0 0 1 0 0 1 0 0 2 4 2 4 5 1"


DATE=`date +%Y%m%d-%H%M%S`
# loss function
LOSS="rll"
ALPHA="0.01"

# noise
TYPE="sym"
LEVEL="0.46"
SCOPE="both"

# others
VALID_FOR_TEST="False"
DATASET="cifar10"
EPOCH=630
SEED=1  # change
SETTING="ALL_NOISY" # change
GPU=3  # change
LAYERS=6
arc_name="${SETTING}_${SEED}"
fixed_arc="${!arc_name}"

if [ "${TYPE}" != "clean" ]
then
    NOISE="${TYPE}_${LEVEL}"
else
    NOISE="${TYPE}"
fi

if [ "${LOSS}" != "cce" ]
then
    LOSS_FUNC="${LOSS}_${ALPHA}"
else
    LOSS_FUNC="${LOSS}"
fi

OUTPUT="/home/yiwei/enas/logs/micro_noisy_final_rll0.01/${DATASET}_seed${SEED}_${LOSS_FUNC}_${arc_name}_gpu${GPU}-${DATE}"

python src/cifar10/main.py \
  --data_format="NCHW" \
  --search_for="micro" \
  --reset_output_dir \
  --data_path="data/cifar10" \
  --output_dir="${OUTPUT}" \
  --batch_size=96 \
  --num_epochs=${EPOCH} \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_fixed_arc="${fixed_arc}" \
  --child_use_aux_heads \
  --child_num_layers=${LAYERS} \
  --child_out_filters=36 \
  --child_num_branches=5 \
  --child_num_cells=5 \
  --child_keep_prob=0.80 \
  --child_drop_path_keep_prob=0.60 \
  --child_l2_reg=2e-4 \
  --child_lr_cosine \
  --child_lr_max=0.05 \
  --child_lr_min=0.0005 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --nocontroller_training \
  --controller_search_whole_channels \
  --controller_entropy_weight=0.0001 \
  --controller_train_every=1 \
  --controller_sync_replicas \
  --controller_num_aggregate=10 \
  --controller_train_steps=50 \
  --controller_lr=0.001 \
  --controller_tanh_constant=1.50 \
  --controller_op_tanh_reduce=2.5 \
  --loss="${LOSS}" \
  --alpha="${ALPHA}" \
  --noise_level="${LEVEL}" \
  --noise_type="${TYPE}" \
  --validation_for_test="${VALID_FOR_TEST}" \
  --scope="${SCOPE}" \
  --gpu="${GPU}" \
  --seed="${SEED}" \
  "$@"

