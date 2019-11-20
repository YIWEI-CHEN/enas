#!/bin/bash

export PYTHONPATH="$(pwd)"

fixed_arc="1 1 1 2 1 2 1 0 1 0 1 0 1 0 0 3 5 1 4 0"
fixed_arc="$fixed_arc 0 3 0 1 0 2 1 4 0 1 1 3 1 1 4 4 0 3 1 2"
DATE=`date +%m%d`

# loss function
LOSS="cce"
ALPHA="0.1"

# noise
LEVEL="0.28"
TYPE="clean"
SCOPE="both"

# others
VALID_FOR_TEST="True"
GPU="7"
RANK="R2"
MODELID="epoch120"

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

python src/cifar10/main.py \
  --data_format="NCHW" \
  --search_for="micro" \
  --reset_output_dir \
  --data_path="data/cifar10" \
  --output_dir="logs/${DATE}/${RANK}/micro_final_${NOISE}_${LOSS_FUNC}_gpu${GPU}_${MODELID}" \
  --batch_size=144 \
  --num_epochs=300 \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_fixed_arc="${fixed_arc}" \
  --child_use_aux_heads \
  --child_num_layers=13 \
  --child_out_filters=36 \
  --child_num_branches=5 \
  --child_num_cells=5 \
  --child_keep_prob=0.80 \
  --child_drop_path_keep_prob=0.60 \
  --child_l2_reg=2e-4 \
  --child_lr_cosine \
  --child_lr_max=0.05 \
  --child_lr_min=0.0001 \
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
  "$@"

