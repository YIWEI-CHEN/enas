#!/bin/bash

Rank="R5"

DST="${Rank}_accuracy"
if [ ! -d ${DST} ]; then
    echo "Create directory ${DST}"
    mkdir -p ${DST}
fi

echo "Extract test accuracy from ${Rank}/micro_final_train_sym_0.28_robust_0.1_gpu0_epoch1/stdout"
grep -oP "(?<=test_accuracy:) 0\.[0-9]{1,4}" "${Rank}/micro_final_train_sym_0.28_robust_0.1_gpu0_epoch1/stdout" > "${DST}/acc_epoch1.txt"

echo "Extract test accuracy from ${Rank}/micro_final_train_sym_0.28_robust_0.1_gpu2_epoch30/stdout"
grep -oP "(?<=test_accuracy:) 0\.[0-9]{1,4}" "${Rank}/micro_final_train_sym_0.28_robust_0.1_gpu2_epoch30/stdout" > "${DST}/acc_epoch30.txt"

echo "Extract test accuracy from ${Rank}/micro_final_train_sym_0.28_robust_0.1_gpu1_epoch60/stdout"
grep -oP "(?<=test_accuracy:) 0\.[0-9]{1,4}" "${Rank}/micro_final_train_sym_0.28_robust_0.1_gpu1_epoch60/stdout" > "${DST}/acc_epoch60.txt"

echo "Extract test accuracy from ${Rank}/micro_final_train_sym_0.28_robust_0.1_gpu3_epoch90/stdout"
grep -oP "(?<=test_accuracy:) 0\.[0-9]{1,4}" "${Rank}/micro_final_train_sym_0.28_robust_0.1_gpu3_epoch90/stdout" > "${DST}/acc_epoch90.txt"

echo "Extract test accuracy from ${Rank}/micro_final_train_sym_0.28_robust_0.1_gpu0_epoch120/stdout"
grep -oP "(?<=test_accuracy:) 0\.[0-9]{1,4}" "${Rank}/micro_final_train_sym_0.28_robust_0.1_gpu0_epoch120/stdout" > "${DST}/acc_epoch120.txt"
