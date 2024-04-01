#!/bin/bash

# echo 10x5扩展op60

# python ./data_utils.py  --n_op=60 \
#                         --n_j=10 \
#                         --n_m=5 \
#                         --op_per_mch_max=5 \
#                         --data_source=10x5xOP60 \
#                         --cover_data_flag=True
# echo 11x5
# python ./data_utils.py  --n_op=50 \
#                         --n_j=11 \
#                         --n_m=5 \
#                         --op_per_mch_max=5 \
#                         --data_source=11x5 \
#                         --cover_data_flag=True
# echo 10x6
# python ./data_utils.py  --n_op=50 \
#                         --n_j=10 \
#                         --n_m=6 \
#                         --op_per_mch_max=5 \
#                         --data_source=10x6 \
#                         --cover_data_flag=True
# echo
# python ./data_utils.py  --n_op=50 \
#                         --n_j=11 \
#                         --n_m=6 \
#                         --op_per_mch_max=5 \
#                         --data_source=11x6 \
#                         --cover_data_flag=True

echo 改变机器数量

# for n_m in 5 7 9 10 11 13 15; do
#     python ./data_utils.py --n_j=20 \
#                             --n_m=$n_m \
#                             --op_per_mch_max=5 \
#                             --cover_data_flag=True
# done



# for n_m in 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5; do
# for n_j in 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5; do
#     python ./data_utils.py --n_j=$n_j  \
#                             --n_m=$n_m \
#                             --op_per_job 10 \
#                             --cover_data_flag=True \
#                             --data_type test
# done
# done

for n_m in 5 10; do
for n_j in 20 15 10; do
    python ./data_utils.py --n_j=$n_j  \
                            --n_m=$n_m \
                            --cover_data_flag=True \
                            --data_type test \
                            --data_source SD2EC
done
done
