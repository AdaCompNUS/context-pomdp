lr=$1
model_version=$2
existing_model=$3
new_model=$4
moreepochs=$5
gpu=$6
export CUDA_VISIBLE_DEVICES=1,3
python3 train.py --batch_size 128 --lr $lr --train train.h5 --val val.h5 --no_vin 0 \
                        --l_h 100 --vinout 28 --w 64 --fit all --ssm 0.03 \
                        --logdir runs/lets_drive_$model_version \
                        --resume $existing_model \
                        --modelfile $new_model \
                        --moreepochs $moreepochs \
                        --exactname 1 2>&1 | tee "train_all_log_$model_version.txt"
