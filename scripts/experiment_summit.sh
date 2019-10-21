#!/bin/sh
SECONDS=0
gpu=$1
s=$2
e=$3
port=$4
record_bags=0
mode=joint_pomdp

echo "User: $USER"
echo "PATH: $PATH"
echo "PYTHON: $(which python3)"

echo $(nvidia-smi)
num_rounds=1
rm exp_log_$s'_'$e
echo "log: exp_log_"$s'_'$e
export CUDA_VISIBLE_DEVICES=$gpu
echo "CUDA_VISIBLE_DEVICES=" $CUDA_VISIBLE_DEVICES
for i in $(seq $s $e)
do
    echo "[repeat_run] starting run_data_collection.py script"
    start_batch=$((i*num_rounds))
    echo "[repeat_run] start_batch: $start_batch"
    end_batch=$(((i+1)*num_rounds))
    echo "[repeat_run] end_batch: $end_batch"
    echo "[repeat_run] gpu_id: $gpu"
    python3 run_data_collection.py --record $record_bags \
    --sround $start_batch --eround $end_batch \
    --make 1 --verb 1 --port $port --baseline $mode 2>&1 | tee -a exp_log_$s'_'$e
    echo "[repeat_run] clearing process"
    python ./clear_process.py $port
    sleep 3
done
echo "Exp finished in "$SECONDS
