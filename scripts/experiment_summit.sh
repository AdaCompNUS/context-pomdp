#!/bin/sh
SECONDS=0
gpu=$1
s=$2
e=$3
port=$4

launch_sim=1
record_bags=0
if (( $# > 4 )); then
    launch_sim=$5
fi
if (( $# > 5 )); then
    record_bags=$6
fi

maploc=random
# maploc=magic
# maploc=meskel_square
# maploc=shi_men_er_lu
# maploc=highway
mode=joint_pomdp
# mode=rollout
# mode=gamma
rands=-1
# rands=9475
eps_len=120.0
debug=0
num_car=75
num_bike=25
num_pedestrian=10

echo "User: $USER"
echo "PATH: $PATH"
echo "PYTHON: $(which python3)"

echo $(nvidia-smi)
num_rounds=1
rm exp_log_$s'_'$e
echo "log: exp_log_"$s'_'$e
echo "CUDA_VISIBLE_DEVICES=" $CUDA_VISIBLE_DEVICES

_term() {
  echo "Caught SIGTERM signal!"
  kill -TERM "$child" 2>/dev/null
}

trap _term SIGTERM

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
    --make 1 --verb 1 --gpu_id $gpu --debug $debug \
    --num-car $num_car --num-bike $num_bike --num-pedestrian $num_pedestrian\
    --port $port --maploc $maploc --rands $rands --launch_sim $launch_sim --eps_len $eps_len --drive_mode $mode &

    child=$!
    wait "$child"
#    echo "[repeat_run] clearing process"
#    python ./clear_process.py $port
#    sleep 3
done
echo "Exp finished in "$SECONDS" seconds"
