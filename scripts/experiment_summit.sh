#!/bin/sh
SECONDS=0
gpu=0
s=0
e=0
port=2000

launch_sim=1
record_bags=0

mode=joint_pomdp

maploc=random
rands=-1

if [ "$#" -gt 0 ]; then                                                                                                      
    mode=$1                                                                                                                  
fi                                                                                                                           
                                                                                                                             
if [ "$#" -gt 1 ]; then                                                                                                      
    gpu=$2                                                                                                                   
fi                                                                                                                           
                                                                                                                             
if [ "$#" -gt 2 ]; then                                                                                                      
    launch_sim=$3                                                                                                            
fi                                                                                                                           
                                                                                                                             
if [ "$#" -gt 3 ]; then                                                                                                      
    record_bags=$4                                                                                                           
fi                                                                                                                           
                                                                                                                             
if [ "$#" -gt 4 ]; then                                                                                                      
    s=$5                                                                                                                     
fi                                                                                                                           
                                                                                                                             
if [ "$#" -gt 5 ]; then                                                                                                      
    e=$6                                                                                                                     
fi                                                                                                                           
                                                                                                                             
if [ "$#" -gt 6 ]; then                                                                                                      
    port=$7                                                                                                                  
fi   

# maploc=magic
# maploc=meskel_square
# maploc=chandni_chowk
# maploc=highway
# mode=rollout
# mode=gamma
# rands=9475
# maploc=shi_men_er_lu
# rands=5271552
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
    kill -9 "$child"
#    echo "[repeat_run] clearing process"
    # python ./clear_process.py $port
    # sleep 3
done
echo "Exp finished in "$SECONDS" seconds"
