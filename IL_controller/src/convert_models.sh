src=$1
des=$2
src_val=$3
des_val=$4
source ~/workspace/bts_rl/bin/activate
python3 pytorch_model_to_c.py --trained_model $src --saving_name $des --batch_size 128 --no_vin 0 --l_h 100 --vinout 28 --w 64 --fit all
python3 pytorch_model_to_c.py --trained_model $src_val --saving_name $des_val --batch_size 128 --no_vin 1 --l_h 100 --vinout 28 --w 16 --fit val
deactivate
