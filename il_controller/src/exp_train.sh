
echo "starting exp_train script"
epochs=50


ped=5
lh=50
k=5
no_vin=0
fit="steer"
echo "Train with setting ped_$ped, lh_$lh, k_$k, novin_$no_vin"
python train.py --fit $fit --batch_size 128 --lr 0.0001 --epochs $epochs --no_ped $ped --l_h $lh --k $k --no_vin $no_vin --train train.h5 --val val.h5 --modelfile trained_models/all > train_log_fit_$fit'_p'$ped'_lh'$lh'_k'$k'_valina'$no_vin'.txt'
echo "Finish training"
sleep 3

ped=5
lh=50
k=5
no_vin=0
fit="acc"
echo "Train with setting ped_$ped, lh_$lh, k_$k, novin_$no_vin"
python train.py --fit $fit --batch_size 128 --lr 0.0001 --epochs $epochs --no_ped $ped --l_h $lh --k $k --no_vin $no_vin --train train.h5 --val val.h5 --modelfile trained_models/all > train_log_fit_$fit'_p'$ped'_lh'$lh'_k'$k'_valina'$no_vin'.txt'
echo "Finish training"
sleep 3

ped=5
lh=50
k=5
no_vin=0
fit="action"
echo "Train with setting ped_$ped, lh_$lh, k_$k, novin_$no_vin"
python train.py --fit $fit --batch_size 128 --lr 0.0001 --epochs $epochs --no_ped $ped --l_h $lh --k $k --no_vin $no_vin --train train.h5 --val val.h5 --modelfile trained_models/all > train_log_fit_$fit'_p'$ped'_lh'$lh'_k'$k'_valina'$no_vin'.txt'
echo "Finish training"
sleep 3

ped=5
lh=50
k=5
no_vin=0
fit="vel"
echo "Train with setting ped_$ped, lh_$lh, k_$k, novin_$no_vin"
python train.py --fit $fit --batch_size 128 --lr 0.0001 --epochs $epochs --no_ped $ped --l_h $lh --k $k --no_vin $no_vin --train train.h5 --val val.h5 --modelfile trained_models/all > train_log_fit_$fit'_p'$ped'_lh'$lh'_k'$k'_valina'$no_vin'.txt'
echo "Finish training"
sleep 3

ped=5
lh=50
k=10
no_vin=0
fit="steer"
echo "Train with setting ped_$ped, lh_$lh, k_$k, novin_$no_vin"
python train.py --fit $fit --batch_size 128 --lr 0.0001 --epochs $epochs --no_ped $ped --l_h $lh --k $k --no_vin $no_vin --train train.h5 --val val.h5 --modelfile trained_models/all > train_log_fit_$fit'_p'$ped'_lh'$lh'_k'$k'_valina'$no_vin'.txt'
echo "Finish training"
sleep 3

ped=5
lh=50
k=10
no_vin=0
fit="acc"
echo "Train with setting ped_$ped, lh_$lh, k_$k, novin_$no_vin"
python train.py --fit $fit --batch_size 128 --lr 0.0001 --epochs $epochs --no_ped $ped --l_h $lh --k $k --no_vin $no_vin --train train.h5 --val val.h5 --modelfile trained_models/all > train_log_fit_$fit'_p'$ped'_lh'$lh'_k'$k'_valina'$no_vin'.txt'
echo "Finish training"
sleep 3

ped=5
lh=50
k=10
no_vin=0
fit="action"
echo "Train with setting ped_$ped, lh_$lh, k_$k, novin_$no_vin"
python train.py --fit $fit --batch_size 128 --lr 0.0001 --epochs $epochs --no_ped $ped --l_h $lh --k $k --no_vin $no_vin --train train.h5 --val val.h5 --modelfile trained_models/all > train_log_fit_$fit'_p'$ped'_lh'$lh'_k'$k'_valina'$no_vin'.txt'
echo "Finish training"
sleep 3

ped=5
lh=50
k=10
no_vin=0
fit="vel"
echo "Train with setting ped_$ped, lh_$lh, k_$k, novin_$no_vin"
python train.py --fit $fit --batch_size 128 --lr 0.0001 --epochs $epochs --no_ped $ped --l_h $lh --k $k --no_vin $no_vin --train train.h5 --val val.h5 --modelfile trained_models/all > train_log_fit_$fit'_p'$ped'_lh'$lh'_k'$k'_valina'$no_vin'.txt'
echo "Finish training"
sleep 3