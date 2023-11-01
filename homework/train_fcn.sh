export PYTHON=/home/william/anaconda3/envs/dl_hw_03/bin/python

export learning_rate=.002
export n_epochs=20
export batch_size=64
export horizontal_flip=.5
export vertical_flip=0
export random_rotate=0
export brightness=.5
export contrast=.1
export saturation=.1
export hue=.1

# Run model
echo "========= Training FCN ========"

$PYTHON train_fcn.py \
	--log_dir FCN_Train_Results \
	--lr $learning_rate \
	--n_epochs $n_epochs \
	--batch_size $batch_size \
	--h_flip $horizontal_flip \
	--v_flip $vertical_flip \
	--rand_rotate $random_rotate \
	--brightness $brightness \
	--contrast $contrast \
	--saturation $saturation \
	--hue $hue \

