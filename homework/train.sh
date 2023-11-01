export PYTHON=/home/william/anaconda3/envs/dl_hw_03/bin/python

export learning_rate=.002
export n_epochs=20
export batch_size=400
export random_crop=64
export horizontal_flip=.5
export vertical_flip=0
export random_rotate=0
export brightness=.5
export contrast=.1
export saturation=.1
export hue=.1
export optim="adamw"
export layers=3

# Run model
echo "========= Training CNN ========"

$PYTHON train_cnn.py \
	--log_dir CNN_Train_Results \
	--lr $learning_rate \
	--n_epochs $n_epochs \
	--batch_size $batch_size \
	--rand_crop $random_crop \
	--h_flip $horizontal_flip \
	--v_flip $vertical_flip \
	--rand_rotate $random_rotate \
	--brightness $brightness \
	--contrast $contrast \
	--saturation $saturation \
	--hue $hue \
	--optim $optim \
	--layers $layers \

