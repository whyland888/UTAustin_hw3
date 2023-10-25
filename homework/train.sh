export learning_rate=.001
export n_epochs=10
export batch_size=400
export random_crop=50
export horizontal_flip=0.1
export vertical_flip=0.1
export random_rotate=(15,90)
export brightness=(.1,.9)
export contrast=(.1,.9)
export saturation=(.1,.9)
export hue=(.1,.5)
export optim="adamw"
export layers=[32,64,128]

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
	--saturation $saturation \
	--hue $hue \
	--optim $optim \
	--layers $layers

