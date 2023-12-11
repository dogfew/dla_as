#python train.py ++optimizer.args.weight_decay=0

#python train.py "++loss.args.weights=[0.5 0.5]"

#python train.py +arch.args.gru_num_layers=1 +arch.args.prebatchnorm_gru=0

#python train.py ++arch.args.take_abs=True ++trainer.epochs=150

#python train.py "++loss.args.weights=[0.5 0.5]"

python train.py +arch.args.min_band_hz=100

python train.py +arch.args.freeze=True