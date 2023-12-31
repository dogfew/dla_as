# Anti-Spoofing project


## Installation

Make sure that your python version >= 3.10

Run commands in `evaluate_script.sh`
```shell 
bash evaluate_script.sh
```

Also, if you want to install data to check metrics on LA part of ASVspoof2019 dataset, please run `prep_script.sh`
The commands in file `prep_script.sh` are: 

```shell
bash prep_script.sh
```

Otherwise, you have to specify protocols file and dir with audios. For example, to get results on train part of dataset with rawnet2-s3, you can run: 

```shell
python test.py +resume="default_test_model/rawnet2-s3-50.pth" data.test.datasets.0.args.wav_dir="data/LA/ASVspoof2019_LA_train/flac" data.test.datasets.0.args.txt_path="data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt" test_settings.skip_test=False
```
## Test

You will get metrics for test part of dataset and also probabilities for files in audio_dir (`test_settings.audio_dir="test_data"`)

### LCNN-lFCC

If you want to check my LCNN-LFCC model: 
```shell
 python test.py --config-name="config_lcnn_lfcc.yaml" +resume="default_test_model/lcnn-lfcc-10.pth" test_settings.skip_test=False
```

### RawNet2-S1
If you want to check RawNet2-S1. 
```shell
python test.py +resume="default_test_model/rawnet2-s1-50.pth" test_settings.skip_test=False
```

### RawNet2-S3
If you want to check RawNet2-S3. 
```shell
python test.py +resume="default_test_model/rawnet2-s3-50.pth" test_settings.skip_test=False test_settings.audio_dir="test_data"
```
### Inference on train or dev

If you want to get metrics on train (or dev, if you change arguments) part of dataset, you can this commands:
##### RawNet2-S1
```shell
python test.py +resume="default_test_model/rawnet2-s1-50.pth" data.test.datasets.0.args.wav_dir="data/LA/ASVspoof2019_LA_train/flac" data.test.datasets.0.args.txt_path="data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt" test_settings.skip_test=False 
```

##### RawNet2-S3
```shell
python test.py +resume="default_test_model/rawnet2-s3-50.pth" data.test.datasets.0.args.wav_dir="data/LA/ASVspoof2019_LA_train/flac" data.test.datasets.0.args.txt_path="data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt" test_settings.skip_test=False 
```
##### LFCC-LCNN
```shell
python test.py --config-name="config_lcnn_lfcc.yaml" +resume="default_test_model/lcnn-lfcc-10.pth"  data.test.datasets.0.args.wav_dir="data/LA/ASVspoof2019_LA_train/flac" data.test.datasets.0.args.txt_path="data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt" test_settings.skip_test=False
```
## Training
To prepare data, run: 
```shell
pip install -r requirements.txt
bash prep_script.sh
```

To reproduce my final RawNet2 model, train: `src/configs/config_rawnet2.yaml`: 
```shell
python train.py
```
If you want to reproduce LCNN, train: `src/configs/config_lcnn_lfcc.yaml`:
```shell
python train.py --config-name="config_lcnn_lfcc.yaml"
```
If you want to reproduce RawNet2-S3 model, train: `src/configs/config_rawnet2.yaml` with s3 arguments:
```shell
python train.py +arch.args.s3=True trainer.mixed_precision=False
```
