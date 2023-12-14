# Anti-Spoofing project


## Installation

Make sure that your python version >= 3.10

Run commands in `evaluate_script.sh`
```shell 
bash evaluate_script.sh
```
The commands in file `evaluate_script.sh` are: 
```shell
pip install -r requirements.txt
pip install gdown>4.7
mkdir -p default_test_model
cd default_test_model
gdown 13X8Y8UNIdLIg7M7E817a_YHrDZx8xeZs -O checkpoint.pth
cd ..
```

## Test

### Generate sentences required in task



If you want to check my LCNN-LFCC model: 
```shell
 python test.py --config-name="config_lcnn_lfcc.yaml" +resume="default_test_model/lcnn-lfcc-10.pth" test_settings.skip_test=True
```

If you want to check RawNet2-S1. ()
```shell
python test.py +resume="default_test_model/rawnet2-s1-50.pth" test_settings.skip_test=True
```

If you want to check RawNet2-S3. (You can skip test for example)
```shell
python test.py +resume="default_test_model/rawnet2-s3-50.pth" test_settings.skip_test=True test_settings.audio_dir="test_data"
```

## Training
To prepare data, run: 
```shell
pip install -r requirements.txt
bash prep_script.sh
```

To reproduce my final model, train Hi-Fi GAN. Config path: `src/configs/config.yaml`: 
```shell
python train.py
```

**Optional Tasks:**

- (up to +1) for Hydra. There were major changes in `src/utils/parse_config.py`, `train.py` and `test.py` files
