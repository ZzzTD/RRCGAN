

# RRCGAN：A Radiometric Resolution Compression Method for Optical Remote Sensing Images Using Contrastive Learning
![rrcgan](https://github.com/ZzzTD/RRCGAN/assets/58813118/9da7d669-2d6c-4fdf-a18c-cc3eb28a9658)

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Update log

5/5/2024: Added related codes.

### RRCGAN Train and Test

- Train the RRCGAN model:
```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT
```

- Test the RRCGAN model:
```bash
python test.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT 
```

### Datesets
All the data mentioned in the article has been uploaded to Baidu Cloud, link is:https://pan.baidu.com/s/1C45OYUsjJ4kO7GIkqCQSOQ(r64k) 

### Acknowledgments
Our code is developed based on [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation). We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation.
