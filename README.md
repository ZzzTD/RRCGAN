

# RRCGAN：A Radiometric Resolution Compression Method for Optical Remote Sensing Images Using Contrastive Learning
![rrcgan](https://github.com/user-attachments/assets/01f0172e-bd1b-4264-b171-d31805916eed)

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Update log

- 5/5/2024: Added related codes.
- 12/24/2024: Added GF2 dataset.

### RRCGAN Train and Test

- Train the RRCGAN model:
```bash
python train.py --dataroot XXX --name XXX
```

- Test the RRCGAN model:
```bash
python test.py --dataroot XXX --name XXX
```

### Datesets
All the data mentioned in the article, eg. JL1,GF7,GF2, has been uploaded to Baidu Cloud, link is: https://pan.baidu.com/s/1VOeaoqv50KVqbpiXJs5WwQ(thkn) 

### Acknowledgments
Our code is developed based on [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) and [Hneg_SRC
](https://github.com/jcy132/Hneg_SRC). We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation.
