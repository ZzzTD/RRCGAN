

# RRCGAN：A Radiometric Resolution Compression Method for Optical Remote Sensing Images Using Contrastive Learning
![新建 Microsoft PowerPoint 演示文稿 (1)](https://github.com/ZzzTD/RRCGAN/assets/58813118/2c84a58f-a55a-414f-8b44-9d2ce5424b5f)
## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Update log

5/5/2024: Added related codes.

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
All the data mentioned in the article has been uploaded to Baidu Cloud, link is:https://pan.baidu.com/s/1C45OYUsjJ4kO7GIkqCQSOQ(r64k) 

### Acknowledgments
Our code is developed based on [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) and [Hneg_SRC
](https://github.com/jcy132/Hneg_SRC). We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation.
