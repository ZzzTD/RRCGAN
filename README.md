

# RRCGAN：A Radiometric Resolution Compression Method for Optical Remote Sensing Images Using Contrastive Learning
![rrcgan](https://github.com/user-attachments/assets/01f0172e-bd1b-4264-b171-d31805916eed)

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Update log

- 5/5/2024: Added related codes.
- 12/24/2024: Added GF2 dataset.
- 02/03/2025: Update dataset.
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
All the data mentioned in the article, eg. JL1,GF7,GF2, has been uploaded to Baidu Cloud, link is: (https://pan.baidu.com/s/1eNE-5UvD4df62ehgllwYsA)(mhiy) .The dataset example is as follows:
![新建 Microsoft Visio Drawing](https://github.com/user-attachments/assets/02afc97c-3fe2-49b3-b6f2-0577334d6873)
@article{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2108.10257},
  year={2021}
}

### Acknowledgments
Our code is developed based on [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) and [Hneg_SRC
](https://github.com/jcy132/Hneg_SRC). We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation.

## Citation
```bash
@article{Zhang2025RRCGAN,
  title={RRCGAN: Unsupervised Compression of Radiometric Resolution of Remote Sensing Images Using Contrastive Learning},
  author={Tengda Zhang; Jiguang Dai; Jinsong Cheng; Hongzhou Li; Ruishan Zhao; Bing Zhang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025}
}.
```
