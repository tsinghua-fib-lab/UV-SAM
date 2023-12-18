
# UV-SAM: Adapting Segment Anything Model for Urban Village Identification
![OverallFramework](./assets/framework.jpg "Overall framework")

The official PyTorch implementation of "UV-SAM: Adapting Segment Anything Model for Urban Village Identification" (AAAI'24).
The code is tested under a Linux desktop with torch 2.0.1 and Python 3.10
## Dataset
Here we provide the dataset of Beijing and Xi'an for reproducibility, and the image data can be downloaded from *[here](https://drive.google.com/drive/folders/1szict970v0Z7LrY78jPQkNG1UzTHT1gb?usp=sharing)*.


## Running a model


Train model:
  ```
  cd tools
  python train.py 
  ```
  Evaluate :
  ```
  python inference_seg.py
  ```
Note:Please check the path of the pretrain models and datasets.
## Requirements
  pip install -r requirements.txt
## Acknowledgement

The implemention is based on *[RSPrompter](https://github.com/KyanChen/RSPrompter)*. Thanks to the developers of the RSPrompter project.
## Citation
If you found this library useful in your research, please consider citing:

```
@article{zhang2024uvsam,
  title={UV-SAM: Adapting Segment Anything Model for Urban Village Identification},
  author={Xin, Zhang and Liu, Yu  and Lin, Yuming and Liao, Qinming and Li, Yong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```