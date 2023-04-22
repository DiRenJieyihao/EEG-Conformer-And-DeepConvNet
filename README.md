# EEG-Conformer-And-DeepConvNet

&emsp;&emsp;我用自己处理好的BNCI2014001的数据来跑EEG Conformer, 并且把得到的结果和用DeepConvNet跑出来的结果进行比较，发现EEG Conformer对准确率的提升还是非常大的。

## 结果对比

|   Method  |  S01     | S02     | S03     |  S04     | S05     |  S06     | S07     | S08     |  S09     | AVG.
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| EEG Conformer | 83.68 | 55.56 | 96.18 | 70.84 | 80.21 | 63.89 | 89.93 | 87.85 | 84.03 | 79.13 |
| DeepConvNet | 56.94 | 43.40 | 67.71 | 41.32 | 64.58 | 42.36 | 56.60 | 58.68 | 62.15 | 54.86 |


## Datasets
- [BCI_competition_IV2a](http://bnci-horizon-2020.eu/database/data-sets), 里面的 Four class motor imagery (001-2014)对应的就是BNCI2014001数据集。我把数据集放在了mne_data里面，运行代码的时候，可以根据报错信息得到存放路径，再把mne_data放到提示的路径下。

## 代码问题

&emsp;&emsp;如果要用generate_data.py生成数据用于跑EEG Conformer, generate_data.py中的第62行和第64行代码要分别加1：```tl[1]```改为```tl[1]+1```。

## 代码运行

&emsp;&emsp;运行bci.sh文件可以通过以下命令：
```shell
nohup bash bci.sh >results.txt 2>&1 &
```


## Citation
```
@article{song2023eeg,
  title = {{{EEG Conformer}}: {{Convolutional Transformer}} for {{EEG Decoding}} and {{Visualization}}},
  shorttitle = {{{EEG Conformer}}},
  author = {Song, Yonghao and Zheng, Qingqing and Liu, Bingchuan and Gao, Xiaorong},
  year = {2023},
  journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume = {31},
  pages = {710--719},
  issn = {1558-0210},
  doi = {10.1109/TNSRE.2022.3230250}
}
```
