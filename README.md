# DIFF (SIGIR'25)
This is the official repository for our paper "**DIFF: Dual Side-Information Filtering and Fusion for Sequential Recommendation**" in SIGIR'25.
You can find our paper on [[arXiv](https://arxiv.org/abs/2505.13974)].

## 🛠️ Installation
To replicate our environment, please follow the steps below:
```bash
# Clone this repository
git clone https://github.com/HyeYoung1218/DIFF.git
cd DIFF

# Create conda environment from the provided file
conda env create -f environment.yml
conda activate diff
```

## 📁 Dataset
We provide four preprocessed datasets, which are included in the `dataset/` directory:

* `Amazon_Beauty`
* `Amazon_Sports_and_Outdoors`
* `Amazon_Toys_and_Games`
* `yelp`

Each dataset is ready to use and requires no additional preprocessing.
The directory structure is as follows:

```
dataset/
├── Amazon_Beauty/
├── Amazon_Sports_and_Outdoors/
├── Amazon_Toys_and_Games/
└── yelp/
```

Each dataset corresponds to a configuration file located in the `configs/` folder. These `.yaml` files define dataset-specific training settings such as model parameters, data paths, and training schedule.

```
├── Amazon_Beauty_diff.yaml
├── Amazon_Sports_and_Outdoors_diff.yaml
├── Amazon_Toys_and_Games_diff.yaml
└── yelp_diff.yaml
```

## 🚀 Reproduction
To train and evaluate our model, simply run the shell script `run_diff.sh`, which includes predefined training commands for each dataset. This script contains the best-performing hyperparameters for each dataset.

Run the following command:
```bash
bash run_diff.sh
```

## 📄 Citation
If you find this repository helpful for your work, please cite the following paper:

```bibtex
@inproceedings{kim2025diff,
  title={DIFF: Dual Side-Information Filtering and Fusion for Sequential Recommendation},
  author={Hye-young Kim and
          Minjin Choi and
          Sunkyung Lee and
          Ilwoong Baek
          Jongwuk Lee},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2025}
}
```

## 🔗 Acknowledgement
This repository is based on [RecBole](https://github.com/RUCAIBox/RecBole) and [MSSR](https://github.com/xiaolLIN/MSSR?tab=readme-ov-file).
