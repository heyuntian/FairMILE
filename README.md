# FairMILE
Code and data for our paper 'FairMILE: Towards an efficient framework for fair graph representation learning' in EAAMO '23.

Link: https://par.nsf.gov/servlets/purl/10508212

Citation: @inproceedings{he2023fairmile,
  title={FairMILE: Towards an efficient framework for fair graph representation learning},
  author={He, Yuntian and Gurukar, Saket and Parthasarathy, Srinivasan},
  booktitle={Proceedings of the 3rd ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization},
  pages={1--10},
  year={2023}
}

## Requirements
Our codes are implemented in Python 3.8. All required packages are included in `requirements.yml`.

## Datasets
We have seven real-world datasets in our experiments. Namely, they are:
* **Node classification**: German, Bail (Recidivism), Credit, and Pokec-n
* **Link prediction**: Cora, Citeseer, Pubmed

This repository only includes pre-processed data for node classification datasets (except Pokec-n). You can download raw data for all datasets (and pre-processed Pokec-n) at [this link](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/he_1773_buckeyemail_osu_edu/EpTr3LaSplJOsfnny_2WnNQBr6hCmcG2AX5UFKj11x5zhA?e=qavTYn). 

Pre-processed data for node classification datasets can be rebuilt using `preprocess.py`. For link prediction datasets, it will automatically process the raw data in `datasets/raw/{dataset}` and store the processed version at the time of first use. Processed link prediction datasets are stored in `datasets/{dataset}/{seed}`.

## How to use
### Node Classification
For instance, to run FairMILE with NetMF on german (c=2), please use this command:
> python main.py --data german --basic-embed netmf --coarsen-level 2

To run the vanilla NetMF, simply add an argument `--baseline`:
> python main.py --data german --basic-embed netmf --baseline

### Link Prediction
To run FairMILE or baselines in link prediction tasks, please add an argument `--task lp`:
> python main.py --task lp --data cora --basic-embed node2vec --coarsen-level 2

> python main.py --task lp --data cora --basic-embed node2vec --baseline

### Arguments
Please check `main.py` for a comprehensive list of arguments.

## Contact
If you have any questions, feel free to [contact me](mailto:he.1773@osu.edu).
