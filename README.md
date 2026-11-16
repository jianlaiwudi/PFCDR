# Privacy-Friendly Cross-Domain Recommendation via Distilling User-irrelevant Information (PFCDR)

## Introduction
This repository provides the implementations of PFCDR and two popular baselines (TGTOnly, EMCDR):
* TGTOnly：Train a MF model with the data of the target domain.
* EMCDR: [Cross-Domain Recommendation: An Embedding and Mapping Approach](https://www.ijcai.org/Proceedings/2017/0343.pdf) (IJCAI 2017)



## Requirements

- Python 3.6
- Pytorch > 1.0
- Pandas
- Numpy
- Tqdm

## File Structure

```
.
├── code
│   ├── config.json             # Common Configurations
│   ├── inversion_config.json   # Configurations for Inversion stage
│   ├── main.py                 # Entry function
│   ├── models.py               # Models based on MF, GMF or Youtube DNN
│   ├── preprocessing.py        # Parsing and Segmentation
│   ├── inversion.py            # Module for model inversion
│   ├── float_embedding.py      # A new version of torch.nn.embedding that support float as input
│   ├── README.md
│   └── run.py                  # Training and Evaluating 
└── data
    ├── mid                 # Mid data
    │   ├── Books.csv
    │   ├── CDs_and_Vinyl.csv
    │   └── Movies_and_TV.csv
    ├── raw                 # Raw data
    │   ├── reviews_Books_5.json.gz
    │   ├── reviews_CDs_and_Vinyl_5.json.gz
    │   └── reviews_Movies_and_TV_5.json.gz
    └── ready               # Ready to use
        ├── _2_8
        ├── _5_5
        └── _8_2
```

## Dataset

We utilized the Amazon Reviews 5-score dataset. 
To download the Amazon dataset, you can use the following link: [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/links.html).
Download the three domains: [CDs and Vinyl](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz), [Movies and TV](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz), [Books](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz) (5-scores), and then put the data in `./data/raw`.

You can use the following command to preprocess the dataset. 
The two-phase data preprocessing includes parsing the raw data and segmenting the mid data. 
The final data will be under `./data/ready`.

```python
python main.py --process_data_mid 1 --process_data_ready 1
```

## Run

Parameter Configuration:

- task: different tasks within `1, 2 or 3`, default for `1`
- base_model: different base models within `MF, GMF or DNN`, default for `MF`
- ratio: train/test ratio within `[0.8, 0.2], [0.5, 0.5] or [0.2, 0.8]`, default for `[0.8, 0.2]`
- epoch: pre-training and CDR mapping training epoches, default for `10`
- seed: random seed, default for `2020`
- gpu: the index of gpu you will use, default for `0`
- lr: learning_rate, default for `0.01`
- model_name: base model for embedding, default for `MF`

You can run this model through:

```powershell
# Run directly with default parameters 
python main.py --process_data_mid 0 --process_data_ready 0

```
It will run a little while since it need to generate prototypes on overlapped users in source and target domains. This 
prototypes will automatically save, next time you can directly use some users for CDR.

If you wanna try different `weight decay`, `embedding dimmension` or more tasks, you may change 
the settings in `./config.json`. If you wanna try the parameter of inversion stage you may change 
the settings in `./inversion_config.json`.

## Book to Movie: 20% test users

This run uses the original Python source files at upstream revision
`c88e6fe`: task 1, MF, ratio `[0.8, 0.2]`, seed 2020, 10 epochs,
embedding dimension 10, learning rate 0.01, prototype learning rate 0.005.
The 20% split is over common users, with all their target-domain ratings
held out. It does not mean holding out 20% of individual ratings.

On an RTX 4060 Laptop GPU with 8 GB VRAM, the original inversion batch of
200 passed a forward/backward/Adam memory check (about 4581 MiB peak).
The original evaluation batch of 1024 ran out of GPU memory. Only task 1's
`batchsize_test` is reduced to 128 (about 1487 MiB in the evaluation check,
including a 400 MiB reserve). Training batches, inversion settings, losses,
mapping functions and the scoring/voting algorithm are unchanged.
The MF implementation's existing hard-coded Top-K remains 70000.

Preprocessing uses Python 3.10.20, NumPy 1.26.4, pandas 2.2.3 and tqdm
4.67.1. Training uses Python 3.11.14, PyTorch 2.3.1 with CUDA 11.8,
NumPy 1.26.4, pandas 2.3.3, SciPy 1.16.2 and tqdm 4.67.1.
Python 3.10 is used for preprocessing to retain the original set sampling
without editing the source. Set `$prepPython` and `$trainPython` to the
respective environment executables before running these PowerShell commands
from the repository directory:

```powershell
$env:PYTHONHASHSEED = '2020'
$env:CUDA_VISIBLE_DEVICES = '0'
& $prepPython -c "import json; from preprocessing import DataPreprocessingMid; c=json.load(open('config.json')); DataPreprocessingMid(c['root'],'Books').main()"
& $prepPython -c "import json; from preprocessing import DataPreprocessingMid; c=json.load(open('config.json')); DataPreprocessingMid(c['root'],'Movies_and_TV').main()"
& $prepPython -c "import json,random; from preprocessing import DataPreprocessingReady; random.seed(2020); c=json.load(open('config.json')); DataPreprocessingReady(c['root'],c['src_tgt_pairs'],'1',[0.8,0.2]).main()"
git rev-parse HEAD
& $trainPython -u main.py --task 1 --base_model MF --gpu 0 --seed 2020 --epoch 10 --lr 0.01 --lr_prototype 0.005
```

The data-preparation commands call the original classes for this task only.
The training command retains the original `main.py -> Run.main()` flow.
Omit the `process_data_*` flags during training: the original parser treats
an explicitly passed string `0` as truthy. The ratio is the original default.

Record the Git commit, full commands, environment versions and console log
for each run. A single run yields the source code's per-epoch minimum test
MAE and RMSE; these two minima may occur in different epochs. It is not the
paper's ten-run mean. For independent repetitions, use different training
seeds and separate model/prototype cache directories, and commit each
parameter change before running. The paper does not publish the exact
split/checkpoint-selection/seeds needed to guarantee the same table values.

### Recorded single run (seed 2020)

The complete run is recorded in
[`results/book_movie_20_seed2020.json`](results/book_movie_20_seed2020.json),
including every epoch, dataset checksums, settings and validation results.

| Method | Minimum MAE | MAE epoch | Minimum RMSE | RMSE epoch |
| --- | ---: | ---: | ---: | ---: |
| TGTOnly | 4.150802 | 10 | 4.740552 | 10 |
| EMCDR | 1.125340 | 8 | 1.425091 | 8 |
| PFCDR (upstream `sfcdr` keys) | 0.943425 | 1 | 1.226853 | 2 |

These are independently selected minima from one run, following the original
`update_results()`. The paper's PFCDR values, 0.9579 / 1.2437, are a ten-run
mean. The locally saved last-epoch PFCDR checkpoint has MAE 0.956729 and RMSE
1.227226; it is distinct from the independently selected best metrics above.

The split contains 29,910 training common users, 7,478 test common users and
170,081 test ratings. All three prototype files have the expected unique
users and ten finite prototype coordinates. Comparing the baseline and
last-epoch checkpoints confirms that source embeddings, target embeddings
and the EMCDR mapping stayed unchanged during PFCDR; only `rp_mapping`
was updated.

The first process encountered a CSV sharing `PermissionError` during target
prototype output. The completed source prototypes and baseline checkpoint
were retained. Target prototypes were rebuilt with the original inversion
class after replaying the original random sequence; all 1,200 previously
written target rows matched exactly. Progress reads were changed to allow
concurrent writes. The repository's Python source files remain identical to
`c88e6fe`, and the experiment settings did not change during recovery.

## MF reproduction: three tasks, three test ratios, two seeds

The execution plan is [experiments/full_reproduction_plan.json](experiments/full_reproduction_plan.json).
It covers TGTOnly, EMCDR and PFCDR for Book → Movie, Book → Music and
Music → Movie at 20%, 50% and 80% held-out common users. Each condition uses
two distinct training seeds, 2020 and 2021, on a fixed split created
with seed 2020 and PYTHONHASHSEED=2020. The verified single run above is
reused once as task 1 / 20% / seed 2020, leaving 17 new runs out of 18 total.
The user reduced the plan from ten to two repetitions per condition on
2026-11-16. The active seed-2021 training continues with its existing parameters.

Each run has a separate working directory, baseline checkpoint and prototype
cache. It calls the original Run.main(), with the original model,
optimizers, losses, inversion and voting functions. Auxiliary execution,
monitoring and recovery scripts are outside this repository. All algorithm
Python files remain identical to c88e6fe.

Only evaluation batch sizes are reduced: task 1 remains 128; task 2 changes
from 256 to 128, and task 3 changes from 512 to 128. Original training
batches, inversion batch 200, inversion iterations 300, epoch count 10,
learning rates, embedding dimension and hard-coded MF Top-K are retained.

Before each run, its parameters are committed in
[experiments/current_run.json](experiments/current_run.json). After completion,
the actual per-epoch metrics are saved under results/full_reproduction/trials/
and committed. Completed seeds and statistics are tracked in
[results/full_reproduction/summary.json](results/full_reproduction/summary.json).
Within a run, the original independently selected minimum MAE and RMSE are
used. Across the two completed seeds, their arithmetic mean and sample standard
deviation are calculated. A row is filled only after both seeds finish.
These are two-run reproduction statistics; the paper reports ten-run means.
Failed attempts and repeated launches of the same seed are never counted
as additional independent runs.

<!-- FULL_REPRODUCTION_TABLE_START -->
| Task | Test users | Runs | TGT MAE | TGT RMSE | EMCDR MAE | EMCDR RMSE | PFCDR MAE | PFCDR RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Book → Movie | 20% | 2/2 | 4.1960 | 4.7702 | 1.1127 | 1.4044 | 0.9347 | 1.2129 |
| Book → Movie | 50% | 2/2 | 4.2233 | 4.7825 | 1.1735 | 1.4846 | 0.9821 | 1.2871 |
| Book → Movie | 80% | 2/2 | 4.2363 | 4.8169 | 1.3265 | 1.6790 | 1.1217 | 1.4745 |
| Book → Music | 20% | 2/2 | 4.4849 | 5.1491 | 1.7406 | 2.0606 | 1.7211 | 1.9910 |
| Book → Music | 50% | 0/2 | — | — | — | — | — | — |
| Book → Music | 80% | 0/2 | — | — | — | — | — | — |
| Music → Movie | 20% | 0/2 | — | — | — | — | — | — |
| Music → Movie | 50% | 0/2 | — | — | — | — | — | — |
| Music → Movie | 80% | 0/2 | — | — | — | — | — | — |
<!-- FULL_REPRODUCTION_TABLE_END -->

If the code is helpful for you, please cite this paper:

```
@inproceedings{wang2025privacy,
  title={Privacy-Friendly Cross-Domain Recommendation via Distilling User-irrelevant Information},
  author={Wang, Cheng and Xu, Wenchao and Wang, Haozhao and Liu, Wei and Li, Ruixuan},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={450--461},
  year={2025}
}
```
