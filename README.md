# RangeFC: Temporal Fact Checking with Year-Range Prediction

<p align="center">
  <img src="logo.png" width="220">
</p>

<p align="center">
  <b>Predicting coherent validity intervals for knowledge graph assertions.</b>
</p>

RangeFC is an interval-aware temporal fact-checking framework that predicts the start and end years of validity for a given knowledge graph assertion ((s,p,o)). The model extends the TemporalFC codebase with a relation-aware Mixture-of-Experts architecture, a shared calendar representation, coupled boundary prediction, and overlap-aware training.

## Highlights

* Predicts complete validity intervals rather than a single timestamp.
* Uses frozen Dihedron entity and relation embeddings.
* Employs relation-aware top-(k) Mixture-of-Experts routing.
* Uses a factorized calendar head for start/end prediction.
* Optimizes interval quality using boundary and overlap-aware objectives.
* Evaluated on five Wikidata temporal property pairs.

## Main Results

| Model       | MAE Start ↓ | MAE End ↓ |     IoU ↑ |
| ----------- | ----------: | --------: | --------: |
| TemporalFC  |       12.07 |     28.74 |     0.624 |
| **RangeFC** |       18.94 | **21.37** | **0.661** |

RangeFC improves overall interval overlap while substantially reducing end-boundary error compared with the adapted TemporalFC baseline.

## Installation

Clone the repository:

```bash
git clone https://github.com/dice-group/RangeFC.git
cd RangeFC
```

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate tfc_clean_gpu
```

## Dataset

The dataset used for the RangeFC experiments is `wikidata6_latest2`.

Due to the size of the dataset and the pre-trained Dihedron embeddings, the dataset files are distributed separately through the GitHub Releases section.

Download `wikidata6_latest2.zip` and extract it into the `data_TP/` directory.

The expected structure is:

```text
RangeFC/
├── main.py
├── executer_TP.py
├── data_TP.py
├── ...
└── data_TP/
    └── wikidata6_latest2/
        ├── train/
        │   └── train
        ├── valid/
        │   └── valid
        ├── test/
        │   └── test
        ├── embeddings/
        │   └── dihedron/
        │       ├── entity.npy
        │       ├── entity.pkl
        │       ├── relation.npy
        │       └── relation.pkl
        ├── entities
        ├── relations
        └── times
```

## Running RangeFC

Example command:

```bash
python main.py \
  --path_dataset_folder "./data_TP/" \
  --eval_dataset "wikidata6_latest2" \
  --task "range-prediction" \
  --model "range-mlp" \
  --emb_type "dihedron" \
  --embedding_dim 100 \
  --batch_size 1024 \
  --val_batch_size 1000 \
  --use_interaction 1 \
  --use_prod 0 \
  --loss_type "huber" \
  --huber_beta 0.7366304701152739 \
  --end_weight 1.0013169655965504 \
  --extra_order_pen 0.06686696713024719 \
  --lr 0.0018887997194523478 \
  --num_workers 4 \
  --max_num_epochs 120 \
  --seed 42 \
  --emb_noise 0.011450934693677826 \
  --hidden_dim 1024 \
  --dropout 0.11699588659730913 \
  --gauss_sigma_idx 1.3507654810031005 \
  --t_dim 128 \
  --num_experts 5 \
  --k_experts 3 \
  --gate_temp_start 1.4307208041278083 \
  --gate_balance 0.023266080281596692 \
  --use_bands 1 \
  --use_prior 0 \
  --band_margin 1.5763792105820356
```

## Acknowledgement

RangeFC builds upon the open-source [TemporalFC](https://github.com/dice-group/TemporalFC) codebase. We thank the TemporalFC authors for making their implementation publicly available.

## Citation

Citation information for the RangeFC paper will be added here.
