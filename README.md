````markdown
# TemporalFC — MLP Range Prediction (Thesis Branch)

This branch contains a **simple, fast MLP** that predicts a **time range** (start year, end year) for a triple **(subject, predicate, object)** from a temporal knowledge graph.

- **Earlier baseline (before this branch):** MAE ≈ **17.6 / 18.7**, IoU ≈ **0.53**  
- **Current best (this branch):** MAE ≈ **3.6 / 5.2**, IoU ≈ **0.75** on `wikidata6`

> This branch focuses only on the MLP range predictor (no ElasticSearch/path components).

---

## 1) Quick Start

```bash
# clone
git clone https://github.com/abdullahqamer/my-temporalfc.git
cd my-temporalfc

# switch to this branch (if not already on it)
git checkout MLP

# create and activate environment (edit env name if you prefer)
conda env create -f environment.yml
conda activate tfc
````

### Get the dataset & embeddings

Download the release and unzip it into the repo root so it creates `data_TP/...`:

* [https://github.com/abdullahqamer/my-temporalfc/releases/tag/v1.0](https://github.com/abdullahqamer/my-temporalfc/releases/tag/v1.0)

Expected layout (key parts):

```
data_TP/
  wikidata6/
    train/
    valid/
    test/
    embeddings/      # e.g., dihedron *.npy
```

---

## 2) Reproducing Thesis Results

### Train the MLP (range prediction)

```bash
python main.py \
  --eval_dataset wikidata6 \
  --task range-prediction \
  --model range-mlp \
  --emb_type dihedron \
  --embedding_dim 100 \
  --batch_size 1024 \
  --val_batch_size 1000 \
  --use_interaction 1 \
  --use_prod 1 \
  --loss_type huber \
  --huber_beta 0.8970321391066037 \
  --end_weight 0.95 \
  --extra_order_pen 0.042741660857969106 \
  --lr 0.00184775182894049 \
  --num_workers 4 \
  --max_num_epochs 120 \
  --seed 42 \
  --emb_noise 0.02
```

**Notes**

* `--emb_noise` adds tiny Gaussian noise to entity/relation embeddings **during training only** (helps generalization).
* A **cosine LR schedule (with a low floor)** and **tiny label jitter** are already handled inside the model code—no extra flags needed.
* To approximate an older/simpler baseline, you can try `--end_weight 0.86` and `--emb_noise 0.00`.

### Evaluate the best checkpoint

Training stores checkpoints under `dataset/HYBRID_Storage/<timestamp>/...`. Evaluate with:

```bash
python evaluate_checkpoint_model_TP.py \
  --checkpoint_dir_folder all \
  --checkpoint_dataset_folder dataset/ \
  --eval_dataset wikidata6 \
  --model range-mlp \
  --task range-prediction \
  --emb_type dihedron \
  --embedding_dim 100
```

---

## 3) Model Overview (What the MLP does)

**Input:** a triple **(s, p, o)**
**Embeddings:**

* `h = emb(s)` (subject), `r = emb(p)` (predicate), `t = emb(o)` (object)

**Interaction features:**

* absolute difference `|h − t|`
* elementwise product `h ⊙ t`

**Concatenate features:**

* `x = [h, r, t, |h − t|, h ⊙ t]`

**MLP trunk (fully-connected stack):**

```
Linear → GELU → Dropout
Linear → GELU → Dropout
Linear → GELU → Dropout
```

**Head (2 outputs):**

* `Linear(out=2) → [s_raw, d_raw]`

**Map to [0,1] and build a valid interval:**

* `start = σ(s_raw)`
* `delta = σ(d_raw)`
* `end   = start + (1 − start) * delta`  (guarantees `end ≥ start` and both in `[0,1]`)

**Training objective:**

* Huber loss on `(start, end)` vs. target years (normalized).
* Small “order/consistency” penalty to discourage `end < start`.

**Key changes that improved results in this branch:**

* **Label jitter (tiny, training-only):** small noise on target time indices → more robust.
* **Cosine learning-rate schedule with a low floor:** fast early learning, gentle late refinement.
* **Slightly higher end weight in the loss:** improves interval quality/IoU.
* **Tiny embedding noise (training-only):** regularizes embeddings without affecting inference.

---

## 4) Directory Structure (relevant parts)

```
.
├── main.py                          # training / validation entrypoint
├── evaluate_checkpoint_model_TP.py  # evaluation for time-range prediction
├── nn_models_TP/
│   └── range_mlp_model.py           # MLP architecture & training logic
├── utils_TP/                        # data utilities, model selection, etc.
└── data_TP/                         # dataset & embeddings (after download)
```

---

## 5) Tips

* Use `--num_workers` to speed up data loading (e.g., 4–8 if your CPU allows).
* Keep `--seed 42` for exact reproducibility in the thesis runs.
* If a GPU is available, PyTorch Lightning will use it automatically.

---

## 6) Acknowledgements & Related Work

This branch builds on and reuses parts of the original **TemporalFC** codebase and ideas:

* **TemporalFC (ISWC 2023)** — temporal fact checking & time prediction framework.
  Please cite the original paper from the upstream project.

Embeddings & tooling:

* **Dihedron** temporal KG embeddings (used for entity/relation vectors).
* **PyTorch** and **PyTorch Lightning** for training infrastructure.

> If you use this branch, please credit the original TemporalFC work and the Dihedron embeddings.

---

## 7) Citation (TemporalFC)

```bibtex
@inproceedings{10.1007/978-3-031-47240-4_25,
  title     = {TemporalFC: A Temporal Fact Checking Approach over Knowledge Graphs},
  author    = {Qudus, Umair and Röder, Michael and Kirrane, Sabrina and Ngomo, Axel-Cyrille Ngonga},
  booktitle = {The Semantic Web – ISWC 2023},
  year      = {2023}
}
```

```bibtex
@inproceedings{NayyeriVKAWBL22,
  author    = {Mojtaba Nayyeri and Sahar Vahdati and Md\,Tansen\,Khan and Mirza\,Mohtashim\,Alam and Lisa\,Wenige and Andreas\,Behrend and Jens\,Lehmann},
  title     = {Dihedron Algebraic Embeddings for Spatio‑Temporal Knowledge Graph Completion},
  booktitle = {The Semantic Web – 19th International Conference (ESWC 2022), Hersonissos, Crete, Greece, May 29 – June 2, 2022, Proceedings},
  series    = {Lecture Notes in Computer Science},
  volume    = {13261},
  pages     = {253--269},
  year      = {2022},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-06981-9_15}
}

```

```
::contentReference[oaicite:0]{index=0}
```

