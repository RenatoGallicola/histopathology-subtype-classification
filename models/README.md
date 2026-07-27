# Model weights

The trained checkpoints are **not committed** — the ResNet18 alone is 45 MB, and
the cached UNI embeddings another 12 MB.

They are not needed to reproduce the model-selection stage: the per-slide
out-of-fold probabilities of both branches are committed under `artifacts/`, and
`scripts/ablation.py` and `scripts/tune_alpha.py` read only those.

## Weights the pipeline produces

| File | Produced by | Purpose |
|---|---|---|
| `resnet18_full_bestalpha.pth` | `notebooks/02_full_pipeline.ipynb`, *refitting on all training slides* | ResNet18 tile classifier, refit on 100% of the training slides |
| `uni_clf_full_bestalpha.pth` | same section | The linear head over UNI embeddings, refit on 100% |
| `uni_feats_train_fp16.npy` | *caching the UNI embeddings* | 1024-dimensional embeddings for every training tile |
| `uni_feats_test_fp16.npy` | same section | The same for the test tiles |

The per-fold checkpoints exist only to generate the out-of-fold probabilities in
`artifacts/`. Once that file exists they are disposable.

## Regenerating them

Open `notebooks/02_full_pipeline.ipynb` on a GPU runtime, add the tile archives
produced by `01_preprocessing_and_tiles.ipynb`, and run top to bottom.

You will also need a Hugging Face account approved for
[UNI](https://huggingface.co/MahmoodLab/UNI): the weights are gated, and the
notebook authenticates before downloading them. Caching the embeddings is the
expensive step — it runs the frozen ViT once over every tile — but it happens
once, after which each cross-validation fold costs only the time to fit the head.

Expect the numbers to differ slightly. cuDNN kernel selection is not deterministic
even with the seed fixed in `src/config.py:set_seed`, and the ensemble weight is
sensitive to small changes in the out-of-fold probabilities — see the plateau
discussion in `docs/RESULTS.md`.
