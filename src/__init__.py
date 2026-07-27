"""
Molecular subtype classification from whole-slide images (AN2DL 2025/26, Challenge 2).

Reference implementation of the submitted solution, extracted from
`notebooks/02_full_pipeline.ipynb` so the models and the pipeline can be read and
diffed without opening the notebook.

The notebooks remain the authoritative record of what was executed on Colab: they
carry the training logs of the submitted run.
"""

__all__ = ["config", "data", "models", "inference", "ensemble", "submission"]
__version__ = "1.0.0"
