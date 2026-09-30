"""Business Entity Resolution pipeline (Amazon ML Challenge 2026).

Stages: normalize -> multi-channel blocking -> learned meta-blocking -> pair features ->
LightGBM/XGBoost pass-1 -> collective pass-2 -> entity model -> per-entity expected-F0.5 decoding.
Entry points: ``python -m src.run`` (train on train/, predict test/) and ``python -m src.evaluate``.
"""
__version__ = "1.0.0"
