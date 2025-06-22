# DISCO (Vehicle + LP Fusion)

This README documents the new functionality contributed by the thesis work on top of the original **DISCO curb-side monitoring** repository.

## What was added

| # | Component | Purpose | Key Files / Classes |
|---|-----------|---------|---------------------|
| **1** | **Transformer-based vehicle embeddings&nbsp;(TransReID)** | Replaces ResNet-50/`ft_net` features used by **StrongSORT** and yields more discriminative global descriptors under occlusion and viewpoint change. | `core/tracking.py` → `_initialize_transreid` |
| **2** | **License-plate Re-Identification** | Detects plates inside each vehicle crop (YOLOv8) and embeds them with a lightweight **FastReID** / **TransReID** head trained on **UFPR-ALPR**. | `core/lp_reid.py` |
| **3** | **Vehicle + Plate score-level fusion** | When a plate is detected with confidence ≥ 0.75, vehicle- and plate-level cosine-similarity scores are fused **before** association in StrongSORT. | `core/fusion.py` |

## Runtime Flags (defined in `core/tracking.py`)

| Variable | Allowed values | Effect |
|----------|----------------|--------|
| `lp_reid` | `True` / `False` | Toggle the license-plate ReID branch. |
| `lp_model_type` | `"transreid"` / `"fastreid"` | Choose the embedding backbone for plate crops. |
| `lp_only` | `True` / `False` | Use only plate descriptors (ignore vehicle features). |
| `veh_model_type` | `"transreid"` / `"ft_net"` | Choose the embedding backbone for vehicle crops. |
| `score_fusion` | `True` / `False` | Enable or disable score-level fusion between vehicle and plate descriptors. |
