#!/bin/bash

# Image-only feature extraction
python scripts/extract_feat_LP_densenet.py \
  --input /path/to/images \
  --output ./outputs/features \
  --checkpoint ./outputs/pretrain_densenet121/best_encoder_only.pth



###With ROI

#!/bin/bash

# ROI-based feature extraction with foreground masks
python scripts/extract_feat_LP_densenet.py \
  --input /path/to/AMOS-clf-tr-val/images \
  --masks_path /path/to/AMOS-clf-tr-val/fg_masks \
  --output ./outputs/features/AMOS_ROI \
  --checkpoint ./outputs/pretrain_densenet121/best_encoder_only.pth