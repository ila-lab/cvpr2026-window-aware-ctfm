# CVPR 2026 Team `ilalab` Solution

This repository provides the implementation of our solution for the CVPR 2026 Foundation Models for General CT Image Diagnosis Challenge.

Task: Task 1 — Linear Probing
Track: Coreset Track
Model: DenseNet121 3D encoder
Training strategy: Window-aware and multi-scale global-local self-supervised contrastive learning

The goal is to train a CT foundation model from scratch using only 10% of the pretraining data, and then use the pretrained encoder to extract CT feature embeddings for downstream linear probing.

1. Method Overview

We train a DenseNet121 3D encoder using self-supervised contrastive learning.

The main components are:

DenseNet121 3D encoder
Window-aware CT augmentation
Multi-scale global and local views
Global-global contrastive learning
Global-local contrastive learning
Feature extraction for downstream linear probing

During pretraining, each CT volume is used to generate:

2 global views
3 local views

The local patch sizes are:

48 × 48 × 48
80 × 80 × 80
104 × 104 × 104

The model is trained to make different views from the same CT scan closer in feature space, while separating features from different patients.

2. Repository Structure
cvpr2026-window-aware-ctfm/
├── models/
│   ├── __init__.py
│   └── densenet121_3d_encoder.py
│
├── scripts/
│   ├── __init__.py
│   ├── pretrain_densenet121.py
│   └── extract_feat_LP_densenet.py
│
├── examples/
│   ├── run_lp_densenet_all.sh
│   └── run_inference_lp_densenet_all.sh
│
├── README.md
├── requirements.txt
└── .gitignore
File	Description
models/densenet121_3d_encoder.py	DenseNet121 3D encoder architecture
scripts/pretrain_densenet121.py	Self-supervised pretraining script
scripts/extract_feat_LP_densenet.py	Feature extraction script for linear probing
examples/run_lp_densenet_all.sh	Example script for downstream linear probing
examples/run_inference_lp_densenet_all.sh	Example script for downstream LP inference
requirements.txt	Required Python packages
.gitignore	Files and folders excluded from Git tracking
3. Environment

The code was developed in a Python and PyTorch environment with GPU support.

Recommended environment:

Python >= 3.10
PyTorch 2.12.0 with CUDA 13.0
torchvision 0.27.0 with CUDA 13.0
MONAI 1.5.2
SimpleITK 2.5.3
NumPy 2.4.4
pandas 3.0.2
SciPy 1.17.1
h5py 3.16.0
tqdm 4.67.1
PyYAML 6.0.3
scikit-learn 1.8.0

Install dependencies with:

pip install -r requirements.txt

The PyTorch packages are pinned with the +cu130 suffix to avoid installing a different CUDA build when package versions are updated.

4. Dataset Structure
4.1 Coreset Data for Self-supervised Pretraining

For self-supervised pretraining, the input data should contain CT volumes in .nii.gz format.

In our experiment, the selected coreset was organized as:

coreset_data2/
└── images/
    ├── Adrenal_Ki67_Seg_001_0000.nii.gz
    ├── Adrenal_Ki67_Seg_002_0000.nii.gz
    ├── Adrenal_Ki67_Seg_004_0000.nii.gz
    └── ...

The --data_root argument should point to the folder containing the CT volumes:

python -m scripts.pretrain_densenet121 \
  --data_root /path/to/coreset_data2/images \
  --save_dir ./outputs/pretrain_densenet121

The pretraining script searches for .nii.gz files under the specified --data_root.

4.2 Downstream Data for Feature Extraction

For downstream linear probing, this repository provides a feature extraction script.
The script does not train the downstream classifier. It only extracts .h5 feature files from CT volumes.

The feature extraction script reads images from the folder specified by --input.
For ROI-based tasks, it also reads masks from the folder specified by --masks_path.

AMOS Classification Tasks

The AMOS downstream dataset is organized as:

AMOS-clf-tr-val/
├── fg_masks/
├── images/
├── images_by_target/
├── labels/
└── test_demo/

For image-only feature extraction:

python -m scripts.extract_feat_LP_densenet \
  --input /path/to/AMOS-clf-tr-val/images \
  --output ./outputs/features/AMOS \
  --checkpoint ./outputs/pretrain_densenet121/best_encoder_only.pth

For ROI-based feature extraction with foreground masks:

python -m scripts.extract_feat_LP_densenet \
  --input /path/to/AMOS-clf-tr-val/images \
  --masks_path /path/to/AMOS-clf-tr-val/fg_masks \
  --output ./outputs/features/AMOS_ROI \
  --checkpoint ./outputs/pretrain_densenet121/best_encoder_only.pth

In ROI-based mode, the mask filename should match the image filename.

COVID-CT and LUNA25 Tasks

The COVID-CT and LUNA25 downstream datasets are organized as:

CVPR26_data/
├── COVID-CT/
│   ├── images/
│   └── labels/
└── LUNA25/
    ├── images/
    └── labels/

For COVID-CT feature extraction:

python -m scripts.extract_feat_LP_densenet \
  --input /path/to/CVPR26_data/COVID-CT/images \
  --output ./outputs/features/COVID-CT \
  --checkpoint ./outputs/pretrain_densenet121/best_encoder_only.pth

For LUNA25 feature extraction:

python -m scripts.extract_feat_LP_densenet \
  --input /path/to/CVPR26_data/LUNA25/images \
  --output ./outputs/features/LUNA25 \
  --checkpoint ./outputs/pretrain_densenet121/best_encoder_only.pth

COVID-CT and LUNA25 are image-only tasks in this pipeline, so --masks_path is not required.

5. Coreset Preparation

In the Coreset Track, only 10% of the pretraining data can be used.

This repository assumes that the coreset has already been prepared. The pretraining script does not perform coreset selection automatically. It directly loads CT volumes from the folder specified by --data_root.

In our experiment, the final coreset contained 1082 CT volumes and was organized under:

coreset_data2/images/

The coreset was selected using an anatomy-aware and dataset-balanced sampling strategy. The CT volumes were grouped by anatomical region and dataset source. The main anatomy groups included:

Abdomen / Pelvis
Chest
Head
PET / Whole-body
Others

The sampling pipeline used in our previous data preparation is available here:

Anatomy-aware coreset sampling pipeline

The pipeline includes:

count_groups.py
    ↓
pretty_counts.txt
    ↓
build_file_list.py
    ↓
all_files.txt
    ↓
sampling.py
    ↓
coreset_1082_equal_dataset.txt

The final sampling target was:

ANATOMY_TARGETS = {
    "Abdomen": 532,
    "Chest": 250,
    "Head": 100,
    "PET": 100,
    "Others": 100,
}

This design was used to preserve data diversity and reduce the risk of the coreset being dominated by a small number of large datasets.

6. Preprocessing
6.1 Pretraining Stage

For each CT volume, the preprocessing pipeline includes:

Read the CT volume using SimpleITK
Resample the image to 1.0 × 1.0 × 1.0 mm spacing
Center crop or pad the volume to 152 × 152 × 152
Apply window-aware augmentation
Generate global and local views

No z-score normalization is used in this version.

6.2 Window-aware Augmentation

CT images are represented by Hounsfield Unit values. Different HU windows highlight different tissues and lesions. Therefore, different CT windows are randomly applied during pretraining.

Window	Center	Width
Soft tissue	50	380
Lung	-600	1550
Bone	450	1900
Abdomen	50	380
Wide	450	2100

Additional augmentations include:

Intensity jitter
Gaussian noise
Random rotation

No flipping is used.

7. Self-supervised Pretraining

Run pretraining with:

python -m scripts.pretrain_densenet121 \
  --data_root /path/to/coreset_data2/images \
  --save_dir ./outputs/pretrain_densenet121

If no additional training arguments are specified, the script uses the default full pretraining setting.

Parameter	Value
Backbone	DenseNet121 3D
Input size	152 × 152 × 152
Local patch sizes	48, 80, 104
Number of global views	2
Number of local views	3
Epochs	80
Batch size	3
Gradient accumulation steps	5
Effective batch size	15
Learning rate	3e-4
Weight decay	1e-4
Temperature	0.2
Dropout	0.2

For a quick sanity check, use:

python -m scripts.pretrain_densenet121 \
  --data_root /path/to/coreset_data2/images \
  --save_dir ./outputs/test_pretrain \
  --epochs 1 \
  --batch_size 1 \
  --num_workers 0

The script saves:

outputs/pretrain_densenet121/
├── latest_full_model.pth
├── latest_encoder_only.pth
├── best_full_model.pth
├── best_encoder_only.pth
└── training_history.csv

For linear probing feature extraction, use:

best_encoder_only.pth
8. Feature Extraction for Linear Probing

After pretraining, the DenseNet121 3D encoder is used to extract CT feature embeddings for downstream linear probing.

The feature extraction script supports two modes:

Image-only mode: use the whole CT volume.
ROI-based mode: use a foreground mask to crop the region of interest before feature extraction.
8.1 Image-only Feature Extraction
python -m scripts.extract_feat_LP_densenet \
  --input /path/to/images \
  --output ./outputs/features \
  --checkpoint ./outputs/pretrain_densenet121/best_encoder_only.pth
8.2 ROI-based Feature Extraction with Masks
python -m scripts.extract_feat_LP_densenet \
  --input /path/to/images \
  --masks_path /path/to/masks \
  --output ./outputs/features_roi \
  --checkpoint ./outputs/pretrain_densenet121/best_encoder_only.pth

The mask filenames should match the image filenames.

8.3 Feature Extraction Preprocessing

The feature extraction pipeline includes:

Read 3D CT images using SimpleITK
Apply soft-tissue CT window
Resample to 1.0 × 1.0 × 1.0 mm spacing
Crop or pad to 152 × 152 × 152
If a mask is available, crop the ROI based on the foreground center
Extract multi-level DenseNet features
Apply adaptive average pooling
Concatenate pooled features into the final embedding

The default soft-tissue window is:

Window	Center	Width	HU range
Soft tissue	50	380	-140 to 240

Each CT scan is saved as one .h5 file:

outputs/features/
├── case001.h5
├── case002.h5
└── ...

Each .h5 file contains:

key: y_hat
value: extracted feature embedding
9. Linear Probing and Inference

This repository provides the DenseNet121 3D pretraining and feature extraction pipeline.

After extracting .h5 feature embeddings, downstream linear probing and inference are performed using the official CVPR26 3D CTFM competition scripts.

The official linear probing and inference scripts are available from:

https://github.com/kmin940/CVPR26-3DCTFMCompetition

The required external scripts are:

run_LP.py
cvpr26_inference_LP.py

Please clone or download the official repository first, and make sure these scripts are available in your working directory.
Alternatively, update the script paths in the example shell scripts according to your local environment.

9.1 Input Features

Before running linear probing, CT feature embeddings should be extracted using:

python -m scripts.extract_feat_LP_densenet \
  --input /path/to/images \
  --output ./outputs/features \
  --checkpoint ./outputs/pretrain_densenet121/best_encoder_only.pth

The output folder should contain one .h5 file for each CT scan. Each .h5 file contains the extracted embedding under the key y_hat.

9.2 Example Shell Scripts

This repository provides example shell scripts for connecting the extracted DenseNet121 feature embeddings to the official linear probing pipeline.

examples/
├── run_lp_densenet_all.sh
└── run_inference_lp_densenet_all.sh

These scripts are examples and may need path modification before running.

9.3 Linear Probing Training

Run:

bash examples/run_lp_densenet_all.sh

This script loops over the AMOS downstream classification targets and calls the official run_LP.py script for each target.

The main paths used in the script are:

EMBEDS_ROOT="/path/to/densenet_lp_features"
LABELS_ROOT="/path/to/AMOS-clf-tr-val/labels"
OUT_ROOT="/path/to/densenet_lp_results"

where:

EMBEDS_ROOT is the folder containing the extracted .h5 feature files.
LABELS_ROOT is the folder containing the downstream classification labels.
OUT_ROOT is the output folder for linear probing results and checkpoints.
9.4 Linear Probing Inference

After linear probing checkpoints are generated, inference can be performed with:

bash examples/run_inference_lp_densenet_all.sh

This script loops over the same downstream targets and calls the official cvpr26_inference_LP.py script.

The main paths used in the script are:

EMBEDS_ROOT="/path/to/densenet_lp_features"
LABELS_ROOT="/path/to/AMOS-clf-tr-val/labels"
RESULTS_ROOT="/path/to/densenet_lp_results"

where:

EMBEDS_ROOT is the folder containing extracted DenseNet121 feature embeddings.
LABELS_ROOT is the folder containing task labels.
RESULTS_ROOT is the folder containing trained linear probing checkpoints.
9.5 AMOS Classification Targets

The provided example shell scripts run linear probing and inference for the following AMOS classification targets:

adrenal_hyperplasia
ascites
atherosclerosis
cholecystitis
colorectal_cancer
fatty_liver
gallstone
hydronephrosis
kidney_stone
liver_calcifications
liver_cyst
liver_lesion
lymphadenopathy
renal_cyst
splenomegaly
9.6 Notes

Before running the example scripts, please check and update:

the path to the official CVPR26 3D CTFM repository
the extracted feature folder
the label folder
the output result folder
the checkpoint folder for inference

This repository does not modify the official linear probing implementation.
It provides DenseNet121 pretraining, feature extraction, and example scripts for connecting the extracted features to the official LP pipeline.

10. Results

The final model reported in our presentation is the DenseNet121 3D model without z-score normalization.

Model	Average Balanced Accuracy	Average AUROC
Baseline	0.622	0.612
DenseNet121 3D	0.710	0.720

Compared with the baseline, our method improves:

Average Balanced Accuracy by +8.8 percentage points
Average AUROC by +10.8 percentage points
Better-performing Tasks
Task	Balanced Accuracy	AUROC
Fatty liver	0.83	0.87
Splenomegaly	0.86	0.87
Hydronephrosis	0.75	0.75
Atherosclerosis	0.75	0.72
Lymphadenopathy	0.74	0.73
Adrenal hyperplasia	0.73	0.72

The model showed stronger performance on abdominal and soft-tissue-related tasks.

11. Reproducibility Workflow

A typical workflow is:

# 1. Install dependencies
pip install -r requirements.txt

# 2. Pretrain DenseNet121 3D encoder
python -m scripts.pretrain_densenet121 \
  --data_root /path/to/coreset_data2/images \
  --save_dir ./outputs/pretrain_densenet121

# 3. Extract features for linear probing
python -m scripts.extract_feat_LP_densenet \
  --input /path/to/images \
  --output ./outputs/features \
  --checkpoint ./outputs/pretrain_densenet121/best_encoder_only.pth

# 4. Run linear probing using the official LP pipeline
bash examples/run_lp_densenet_all.sh

# 5. Run LP inference
bash examples/run_inference_lp_densenet_all.sh

The extracted .h5 files can be used as input features for downstream linear probing.

12. Limitations

This repository provides the main pretraining and feature extraction pipeline. There are several limitations in the current version:

The linear probing feature extraction uses a single soft-tissue window by default.
This may not be optimal for lung, bone, calcification, or stone-related tasks.
Local patches are randomly cropped and may not always include lesion regions.
The coreset selection scripts are provided in a separate repository link rather than directly integrated into this repository.
13. Future Work

Potential improvements include:

Multi-window feature concatenation during feature extraction
Lung-window features for chest-related tasks
Bone-window features for calcification or stone-related tasks
Organ-aware or lesion-aware local cropping
Task-specific feature analysis for low-performing diseases
14. Acknowledgement

We thank the organizers of the CVPR 2026 Foundation Models for General CT Image Diagnosis Challenge for providing the benchmark and evaluation platform.

We also thank our team members and advisors for their support throughout the development and evaluation of this method.
# 15. Contact

**CHENG, JU YUN**

Department of Artificial Intelligence

Chang Gung University

Email: layla910104@gmail.com

**Prof. Ying-Jia Lin**

Assistant Professor, Chang Gung University

Email: yjlin@cgu.edu.tw

**Prof. Chi-Tung Cheng**

Assistant Professor, Chang Gung Memorial Hospital

Email: atong89130@gmail.com
