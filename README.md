# FORESIGHT-MSI

FORESIGHT (Fourier-optimized Reconstruction of Spatial Imaging for
High-resolution Tissue) reconstructs damaged MALDI-TOF mass-spectrometry ion
images of wheat grain sections. This repository accompanies the published article
[“Refining the functional dynamics of deoxynivalenol in wheat Fusarium head
blight via MALDI-TOF-MSI and deep learning”](https://doi.org/10.1016/j.foodchem.2026.151138).

**Food Chemistry, 529 (2026), Article 151138.**
Available online: 13 September 2026.
DOI: [10.1016/j.foodchem.2026.151138](https://doi.org/10.1016/j.foodchem.2026.151138).

## What is included

- representative-ion selection from imzML data;
- black-canvas image preparation and LabelMe polygon conversion;
- training-pair conversion with a documented mask convention;
- the checkpoint-compatible FORESIGHT architecture;
- deterministic data splits and experiment configurations;
- model training, inference, reconstruction, and performance evaluation;
- full-image, reconstruction-region, and tissue-foreground metrics.

Raw imzML/ibd files, SCiLS Lab projects, copyrighted software, training logs,
intermediate checkpoints, and manuscript working files are intentionally not
part of the source repository. See [data/README.md](data/README.md) and
[checkpoints/README.md](checkpoints/README.md).

## Model architecture

The implementation used for the reported checkpoints is a full-resolution
network. A 3x3 input projection maps the three RGB channels to 128 hidden
channels. Four residual Fourier blocks retain eight low-frequency modes in a
learned complex spectral convolution, followed by a 1x1 channel-mixing
convolution. A two-layer output projection returns three channels. Dropout is
0.10. The predicted values replace only pixels for which the repair mask is 1.

This description follows the executable training code. The implementation does
not contain a separate down-sampling encoder, up-sampling decoder, or explicit
local/global channel split.

## Mask convention and ratio

All public commands use one convention:

```text
1 (white) = reconstruct this pixel
0 (black) = preserve this pixel
```

The nominal mask ratio is defined relative to detected tissue foreground, not
the black canvas. The training checkpoints used the historical `legacy-v4`
random 1–3 pixel block generator. Because legacy blocks can overlap and can
partly cross the foreground boundary, their realized ratio is recorded and may
differ from the nominal ratio. Supplementary Figure S10 used the deterministic
`exact` generator, which masks exactly 40% of detected foreground pixels.

## Installation

### Historical CUDA environment

The training logs record Python 3.7.11, PyTorch 1.13.1+cu117, CUDA, and an NVIDIA
GeForce RTX 3090. To reproduce that software environment on Linux:

```bash
conda env create -f environment-reproduction.yml
conda activate foresight-reproduction
python -m pip install -e .
```

### Current macOS inference environment

```bash
conda env create -f environment-macos.yml
conda activate foresight-macos
python -m pip install -e .
```

CPU inference is the default on macOS because complex FFT support on MPS varies
across PyTorch releases.

Generate a non-biological smoke-test image and mask with:

```bash
python scripts/generate_synthetic_example.py
```

## Data preparation

Rank the top 200 ion features from one or more imzML files:

```bash
python scripts/select_mz_values.py sample.imzML \
  --output-dir selected_ions --top-k 200 --sample-pixels 500 --seed 42
```

The seed makes new runs deterministic. The historical selector did not record
a seed; see [docs/reproducibility.md](docs/reproducibility.md).

Add a square black canvas:

```bash
python scripts/add_black_margin.py input.png output_margin.png --margin-ratio 0.1
```

Convert a LabelMe polygon JSON file to a binary mask and then to NumPy tensors:

```bash
python scripts/labelme_to_mask.py damage.json damage_mask.png
python scripts/prepare_training_pair.py \
  --image output_margin.png --mask damage_mask.png --output-dir prepared_pair
```

## Training

Place one experiment group's intact ion images in a directory. Use the supplied
split manifest when the filenames are available:

```bash
python scripts/train.py \
  --config configs/14d1_mask40.json \
  --data-dir /path/to/14d1_images \
  --manifest manifests/14d1_split.csv \
  --output-dir runs/14d1_mask40
```

The manuscript models contain 1,000 images for 14d1, 1,200 for 17d1, and 900
for 17d2. Training ratios were 40%, 50%, 60%, and 70%. Copy
`configs/14d1_mask40.json` and change only the model name, expected count, and
mask ratio for the other combinations.

## Reconstructing a manually annotated image

```bash
python scripts/reconstruct.py \
  --model /path/to/14d1_mask40_best_model.pth \
  --image output_margin.png \
  --mask damage_mask.png \
  --output-dir reconstruction --device cpu
```

PNG/JPEG and the NumPy arrays produced by `prepare_training_pair.py` are both
accepted.

## Evaluation and Supplementary Figure S10

Evaluate intact images with deterministic masks:

```bash
python scripts/evaluate.py \
  --model /path/to/14d1_mask40_best_model.pth \
  --input-dir /path/to/five_independent_DON_images \
  --reference-data-dir /path/to/14d1_training_corpus \
  --output-dir results/five_DON \
  --mask-ratio 0.4 --mask-algorithm exact \
  --seed 20260821 --device cpu
```

The reference directory is optional but recommended: evaluation stops if an
input is pixel-identical to an image in the reference corpus after
preprocessing.

Metric scopes are explicit in the CSV/JSON output:

- `full`: every pixel in the 256x256 RGB canvas, including black background;
- `repair_region`: only pixels selected for reconstruction;
- `tissue_foreground`: only pixels identified by the published foreground rule.

MSE and PSNR use images scaled to [0,1]. SSIM uses
`skimage.metrics.structural_similarity(channel_axis=2, data_range=1.0)`.

## Reproducibility notes

Read [docs/reproducibility.md](docs/reproducibility.md) before comparing new
training runs with archived checkpoints. The public names are 14d1, 17d1, and
17d2. Historical development-directory labels are not used as scientific group
names.

## Citation

If you use FORESIGHT-MSI, please cite the published article:

> Tang, M., He, W., Tian, Y., Sun, S., Zhao, C., Guo, M., Yan, Z., Chu, Q.,
> Liu, N., Yu, D., Zhang, J., & Wu, A. (2026). Refining the functional
> dynamics of deoxynivalenol in wheat Fusarium head blight via MALDI-TOF-MSI
> and deep learning. *Food Chemistry, 529*, Article 151138.
> https://doi.org/10.1016/j.foodchem.2026.151138

Citation metadata is provided in [CITATION.cff](CITATION.cff), and a BibTeX
entry is available in [CITATION.bib](CITATION.bib).
For research-data and model-checkpoint information, see
[data/README.md](data/README.md) and [checkpoints/README.md](checkpoints/README.md).
