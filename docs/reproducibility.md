# Reproducibility and provenance

## Reported training environment

The archived training logs record:

- Python 3.7.11;
- PyTorch 1.13.1+cu117;
- NVIDIA GeForce RTX 3090 with approximately 24 GB VRAM;
- 100 maximum epochs, early-stopping patience 7;
- AdamW, learning rate 1e-4, weight decay 1e-4;
- five-epoch linear warmup followed by cosine annealing to 1e-6;
- batch size 16 and no automatic mixed precision.

## Dataset partitions

The historical loader used scikit-learn's `train_test_split` with seed 42 and
a 70/15/15 partition. It consumed filesystem order rather than a sorted list.
The supplied manifests record the split reconstructed from the preserved local
corpora. Because the original directory order was not saved in the training
logs, identical historical ordering cannot be independently proven. The
manifests freeze the preserved release-time order and should accompany any
shared image archive so future runs do not depend on directory ordering.

## Randomness and exact retraining

The historical data-loader entry point seeded Python's `random` module and
NumPy with 42 before creating the 70/15/15 split. It did not explicitly seed
PyTorch/CUDA, and the training masks were generated online in worker processes.
Consequently, the archived source and configuration reproduce the reported
method, but a fresh historical run is not expected to produce bitwise-identical
weights. The cleaned training command seeds Python, NumPy, PyTorch, the
DataLoader, and each epoch's online masks so that future runs are deterministic
under the same software and hardware stack. This is a reproducibility safeguard,
not a claim that the original stochastic trajectory was recorded.

## Representative-ion selection

The historical imzML script sampled at most 500 spectra with Python's
`random.sample`, evaluated every tenth reference m/z value within a tolerance
of 0.02, required at least ten intensity observations, and retained the top 200
quality scores. No random seed was set in that script. The public command adds
an explicit `--seed` (default 42) for prospective reproducibility. The exact CSV
lists used for SCiLS Lab export were not present in the code archive inspected
for this release; if recovered, they should be deposited with the research data
and cited by checksum. Without those lists, the historical ion-selection draw
cannot be reconstructed exactly from the script alone.

## Training augmentation

The executable training code applies horizontal flipping with probability
0.30. It does not apply random rotation or random scaling. This repository does
not add transformations that were absent from the original training run.

## Historical mask behavior

Training used `legacy_v4_block_mask`. The target was computed as the nominal
ratio multiplied by the detected foreground count. Candidate 1x1, 2x2, and 3x3
blocks were shuffled and selected until their cumulative area reached that
target. Overlap and partial background coverage were not subtracted from the
cumulative area; therefore the realized foreground ratio is not guaranteed to
equal the nominal ratio.

For unambiguous new validation, `exact_foreground_block_mask` returns exactly
the requested number of foreground pixels and records both foreground and
whole-image ratios.

## Historical loss behavior

The original loss code evaluated both predicted and target VGG16 features
inside `torch.no_grad()`. Thus the VGG feature term affected the displayed and
validation loss values, including early stopping, but contributed no gradient
to model parameters. The archived checkpoints therefore learned through the
masked L1 gradient. `legacy_detached_perceptual=true` preserves that behavior.

Changing this option to `false` is a method change and will not reproduce the
reported checkpoints; results from such a model must be labeled as a new
experiment.

## Metrics

The canonical public evaluation uses scikit-image SSIM. Older internal scripts
contained simplified global and average-pooling implementations; they are not
used by the public command. Full-image results include the black canvas and are
reported alongside repair-region and tissue-foreground results so that the
scope is transparent.
