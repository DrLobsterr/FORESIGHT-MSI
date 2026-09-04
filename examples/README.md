# Minimal example

Generate a deterministic, non-study 256x256 ion-like image, binary mask, and
LabelMe-compatible polygon annotation:

```bash
python scripts/generate_synthetic_example.py
```

The generated image is only a software smoke test. It is not biological data
and must not be used to assess scientific performance.

The generated annotation can be used to check the preprocessing commands:

```bash
python scripts/labelme_to_mask.py \
  examples/synthetic/synthetic_labelme.json \
  examples/synthetic/converted_mask.png
python scripts/prepare_training_pair.py \
  --image examples/synthetic/synthetic_ion.png \
  --mask examples/synthetic/converted_mask.png \
  --output-dir examples/synthetic/prepared_pair
```
