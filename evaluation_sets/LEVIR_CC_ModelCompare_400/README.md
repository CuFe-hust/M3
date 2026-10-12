# LEVIR_CC_ModelCompare_400

Fixed LEVIR-CC validation subset for early model selection.

- 400 bi-temporal image pairs
- 200 change pairs
- 200 no-change pairs
- 5 original reference sentences per pair
- seed: 42

The formal `LevirCCcaptions.json` contains unmodified source records. Derived
sampling information appears only in auxiliary files. Image A is the pre-phase
image and image B is the post-phase image.

Run independent validation:

```bash
python scripts/validate_levir_cc.py \
  --source-root "/home/user/下载/datasets/levir_cc/Levir-CC-dataset" \
  --dataset-root "/home/user/silverdew/LEVIR_CC_ModelCompare_400" \
  --write-report
```
