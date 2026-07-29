# VRSBench_ModelCompare_1100

This directory is a deterministic 1,100-record subset of the official VRSBench
validation annotations for model comparison.

- Caption: 250 records
- Referring: 250 records
- VQA: 600 records
- Seed: 42

Records in the three official JSON files are copied without field changes.
Derived sampling and difficulty information is stored only in auxiliary files.
Image reuse is allowed; selection merely prefers image IDs not already used.
The difficulty labels are task-specific heuristics, not official VRSBench labels.

Re-run validation:

```bash
python scripts/validate_vrsbench.py \
  --source-root "/home/user/下载/datasets/vrsbench" \
  --dataset-root "/home/user/silverdew/VRSBench_ModelCompare_1100" \
  --write-report
```

VRSBench text annotations are published under CC-BY-4.0. Some underlying images
come from DOTA and are restricted to academic use. This subset does not alter
the licenses of the source annotations or images.
