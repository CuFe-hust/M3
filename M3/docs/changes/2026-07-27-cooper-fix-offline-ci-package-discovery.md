# Modification Note: Fix Offline CI Package Discovery - 2026-07-27 10:38:13 +08:00

## Modification Time

2026-07-27 10:38:13 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Make the offline GitHub Actions dependency-installation step succeed by preventing setuptools
from treating non-package repository directories as distributable top-level packages.

## Modified Files

- `pyproject.toml`
- `tests/test_packaging.py`
- `DETAILS.md`
- `.gitignore`

## Core Changes

- Configured setuptools package discovery to include only `spacers_agent*`.
- Declared the existing `Pillow>=10.0.0` runtime dependency in `pyproject.toml`, matching
  `requirements.txt` and the imports exercised by the offline suite.
- Added offline pytest assertions that protect the package-discovery and Pillow-dependency
  contracts.
- Documented the editable-installation and offline-dependency boundary used by the workflow.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

Yes. Packaging discovery is now explicit and the existing Pillow runtime dependency is declared;
runtime configuration keys and defaults are unchanged.

## Whether Evaluation Was Affected

No metric, split, reference-answer reader, or result post-processing rule changed.

## Whether Deployment Was Affected

No. Optional YOLO dependencies and deployment paths remain unchanged.

## Whether pytest Was Updated

Yes. Added `tests/test_packaging.py` to protect the setuptools package-discovery configuration.

## Whether .gitignore Was Updated

Yes. Added `*.egg-info/` so editable-install build metadata is not committed.

## Validation Method

- Reproduced the original editable-install failure in a clean Python 3.11 virtual environment.
- `python -m pip install --isolated -e ".[dev]"` succeeded in that clean environment after the
  final metadata updates.
- Python 3.11 `compileall`, CLI help, and `pytest -q --basetemp <writable-temp> -p
  no:cacheprovider` passed in the clean environment: 365 tests.
- `git diff --check` passed.
- GitHub Actions run 3 exposed the missing `PIL` import dependency; the final rerun after the
  Pillow metadata update is pending push.

## Risks and Follow-up TODOs

- This change limits distribution contents to the runtime package only; legacy top-level source
  directories remain available in the repository but are intentionally not installed as packages.
