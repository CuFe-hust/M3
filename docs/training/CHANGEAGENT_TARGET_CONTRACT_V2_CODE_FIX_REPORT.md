# ChangeAgent SFT target-contract V2 code fix

Recorded: 2026-08-25  
Baseline after main merge: `cc372c860751d3c7a11d0854414f04139dd7b8e4`  
Qualified training-code commit: `a917c935369ab01b85aeb0cb2944377d80704834`

## Result

```text
TARGET_CONTRACT_CODE_GATE=PASS
TARGET_WRITER_GATE=PASS
TARGET_PROFILE_STRICT_GATE=PASS
TARGET_MANIFEST_IDENTITY_GATE=PASS
FORMAL_TRAINING_STARTED=false
```

The shared target helper binds episode schema 2 to
`change_initial_result_v2_no_legacy_evidence` and derives its descriptor from
the serialization JSON schema of `ChangeInitialResult` and the public fields of
`VisualEvidence`. Writers emit canonical results only. Every formal row carries
the contract version; manifests carry the identity SHA; checkpoint/resume data
identity inherits the same contract.

The formal profile rejects schema V1, legacy `evidence`, legacy `confidence`, a
missing contract identity, and any identity mismatch. Assistant supervision and
metadata serialize the canonical copy, never the untrusted raw target.

Runtime legacy-read compatibility remains intact: historical `evidence` and
`confidence` inputs validate and are omitted from `model_dump()`. No legacy
field was restored to the public runtime output contract.

The formal multi-source builder keeps source spec, pair ID, official split, 511
exclusions, deduplication, episode ID, question, answer, images, request payload,
and provenance unchanged. It writes to a new directory atomically, refuses
overwrite, and records builder commit/tree in the manifest.

Relevant implementation and test commits:

```text
140875c fix: migrate ChangeAgent SFT target contract to v2
7dbcf89 fix: bootstrap Change SFT maintenance CLIs
a917c93 posttrain: bind formal corpus to builder identity
```

Targeted contract/profile/writer/migration tests passed. The final Spark full
suite passed with 2525 tests and 6 pre-existing warnings.
