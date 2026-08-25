# ChangeAgent SFT target-contract V2 corpus migration

Recorded: 2026-08-25

## Old corpus audit

```text
old=/home/user/cooper/posttrain_formal_prep/sft_corpus/v1
train_sha256=4ed0ea5f432737c8ae7bf6b6b4b8e5bf52516a337d33b9fd6541aaa2bc008918
validation_sha256=89c66e64588180afe1832961e43243b0d595e8d9b0d61bca94d74e9e7131bf7d
manifest_sha256=59b4979ee658c000fa606135df4d586c8a5313a41c66b43ac08867e2c644a2cb
rows=108956
legacy_evidence_rows=108956
empty_evidence_rows=108956
nonempty_evidence_rows=0
confidence_fields=0
other_noncanonical_diff_count=0
schema_type_errors=0
```

The old directory is retained and marked `STALE_TARGET_CONTRACT_DO_NOT_TRAIN`.
The safe migrated mirror under
`/home/user/cooper/posttrain_target_contract_v2/migrated_reference` is for audit
only and is not an authoritative training corpus.

## Authoritative source rebuild

```text
new=/home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence
source_spec_sha256=f619fe38b4f6a7ed6906e1283f9bb05416476f9171b826a127865735e42c59d6
pair_registry_sha256=a1ed5089db3d0bc875bf2622aace21257f3ed0dc38f8c65eaf849fc461096b54
exclusion_sha256=f6788a3df8f62a0bd31369966ea29128227dbf11c62af8e3686bb6254f5101ec
prompt_sha256=5ca66d02fafc722b9946734dd09a8fcdb26a15b15b451d7a104ed088eedad54f
builder_commit=a917c935369ab01b85aeb0cb2944377d80704834
builder_tree=68713f267371d7c345e309fa9cf6d603a139a1aa
train_sha256=626958265486660783e7caab336a91113057c59d979b3f57b2e2e0e46f3a18a7
validation_sha256=d709c8e01f4ced952aaf198efb0652bd72014749913bb1f0fff4f13925428297
manifest_sha256=b3e891d18a6d564c843af38aac29f5a05f58f947bae38477b240d02d36786587
```

Counts and invariants:

```text
train episodes=102758
validation episodes=6198
train unique pairs=6465
validation unique pairs=1242
train/validation pair intersection=0
change_caption=70746
change_qa=38210
511 unique/mapped=511/511
missing episode IDs=0
added episode IDs=0
unexpected diff rows=0
unexpected diff count=0
remaining evidence keys=0
remaining confidence keys=0
```

The old/new `rejected.jsonl`, `pair_registry.jsonl`,
`changechat_row_map.jsonl`, and `source_summary.json` are byte-identical. Every
episode matches the allowlisted transformation: schema 1 to 2, target contract
version insertion, removal of empty legacy `evidence`, and manifest/hash updates.

Evidence:

```text
old audit SHA256=d64f320f20937c28509f6301d962a39abe4d83f1a634cdaf9068bf5d2facecfd
V2 field audit SHA256=efa023d0e2307826c03ff6ad1d4809e3e2398120337ba601cf0449849c8af127
corpus diff SHA256=960f16780b23da9e2c563d89de20bf9ee082751d0b50fa6b913af155a7092db4
```
