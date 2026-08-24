"""Pure prompt fragments shared by ChangeAgent runtime and SFT tooling."""

from __future__ import annotations


INITIAL_RESPONSE_SUFFIX = (
    "Decision stage is initial. Return valid JSON matching ChangeInitialResult only. "
    "Set agent_name to change_agent and status to completed."
)


def evidence_label(role: str) -> str:
    """Return the stable human-readable label for one evidence role."""
    if role == "raw_full_t1":
        return "AUTHORITATIVE RAW T1 - earlier full scene"
    if role == "raw_full_t2":
        return "AUTHORITATIVE RAW T2 - later full scene"
    if role == "proposal_overlay":
        return "AUXILIARY PROPOSAL OVERLAY - attention guidance only; not proof of change"
    if role == "transient_context_t1":
        return "TRANSIENT CONTEXT T1 - expanded same-location context"
    if role == "transient_context_t2":
        return "TRANSIENT CONTEXT T2 - expanded same-location context"
    if role.startswith("building_rescue:"):
        temporal = role.rsplit(":", 1)[-1]
        if temporal == "t1":
            return (
                "T1 SAME-LOCATION VISUAL EVIDENCE - the thin marked box is the exact candidate ROI; "
                "the pixels inside it are authoritative"
            )
        if temporal == "t2":
            return (
                "T2 SAME-LOCATION VISUAL EVIDENCE - the thin marked box is the exact same candidate ROI; "
                "the pixels inside it are authoritative"
            )
    if ":" in role:
        proposal_id, crop_role = role.split(":", 1)
        if crop_role == "reference_t1_crop":
            return f"CANDIDATE {proposal_id} - T1 reference crop - inspect the same location"
        if crop_role in {"t2_registered_crop", "t2_raw_fallback_crop"}:
            return f"CANDIDATE {proposal_id} - T2 comparison crop - inspect the same location"
    return f"AUXILIARY {role} - attention evidence only"
