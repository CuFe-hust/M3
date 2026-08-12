"""VQA object-evidence subpackage; re-exports only, no business logic.

VQA 对象证据子包；仅 re-export，不包含任何业务逻辑。该子包归属
GeneralVQAAgent 内部领域行为；不得创建 agents/object_evidence/ 或第二个
通用证据 Agent。
"""

from agents.general_vqa.evidence.schema import (
    EvidenceLayer,
    EvidenceState,
    LayerStateRecord,
    ModelCallAudit,
    RoiEvidenceRecord,
    SegFormerEvidenceRecord,
    VqaEvidenceBundle,
    YoloDetectionRecord,
)

__all__ = [
    "EvidenceLayer",
    "EvidenceState",
    "LayerStateRecord",
    "ModelCallAudit",
    "RoiEvidenceRecord",
    "SegFormerEvidenceRecord",
    "VqaEvidenceBundle",
    "YoloDetectionRecord",
]
