"""M3 bridge for the challenge participant API (Protocol 2.0).

挑战杯评测 M3 桥接包：把 Protocol 2.0 单题请求映射到 M3 的
SampleRunner.run_one() 直连调用，并规范化答案。
"""

from .errors import AdapterError

__all__ = ["AdapterError"]
