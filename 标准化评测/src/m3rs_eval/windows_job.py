"""Native Windows Job Object ownership for timed external commands."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Any


class WindowsJobError(RuntimeError):
    """Raised when Windows process ownership cannot be established safely."""


_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class WindowsJobController:
    """Own a process tree; closing the job kills all assigned processes on Windows."""

    def __init__(self, handle: int) -> None:
        self._handle = handle
        self._closed = False

    @classmethod
    def create(cls) -> "WindowsJobController":
        if os.name != "nt":
            raise WindowsJobError("Windows Job Objects are unavailable on this platform")
        kernel32 = _kernel32()
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise WindowsJobError(f"CreateJobObjectW failed: {ctypes.get_last_error()}")
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            wintypes.HANDLE(handle),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            kernel32.CloseHandle(wintypes.HANDLE(handle))
            raise WindowsJobError(f"SetInformationJobObject failed: {ctypes.get_last_error()}")
        return cls(int(handle))

    def assign(self, process: Any) -> None:
        if self._closed:
            raise WindowsJobError("cannot assign a process to a closed Job Object")
        process_handle = getattr(process, "_handle", None)
        if process_handle is None:
            raise WindowsJobError("Popen process has no Windows handle for Job Object assignment")
        if not _kernel32().AssignProcessToJobObject(
            wintypes.HANDLE(self._handle), wintypes.HANDLE(process_handle)
        ):
            raise WindowsJobError(f"AssignProcessToJobObject failed: {ctypes.get_last_error()}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if os.name == "nt" and not _kernel32().CloseHandle(wintypes.HANDLE(self._handle)):
            raise WindowsJobError(f"CloseHandle(Job Object) failed: {ctypes.get_last_error()}")


def _kernel32():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE, wintypes.INT, wintypes.LPVOID, wintypes.DWORD
    )
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32
