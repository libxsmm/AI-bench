try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


class FlopsUnit(StrEnum):
    """Control FLOPS measurement unit."""

    TFLOPS = "TFLOPS"
    GFLOPS = "GFLOPS"


class MemBwUnit(StrEnum):
    """Control memory bandwidth unit."""

    GBS = "GB/s"
    MBS = "MB/s"


class NotesSymbols(StrEnum):
    """Notes annotation symbols."""

    ESTIMATE = "⚠️"
