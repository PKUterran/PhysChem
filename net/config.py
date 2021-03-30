from enum import Enum


class ConfType(Enum):
    NONE = 0,
    RDKIT = 1,
    NEWTON = 2,
    ONLY = 3,
    NEWTON_RGT = 4,
    REAL = 5,
