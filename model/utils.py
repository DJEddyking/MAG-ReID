

from enum import Enum


class InputImageType(Enum):
    rgb: str = 'rgb'
    sketch: str = 'sketch'
    color_pencil: str = 'color_pencil'
    nir: str = 'nir'