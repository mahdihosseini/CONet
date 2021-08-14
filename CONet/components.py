from typing import NamedTuple, List
from enum import Enum

# import logging


# class LogLevel(Enum):
#     '''
#     What the stdlib did not provide!
#     '''
#     DEBUG = logging.DEBUG
#     INFO = logging.INFO
#     WARNING = logging.WARNING
#     ERROR = logging.ERROR
#     CRITICAL = logging.CRITICAL
#
#     def __str__(self):
#         return self.name


class LayerType(Enum):
    CONV = 1
    FC = 2
    NON_CONV = 3


class IOMetrics(NamedTuple):
    mode_12_channel_rank: List[float]
    mode_12_channel_S: List[float]
    mode_12_channel_condition: List[float]
    input_channel_rank: List[float]
    input_channel_S: List[float]
    input_channel_condition: List[float]
    output_channel_rank: List[float]
    output_channel_S: List[float]
    output_channel_condition: List[float]
    fc_S: float
    fc_rank: float


class LRMetrics(NamedTuple):
    rank_velocity: List[float]
    r_conv: List[float]

class Statistics(NamedTuple):
    ram: float
    gpu_mem: float
    epoch_time: float
    step_time: float
