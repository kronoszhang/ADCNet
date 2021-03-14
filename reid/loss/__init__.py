from __future__ import absolute_import

from .triplet import TripletLoss_XT, TripletLoss, CenterLoss, CrossEntropyLoss_LabelSmooth
from .lsr import LSRLoss

__all__ = [
    'TripletLoss_XT',
    'TripletLoss',
    'LSRLoss',
    'CenterLoss',
    'CrossEntropyLoss_LabelSmooth',
]
