import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(maxlen=128, num_words=100)
print(x_train.shape, y_train.shape)
print(x_train[0],y_train[0])
print(len(x_train[0]),y_train[0])
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=128)
print(tf.equal(x_train, 0))
print(x_test.shape, y_test.shape)



import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple
from collections import defaultdict
import os
from pathlib import Path
import sys
from joblib import Parallel, delayed
from functools import partial
import multiprocessing
import warnings


def get_pick_finish_time(hebao_finish_time: datetime):
    '''
    从合包完成时间得到拣货时间
    Args:
        hebao_finish_time:合包完成时间

    Returns:
        pick_finish_time：拣货完成时间
    '''
    if hebao_finish_time.hour > 7 and hebao_finish_time.hour < 19:
        pick_finish_time = hebao_finish_time + timedelta(hours=1)  # 合包到拣货时间差
    elif hebao_finish_time.hour < 7:
        pick_finish_time = hebao_finish_time.replace(hour=9, minute=0)
    elif hebao_finish_time.hour > 19:
        pick_finish_time = hebao_finish_time.replace(hour=9, minute=0)
        pick_finish_time += timedelta(days=1)
    return pick_finish_time


def get_exit_warehouse_time(pick_finish_time):
    '''
    拣货到出库时间
    大部分拣货到出库时间花费1-2小时，少数花费13-14小时，经分析这些包裹99%+拣货完成在17:00之后
    Args:
        pick_finish_time:拣货完成时间

    Returns:出库时间

    '''
    if pick_finish_time.hour < 18:
        exit_warehouse_time = pick_finish_time + timedelta(hours=1) # 拣货到出库时间差
    else:
        exit_warehouse_time = pick_finish_time + timedelta(hours=14)  # 拣货到出库时间差
    return exit_warehouse_time