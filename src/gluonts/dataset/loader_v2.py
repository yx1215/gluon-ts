# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Iterable, List, Callable, Optional
import itertools
import random

import numpy as np

from gluonts.core.component import DType
from gluonts.dataset.common import DataBatch, DataEntry, Dataset


def stack(data, make_array_fn):
    if isinstance(data[0], np.ndarray):
        data = make_array_fn(data)
    elif isinstance(data[0], (list, tuple)):
        return list(stack(t, make_array_fn) for t in zip(*data))
    return data


def batchify(
    data: List[dict], make_array_fn: Callable = np.asarray
) -> DataBatch:
    return {
        key: stack(
            data=[item[key] for item in data], make_array_fn=make_array_fn
        )
        for key in data[0].keys()
    }


class CyclicIterable(Iterable):
    def __init__(self, base_iterable: Iterable) -> None:
        self.base_iterable = base_iterable
        self.iterator = iter(base_iterable)

    def __iter__(self):
        while True:
            try:
                yield next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.base_iterable)


class PseudoShuffledIterable(Iterable):
    def __init__(self, base_iterable: Iterable, buffer_length: int):
        self.base_iterable = base_iterable
        self.buffer_length = buffer_length
        self.shuffle_buffer: list = []

    def sample_from_buffer(self):
        idx = random.randint(0, len(self.shuffle_buffer) - 1)
        return self.shuffle_buffer.pop(idx)

    def __iter__(self):
        for x in self.base_iterable:
            if len(self.shuffle_buffer) < self.buffer_length:
                self.shuffle_buffer.append(x)
            if len(self.shuffle_buffer) == self.buffer_length:
                yield self.sample_from_buffer()
        while len(self.shuffle_buffer):
            yield self.sample_from_buffer()


class DataLoader(Iterable[DataBatch]):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        batchify_fn: Callable = batchify,
        make_array_fn: Callable = np.asarray,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.batchify_fn = batchify_fn
        self.make_array_fn = make_array_fn

    def __iter__(self):
        iterator = iter(self.dataset)

        while True:
            batch_elements = list(itertools.islice(iterator, self.batch_size))
            if not batch_elements:
                break
            yield self.batchify_fn(
                data=batch_elements, make_array_fn=self.make_array_fn
            )

        return
