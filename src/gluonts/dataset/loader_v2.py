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

from typing import Iterable, Iterator, List, Callable, Optional
import itertools
import random
from multiprocessing import Process, Queue, Event
from queue import Empty

import numpy as np

from gluonts.dataset.common import DataBatch, Dataset
from gluonts.dataset.util import MPWorkerInfo


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

    def __iter__(self):
        while True:
            yield from self.base_iterable


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


class MultiProcessIterator(Iterator):
    def __init__(
        self,
        base_iterable: Iterable,
        num_workers: int,
        num_entries: int,
        max_queue_size: Optional[int] = None,
    ):
        assert num_workers >= 1
        assert num_entries >= 0
        assert max_queue_size is None or max_queue_size >= num_workers

        self.base_iterable = base_iterable
        self.num_workers = num_workers
        self.num_entries = num_entries
        self.max_queue_size = (
            max_queue_size if max_queue_size is not None else 5 * num_workers
        )

        self.data_queue: Queue = Queue(maxsize=self.max_queue_size)
        self.done_event = Event()
        self.processes = []

        for wid in range(self.num_workers):
            p = Process(
                target=self.worker_fn,
                args=(
                    wid,
                    self.num_workers,
                    self.base_iterable,
                    self.data_queue,
                    self.done_event,
                ),
            )
            p.start()
            self.processes.append(p)

        self.count = 0

    @staticmethod
    def worker_fn(
        worker_id: int,
        num_workers: int,
        iterable: Iterable,
        data_queue: Queue,
        end_event,
    ):
        MPWorkerInfo.worker_process = True
        MPWorkerInfo.worker_id = worker_id
        MPWorkerInfo.num_workers = num_workers

        for entry in iterable:
            if end_event.is_set():
                break
            data_queue.put((worker_id, entry))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            wid, entry = self.data_queue.get(timeout=1.0)
        except Empty:
            raise StopIteration()

        self.count += 1

        if self.count == self.num_entries:
            self.done_event.set()
        elif self.count > self.num_entries:
            raise StopIteration()

        return entry

    def _empty_queue(self):
        try:
            item = self.data_queue.get(block=False)
            while item:
                self.data_queue.get(block=False)
        except Empty:
            pass

    def __del__(self):
        # Send termination message to workers
        self.done_event.set()
        # Empty queue to make sure workers get the message
        self._empty_queue()
        for p in self.processes:
            p.join()
        self.data_queue.cancel_join_thread()
        self.data_queue.close()


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
