import time
import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mxnet as mx
import torch

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.transform import TransformedDataset
from gluonts.dataset.loader_v2 import (
    DataLoader,
    CyclicIterable,
    PseudoShuffledIterable,
    MultiProcessIterator,
)
from gluonts.model.deepar import DeepAREstimator

dataset = get_dataset("electricity")
dataset_train = dataset.train
freq = dataset.metadata.freq
prediction_length = dataset.metadata.prediction_length

batch_size = 32
num_batches_per_epoch = 8
num_epochs = 5

estimator = DeepAREstimator(freq=freq, prediction_length=prediction_length,)

transform = estimator.create_transformation()

## What happens during training?

transformed_dataset = TransformedDataset(
    base_dataset=CyclicIterable(dataset_train),
    transformation=transform,
    is_train=True,
)

training_loader = DataLoader(
    PseudoShuffledIterable(
        # The following line gives single process data loading
        # base_iterable=iter(transformed_dataset),
        # The following lines give multi process data loading
        base_iterable=MultiProcessIterator(
            transformed_dataset,
            num_workers=2,
            num_entries=num_epochs * num_batches_per_epoch * batch_size,
        ),
        buffer_length=20,
    ),
    batch_size=batch_size,
    make_array_fn=lambda a: torch.tensor(a, device=torch.device("cpu")),
)

exp_total_batches = num_batches_per_epoch * num_epochs

start = time.time()
batch_ids = []
for epoch_no in range(num_epochs):
    for batch in itertools.islice(training_loader, num_batches_per_epoch):
        assert isinstance(batch["past_target"], torch.Tensor)
        assert (
            batch["past_target"].shape[0] == batch_size
        ), f"{batch['past_target'].shape[0]} vs {batch_size}"
        batch_ids.append(batch["item_id"])
end = time.time()

assert (
    len(batch_ids) == exp_total_batches
), f"{len(batch_ids)} vs {exp_total_batches}"

count_per_batch = np.zeros(shape=(len(dataset_train), len(batch_ids)))
for k, ids in enumerate(batch_ids):
    for id in ids:
        count_per_batch[id, k] += 1

print(f"batches loaded: {len(batch_ids)}")
print(f"elapsed time: {end - start}")
print(f"average time: {(end - start) / len(batch_ids)}")

plt.figure(figsize=(20, 10))
sns.heatmap(count_per_batch)
plt.title("Training batches")
plt.show()

# The following is needed to trigger subprocesses termination
del training_loader

## What happens during inference?

transformed_dataset = TransformedDataset(
    base_dataset=dataset_train, transformation=transform, is_train=False,
)

inference_loader = DataLoader(
    transformed_dataset,
    batch_size=batch_size,
    make_array_fn=lambda a: mx.nd.array(a, ctx=mx.cpu(1)),
)

start = time.time()
batch_ids = []
for batch in inference_loader:
    assert isinstance(batch["past_target"], mx.nd.NDArray)
    assert batch["past_target"].shape[0] <= batch_size
    batch_ids.append(batch["item_id"])
end = time.time()

count_per_batch = np.zeros(shape=(len(dataset_train), len(batch_ids)))
for k, ids in enumerate(batch_ids):
    for id in ids:
        count_per_batch[id, k] += 1

print(f"batches loaded: {len(batch_ids)}")
print(f"elapsed time: {end - start}")
print(f"average time: {(end - start) / len(batch_ids)}")

plt.figure(figsize=(20, 10))
sns.heatmap(count_per_batch)
plt.title("Inference batches")
plt.show()
