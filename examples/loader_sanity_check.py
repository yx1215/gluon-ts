import time

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
            transformed_dataset, num_workers=2, num_entries=100
        ),
        buffer_length=20,
    ),
    batch_size=batch_size,
    make_array_fn=lambda a: torch.tensor(a, device=torch.device("cpu")),
)

time.sleep(1.0)

del training_loader
