import time
import mxnet as mx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from gluonts.dataset.loader import TrainDataLoader
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.artificial import shuffle_testing_dataset
from gluonts.distribution.neg_binomial import NegativeBinomialOutput
BATCH_SIZE = 32
NUM_BATCH_PER_EPOCH = 50
n = 300
time_list = []
range_list = range(1, 30)
for c in range_list:
    train_ds = shuffle_testing_dataset(n)
    estimator = DeepAREstimator(
        prediction_length=20,
        freq="D",
    )
    transform = estimator.create_transformation()
    loader = TrainDataLoader(
        train_ds,
        transform=transform,
        batch_size=BATCH_SIZE,
        ctx=mx.cpu(),
        num_batches_per_epoch=NUM_BATCH_PER_EPOCH,
        shuffle_buffer_length=c * BATCH_SIZE,
        num_workers=2,
    )
    hit_list = [[0 for i in range(NUM_BATCH_PER_EPOCH)] for j in range(n)]
    print(f"dataset size: {len(train_ds)}")
    start_time = time.time()
    count = 0
    for batch in loader:
        print(f"item id: {batch['item_id']}")
        for item_id in batch['item_id'][0]:
            hit_list[item_id][count] += 1
        count += 1
    end_time = time.time()
    time_list.append(end_time - start_time)
    print(f"elapsed time: {end_time - start_time}")
    sns.set()
    ax = sns.heatmap(hit_list)
    plt.title("buffer length={} * batch_size".format(c))
    plt.show()
plt.plot(range_list, time_list)
plt.show()