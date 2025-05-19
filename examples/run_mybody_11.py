import random
import numpy as np
import datetime

from ga.run_mybody11 import run_mybody

if __name__ == "__main__":
    seed = 20
    random.seed(seed)
    np.random.seed(seed)
    run_mybody(
        structure_shape = (5,5),
        experiment_name = "205.5",
        max_evaluations = 12,
        train_iters = 1000,
        num_cores = 12
    )

    print('run_universal over at ', datetime.datetime.now())