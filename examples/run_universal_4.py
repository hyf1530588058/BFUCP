import random
import numpy as np
import datetime

from ga.run_universal4 import run_universal4

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    print('run_universal start at ', datetime.datetime.now())
    run_universal4(
        structure_shape = (5,5),
        experiment_name = "MDP&Model_catcher",
        max_evaluations = 10,
        train_iters = 1000,
        num_cores = 10
    )

    print('run_universal over at ', datetime.datetime.now())