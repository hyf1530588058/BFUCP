import random
import numpy as np
import datetime

from ga.run_universaladv import run_universal1

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    print('run_universal start at ', datetime.datetime.now())
    run_universal1(
        structure_shape = (5,5),
        experiment_name = "adv_03",
        max_evaluations = 20,
        train_iters = 300,
        num_cores = 5,
        iters = 10
        #worst_k = 5
    )

    print('run_universal over at ', datetime.datetime.now())