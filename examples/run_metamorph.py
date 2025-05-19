import random
import numpy as np
import datetime
from ga.run_meta_adv import run_meta

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    print('run_universal start at ', datetime.datetime.now())
    run_meta(
        structure_shape = (5,5),
        experiment_name = "traditional_action_adversary_universal_carrier_10",
        max_evaluations = 10,
        train_iters = 1000,
        num_cores = 10
    )
    print('run_universal over at ', datetime.datetime.now())