import random
import numpy as np
import datetime
from ga.run_se import run_se
import time
from ppo.arguments import get_args

import torch

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    print('run_universal start at ', datetime.datetime.now())
    
    run_se(
        pop_size = 20,
        structure_shape = (5,5),
        experiment_name = "SE_baseline(750)_pusher_01",
        max_evaluations = 750,
        train_iters = 1000,
        num_cores = 10,
    )
    print('run_universal over at ', datetime.datetime.now())
# python run_se.py --env-name "Walker-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50 2>&1 | tee -a Abalation_all_walker_03.log