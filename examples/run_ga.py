import random
import numpy as np
import datetime

from ga.run_ga_universal import run_meta

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    run_meta(
        pop_size = 20,
        structure_shape = (5,5),
        experiment_name = "test_ga",
        max_evaluations = 5000,
        train_iters = 50,
        num_cores = 8,
    )

    print('run_ga over at ', datetime.datetime.now())

# python run_ga.py --env-name "Walker-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50