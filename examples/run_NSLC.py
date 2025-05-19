import random
import numpy as np
import datetime

#from ga.runga2 import run_ga
from ga.run_nslc import run_nslc

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    print('run_ga start at ', datetime.datetime.now())
    run_nslc(
        pop_size = 3,
        structure_shape = (5,5),
        experiment_name = "NSLC_test",
        max_evaluations = 12,
        train_iters = 50,
        num_cores = 3,
        archive_size = 6
    )

    print('run_ga over at ', datetime.datetime.now())
# python run_NSLC.py --env-name "Walker-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 32 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50 2>&1 | tee -a NSLC_test.log