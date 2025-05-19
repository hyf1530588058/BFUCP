import random
import numpy as np
import datetime

from ga.run_universal2 import run_universal2

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    print('run_universal start at ', datetime.datetime.now())    
    run_universal2(
        structure_shape = (5,5),  
        experiment_name = "test_HM",
        max_evaluations = 3,
        train_iters = 50,
        num_cores = 3
    )

    print('run_universal over at ', datetime.datetime.now())

# python run_universal_2.py --env-name "Walker-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 32 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50 2>&1 | tee -a test_HM.log