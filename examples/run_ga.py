import random
import numpy as np
import datetime


from ga.run_ga import run_ga

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    print('run_ga start at ', datetime.datetime.now())
    run_ga(
        pop_size = 20,
        structure_shape = (5,5),
        experiment_name = "GA_baseline(750)_pusher_01",
        max_evaluations = 750,
        train_iters = 1000,
        num_cores = 20,
    )

    print('run_ga over at ', datetime.datetime.now())

# python run_ga.py --env-name "Pusher-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50 2>&1 | tee -a GA_baseline750_pusher_01.log
