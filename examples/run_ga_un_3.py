import random
import numpy as np
import datetime

from ga.run_universal2 import run_universal2

if __name__ == "__main__":
    seed = 100 
    random.seed(seed)
    np.random.seed(seed)
    print('run_universal start at ', datetime.datetime.now())
    run_universal2(
        structure_shape = (5,5),
        experiment_name = "zero_Adversary_HM_walker_10_3",
        max_evaluations = 30,
        train_iters = 500,
        num_cores = 15
    )

    print('run_universal over at ', datetime.datetime.now()) 
# python run_ga_un_3.py --env-name "Walker-v0" --algo ppo --use-gae --lr 0.0001 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 16 --num-steps 128 --num-mini-batch 12 --log-interva  100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50 2>&1 | tee -a robust_01.log
# python run_ga_un_3.py --env-name "Walker-v0" --algo ppo --use-gae --lr 0.0001 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 2 --num-steps 64 --num-mini-batch 12 --log-interva  100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50 2>&1 | tee -a new_test_robust_advantage_01.log
# python run_ga_un_3.py --env-name "Pusher-v0" --algo ppo --use-gae --lr 0.0001 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 16 --num-steps 128 --num-mini-batch 12 --log-interva  100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50 2>&1 | tee -a robust_advantage_pusher_01.log
# python run_ga_un_3.py --env-name "Walker-v0" --algo ppo --use-gae --lr 0.0001 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 2 --num-steps 64 --num-mini-batch 12 --log-interva  100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50 2>&1 | tee -a zero_random_HM_walker_2.log