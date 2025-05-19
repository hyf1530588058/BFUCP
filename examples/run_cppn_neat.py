import random
import numpy as np
import datetime
from cppn_neat.run_cppn import run_cppn_neat

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    print('run_universal start at ', datetime.datetime.now())
    best_robot, best_fitness = run_cppn_neat(
        experiment_name='CPPN_baseline(750)_pusher_01',
        structure_shape=(5, 5),
        pop_size=20,
        max_evaluations=750,
        train_iters=1000,
        num_cores=20,
    )
    print('run_universal start at ', datetime.datetime.now())
    
    print('Best robot:')
    print(best_robot)
    print('Best fitness:', best_fitness)

# python run_cppn_neat.py --env-name "Pusher-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50 2>&1 | tee -a CPPN_baseline750_pusher_01.log