import random
import numpy as np
import datetime

# from ga.run_nslc import run_nslc
# from ga.run_ga import run_ga
from ga.Distillse_run import run_se

if __name__ == "__main__":
    seed = 3407
    random.seed(seed)
    np.random.seed(seed)
    print('run_universal start at ', datetime.datetime.now())
    run_se(
        pop_size = 20,
        structure_shape = (5,5),
        experiment_name = "Distill_SE_(750)_walker_01",
        load_name = "NSLC_Pretrain_walker_001_10_100",
        max_evaluations = 750,
        train_iters = 1000,
        num_cores = 20,
    )

    print('run_ga over at ', datetime.datetime.now())
#python run_universal_1.py --env-name "Walker-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50 2>&1 | tee -a DistillGA_750_upstepper_3.log