from distutils.command.config import config
import os
from re import X
import shutil
import random
import numpy as np

from GPyOpt.core.task.space import Design_space
from GPyOpt.models import GPModel
from GPyOpt.optimization import AcquisitionOptimizer
from GPyOpt.experiment_design import initial_design
from GPyOpt.acquisitions import AcquisitionEI
from GPyOpt.core.evaluators import ThompsonBatch
from .optimizer import Objective, Optimization

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

import evogym.envs
from evogym import is_connected, has_actuator, get_full_connectivity
from utils.algo_utils import TerminationCondition
from ppo.NSLCrun import run_nslcppo

def get_robot_from_genome(genome, config):
    '''
    genome is a 1d vector
    robot is a 2d matrix
    '''
    structure_shape = config['structure_shape']
    robot = genome.reshape(structure_shape)
    return robot

def eval_genome_cost(genome, config, genome_id, generation):
    robot = get_robot_from_genome(genome, config)
    if not (is_connected(robot) and has_actuator(robot)):
        return 10
    else:
        connectivity = get_full_connectivity(robot)
        save_path_generation = os.path.join(config['save_path'], f'generation_{generation}')
        save_path_structure = os.path.join(save_path_generation, 'structure', f'{genome_id}')
        save_path_controller = os.path.join(save_path_generation, 'controller')
        np.savez(save_path_structure, robot, connectivity)
        fitness = run_nslcppo(
            structure=(robot, connectivity),
            termination_condition=TerminationCondition(config['train_iters']),
            saving_convention=(save_path_controller, genome_id),
        )
        cost = -fitness
        return cost

def eval_genome_constraint(genomes, config):
    all_violation = []
    for genome in genomes:
        robot = get_robot_from_genome(genome, config)
        violation = not (is_connected(robot) and has_actuator(robot))
        all_violation.append(violation)
    return np.array(all_violation)

def run_bo(
        experiment_name,
        structure_shape,
        pop_size,
        max_evaluations,
        train_iters,
        num_cores,
    ):

    save_path = os.path.join(root_dir, 'saved_data', experiment_name)
    unique_structures = []  # 用于存储所有独特的形态
    ALL_structure_hashes = set()  # 用于存储所有形态的哈希值

    try:
        os.makedirs(save_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print('Override? (y/n): ', end='')
        ans = input()
        if ans.lower() == 'y':
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            return None, None
        print()

    save_path_metadata = os.path.join(save_path, 'metadata.txt')
    with open(save_path_metadata, 'w') as f:
        f.write(f'POP_SIZE: {pop_size}\n' \
            f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n' \
            f'MAX_EVALUATIONS: {max_evaluations}\n' \
            f'TRAIN_ITERS: {train_iters}\n')
    # 新增：创建最终结果文件夹
    final_result_path = os.path.join(save_path, 'final_results')
    os.makedirs(final_result_path, exist_ok=True)

    config = {
        'structure_shape': structure_shape,
        'train_iters': train_iters,
        'save_path': save_path,
    }
    
    def constraint_func(genome): 
        return eval_genome_constraint(genome, config)

    def before_evaluate(generation):
        save_path = config['save_path']
        save_path_structure = os.path.join(save_path, f'generation_{generation}', 'structure')
        save_path_controller = os.path.join(save_path, f'generation_{generation}', 'controller')
        os.makedirs(save_path_structure, exist_ok=True)
        os.makedirs(save_path_controller, exist_ok=True)

    def after_evaluate(generation, population_cost):
        save_path = config['save_path']
        save_path_ranking = os.path.join(save_path, f'generation_{generation}', 'output.txt')
        
        # 新增调试信息
        print(f"Processing generation {generation}")
        print(f"Population size: {len(population_cost)}")
        
        # 确保获取当前代的正确数据
        current_population = bo.X[-len(population_cost):]  # 获取最新一代的种群
        
        # 新增调试信息
        print(f"Current population shape: {current_population.shape}")
        
        # 处理每个个体
        for i, (genome, cost) in enumerate(zip(current_population, population_cost)):
            try:
                robot = get_robot_from_genome(genome, config)
                structure_hash = hash(robot.tobytes())
                
                if structure_hash not in ALL_structure_hashes:
                    ALL_structure_hashes.add(structure_hash)
                    unique_structures.append((generation, i, -cost, robot))
                    
                    # 立即保存当前代的独特形态
                    structure_name = f"{generation}_{i}"
                    np.save(os.path.join(final_result_path, f"{structure_name}.npy"), robot)
                    
                    # 新增调试信息
                    print(f"Saved new structure: {structure_name}")
                    
            except Exception as e:
                print(f"Error processing individual {i} in generation {generation}: {str(e)}")
                continue

        # 保存排名信息
        genome_fitness_list = -population_cost
        genome_id_list = np.argsort(population_cost)
        genome_fitness_list = np.array(genome_fitness_list)[genome_id_list]
        
        with open(save_path_ranking, 'w') as f:
            out = ''
            for genome_id, genome_fitness in zip(genome_id_list, genome_fitness_list):
                out += f'{genome_id}\t\t{genome_fitness}\n'
            f.write(out)

    space = Design_space(
        space=[{'name': 'x', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4), 'dimensionality': np.prod(structure_shape)}], 
        constraints=[{'name': 'const', 'constraint': constraint_func}]
    )

    objective = Objective(eval_genome_cost, config, num_cores=num_cores)

    model = GPModel()

    acquisition = AcquisitionEI(
        model, 
        space, 
        optimizer=AcquisitionOptimizer(space)
    )

    evaluator = ThompsonBatch(acquisition, batch_size=pop_size)
    X_init = initial_design('random', space, pop_size)

    bo = Optimization(model, space, objective, acquisition, evaluator, X_init, de_duplication=True)
        # 新增：确保bo对象有batch_size属性
    bo.batch_size = pop_size
    
    bo.run_optimization(
        max_iter=np.ceil(max_evaluations / pop_size) - 1,
        verbosity=True,
        before_evaluate=before_evaluate,
        after_evaluate=after_evaluate
    )
    best_robot, best_fitness = bo.x_opt, -bo.fx_opt
    # 新增：保存最终结果
    if unique_structures:
        # 保存所有独特形态
        for gen, idx, fitness, structure in unique_structures:
            structure_name = f"{gen}_{idx}"
            np.save(os.path.join(final_result_path, f"{structure_name}.npy"), structure)

        # 计算并保存统计信息
        all_fitness = [f for _, _, f, _ in unique_structures]
        best_fitness = max(all_fitness)
        avg_fitness = sum(all_fitness) / len(all_fitness)

        # 按生成数量分段统计
        segment_sizes = [25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600,625,650,675,700,725,750]
        segment_stats = []
        sorted_by_generation = sorted(unique_structures, key=lambda x: (x[0], x[1]))

        for size in segment_sizes:
            if len(sorted_by_generation) >= size:
                segment = sorted_by_generation[:size]
                segment_fitness = [s[2] for s in segment]
                segment_best = max(segment_fitness)
                segment_avg = sum(segment_fitness) / len(segment_fitness)
                segment_stats.append((size, segment_best, segment_avg))

        # 保存统计信息
        with open(os.path.join(final_result_path, "summary.txt"), 'w') as f:
            f.write(f"Best Fitness: {best_fitness}\n")
            f.write(f"Average Fitness: {avg_fitness}\n")
            f.write(f"Total Unique Structures: {len(unique_structures)}\n")
            
            f.write("\nSegment Statistics:\n")
            for size, seg_best, seg_avg in segment_stats:
                f.write(f"Top {size}:\n")
                f.write(f"  Best Fitness: {seg_best}\n")
                f.write(f"  Average Fitness: {seg_avg}\n")

        # 保存生成顺序
        with open(os.path.join(final_result_path, "generation_sequence.txt"), 'w') as f:
            f.write("Generation\tLabel\tFitness\n")
            for gen, label, fitness, _ in sorted_by_generation:
                f.write(f"{gen}\t{label}\t{fitness}\n")

    return best_robot, best_fitness
