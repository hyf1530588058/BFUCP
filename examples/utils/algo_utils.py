import math
from os import name
from evogym import is_connected, has_actuator, get_full_connectivity, draw, get_uniform
import numpy as np
class Structure():

    def __init__(self, body, connections, label):
        self.body = body
        self.connections = connections

        self.reward = 0
        self.fitness = self.compute_fitness()
        
        self.is_survivor = False
        self.prev_gen_label = 0

        self.label = label
        self.init_pt_path = None

    def compute_fitness(self):

        self.fitness = self.reward
        return self.fitness

    def set_reward(self, reward):

        self.reward = reward
        self.compute_fitness()

    def __str__(self):
        return f'\n\nStructure:\n{self.body}\nF: {self.fitness}\tR: {self.reward}\tID: {self.label}'

    def __repr__(self):
        return self.__str__()

class TerminationCondition():

    def __init__(self, max_iters):
        self.max_iters = max_iters

    def __call__(self, iters):
        return iters >= self.max_iters

    def change_target(self, max_iters):
        self.max_iters = max_iters

def mutate(child, mutation_rate=0.1, num_attempts=10):
    
    pd = get_uniform(5)  
    pd[0] = 0.6 #it is 3X more likely for a cell to become empty

    # iterate until valid robot found
    for n in range(num_attempts):
        # for every cell there is mutation_rate% chance of mutation
        for i in range(child.shape[0]):
            for j in range(child.shape[1]):
                mutation = [mutation_rate, 1-mutation_rate]
                if draw(mutation) == 0: # mutation
                    child[i][j] = draw(pd)
        
        if is_connected(child) and has_actuator(child):
            return (child, get_full_connectivity(child))

    # no valid robot found after num_attempts
    return None
def mutate_new(child, mutation_rate=0.1, edge_mutation_rate=0.2, num_attempts=10):
    # 创建一个新的概率分布，只包含从0到2的整数值
    pd = get_uniform(3)  # 只有3个选项，对应0, 1, 2
    pd[0] = 0.6
    # 尝试直到找到有效的机器人
    for n in range(num_attempts):
        # 每个细胞都有一定的突变几率
        for i in range(child.shape[0]):
            for j in range(child.shape[1]):
                # 检查当前位置是否在最外围一圈
                if i == 0 or i == child.shape[0]-1 or j == 0 or j == child.shape[1]-1:
                    # 如果在最外围一圈，使用更高的突变率
                    current_mutation_rate = edge_mutation_rate
                else:
                    current_mutation_rate = mutation_rate
                
                mutation = [current_mutation_rate, 1-current_mutation_rate]
                # 使用draw函数来判断是否发生变异
                if draw(mutation) == 0: # mutation
                    # 如果元素值为3或4，则可以突变为0到2
                    if child[i][j] in [3, 4]:
                        child[i][j] = draw(pd)
                    # 如果元素值为1或2，有概率变为0，也可能保持不变
                    elif child[i][j] in [1, 2]:
                        # 这里我们通过增加一个判断来决定是否真的进行突变
                        if draw([0.5, 0.5]) == 0:  # 假设有50%的概率进行突变
                            child[i][j] = 0
                    # 如果元素值为0，则保持不变

        if is_connected(child) and has_actuator(child):
            return (child, get_full_connectivity(child))

def mutate_new_nozero(child, mutation_rate=0.1, edge_mutation_rate=0.15, num_attempts=10):
    # 创建一个新的概率分布，只包含从0到2的整数值
    pd = get_uniform(3)  # 只有3个选项，对应0, 1, 2
    pd[0] = 0.4
    count = 0
    # 尝试直到找到有效的机器人
    for n in range(num_attempts):
        # 每个细胞都有一定的突变几率
        for i in range(child.shape[0]):
            if count >= 2: # 5
                break
            for j in range(child.shape[1]):
                # 检查当前位置是否在最外围一圈
                if i == 0 or i == child.shape[0]-1 or j == 0 or j == child.shape[1]-1:
                    # 如果在最外围一圈，使用更高的突变率
                    current_mutation_rate = edge_mutation_rate
                else:
                    current_mutation_rate = mutation_rate
                
                mutation = [current_mutation_rate, 1-current_mutation_rate]
                if draw(mutation) == 0: # mutation
                    # 如果元素值为3或4，则可以突变为0到2
                    if child[i][j] in [3, 4]:
                        child[i][j] = draw(pd)
                        count += 1
                    # 如果元素值为1或2，有概率变为0，也可能保持不变
                    elif child[i][j] in [1, 2]:
                        if draw([0.3, 0.7]) == 0:  
                            child[i][j] = 0
                            count += 1
                    # 如果元素值为0，则保持不变
                if count >= 2:
                    break
        if is_connected(child) and has_actuator(child):
            return (child, get_full_connectivity(child))

def get_percent_survival(gen, max_gen):
    low = 0.0
    high = 0.8
    return ((max_gen-gen-1)/(max_gen-1))**1.5 * (high-low) + low

def total_robots_explored(pop_size, max_gen):
    total = pop_size
    for i in range(1, max_gen):
        total += pop_size - max(2, math.ceil(pop_size * get_percent_survival(i-1, max_gen))) 
    return total

def total_robots_explored_breakpoints(pop_size, max_gen, max_evaluations):
    
    total = pop_size
    out = []
    out.append(total)

    for i in range(1, max_gen):
        total += pop_size - max(2, math.ceil(pop_size * get_percent_survival(i-1, max_gen))) 
        if total > max_evaluations:
            total = max_evaluations
        out.append(total)

    return out

def search_max_gen_target(pop_size, evaluations):
    target = 0
    while total_robots_explored(pop_size, target) < evaluations:
        target += 1
    return target
    


def parse_range(str_inp, rbt_max):
    
    inp_with_spaces = ""
    out = []
    
    for token in str_inp:
        if token == "-":
            inp_with_spaces += " " + token + " "
        else:
            inp_with_spaces += token
    
    tokens = inp_with_spaces.split()

    count = 0
    while count < len(tokens):
        if (count+1) < len(tokens) and tokens[count].isnumeric() and tokens[count+1] == "-":
            curr = tokens[count]
            last = rbt_max
            if (count+2) < len(tokens) and tokens[count+2].isnumeric():
                last = tokens[count+2]
            for i in range (int(curr), int(last)+1):
                out.append(i)
            count += 3
        else:
            if tokens[count].isnumeric():
                out.append(int(tokens[count]))
            count += 1
    return out

def pretty_print(list_org, max_name_length=30):

    list_formatted = []
    for i in range(len(list_org)//4 +1):
        list_formatted.append([])

    for i in range(len(list_org)):
        row = i%(len(list_org)//4 +1)
        list_formatted[row].append(list_org[i])

    print()
    for row in list_formatted:
        out = ""
        for el in row:
            out += str(el) + " "*(max_name_length - len(str(el)))
        print(out)

def get_percent_survival_evals(curr_eval, max_evals):
    low = 0.0
    high = 0.6
    return ((max_evals-curr_eval-1)/(max_evals-1)) * (high-low) + low

def total_robots_explored_breakpoints_evals(pop_size, max_evals):
    
    num_evals = pop_size
    out = []
    out.append(num_evals)
    while num_evals < max_evals:
        num_survivors = max(2,  math.ceil(pop_size*get_percent_survival_evals(num_evals, max_evals)))
        new_robots = pop_size - num_survivors
        num_evals += new_robots
        if num_evals > max_evals:
            num_evals = max_evals
        out.append(num_evals)

    return out

if __name__ == "__main__":

    pop_size = 25
    num_evals = pop_size
    max_evals = 750

    count = 1
    print(num_evals, num_evals, count)
    while num_evals < max_evals:
        num_survivors = max(2,  math.ceil(pop_size*get_percent_survival_evals(num_evals, max_evals)))
        new_robots = pop_size - num_survivors
        num_evals += new_robots
        count += 1
        print(new_robots, num_evals, count)

    print(total_robots_explored_breakpoints_evals(pop_size, max_evals))
        
    # target = search_max_gen_target(25, 500)
    # print(target)
    # print(total_robots_explored(25, target-1))
    # print(total_robots_explored(25, target))

    # print(total_robots_explored_breakpoints(25, target, 500))