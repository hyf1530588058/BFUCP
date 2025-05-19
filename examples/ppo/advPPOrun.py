import os, sys
sys.path.insert(1, os.path.join(sys.path[0], 'externals', 'pytorch_a2c_ppo_acktr_gail'))

import numpy as np
import time
from collections import deque
import torch

from ppo import utils
from ppo.arguments import get_args
from ppo.evaluate import evaluate
from ppo.envs import make_vec_envs

from a2c_ppo_acktr import algo          
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.storage import RolloutStorage
import evogym.envs  
from ppo.RARLmodel_MLP import Policy
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir,'..')
origin_dir =  os.path.join(root_dir,'..')
def run_advmetappo_2(          #此文件功能与ppo原文件夹中的run文件功能一致#
    structure,        #机器人结构，包括体素矩阵和连接矩阵
    termination_condition,    #终止条件
    saving_convention,    #控制器保存路径
    pro_actor_critic,
    adv_actor_critic,
    args,
    override_env_name = None,
    verbose = True):

    assert (structure == None) == (termination_condition == None) and (structure == None) == (saving_convention == None)    #断言语句，若后续表达式为false则程序中断,此处用于保证所有参数均被输入#

    print(f'Starting training on \n{structure}\nat {saving_convention}...\n')
    if override_env_name:
        args.env_name = override_env_name

    torch.manual_seed(args.seed)   #设置CPU生成随机数种子#
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = args.log_dir
    if saving_convention != None:
        log_dir = os.path.join(saving_convention[0], log_dir, "robot_" + str(saving_convention[1]))
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:1" if args.cuda else "cpu")
    #device = torch.device('cpu')

    envs = make_vec_envs(args.env_name, structure, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    
    pro_actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    pro_actor_critic.to(device)
    # pro_actor_critic.load_state_dict(torch.load(os.path.join(root_dir,"saved_data","RARL_MLP_onlyact_walker","generation_0","controller","robot_"+str(saving_convention[1])+"_controller"+".pt"))[0].state_dict())
    adv_actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    adv_actor_critic.to(device)

    pro_agent = algo.PPO_base(
        pro_actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)
    
    adv_agent = algo.PPO_base(
        adv_actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)



    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts_agent = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              pro_actor_critic.recurrent_hidden_state_size)
    
    rollouts_adv = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              pro_actor_critic.recurrent_hidden_state_size)    

    obs = envs.reset()
    rollouts_agent.obs[0].copy_(obs)
    rollouts_agent.to(device)
    rollouts_adv.obs[0].copy_(obs)
    rollouts_adv.to(device)


    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes   

    rewards_tracker = []
    avg_rewards_tracker = []
    sliding_window_size = 10
    max_determ_avg_reward = float('-inf')

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                pro_agent.optimizer, j, num_updates,
                pro_agent.optimizer.lr if args.algo == "acktr" else args.lr)
            utils.update_linear_schedule(
                adv_agent.optimizer, j, num_updates,
                adv_agent.optimizer.lr if args.algo == "acktr" else args.lr)

        # 训练主策略
        pro_actor_critic.train()
        adv_actor_critic.eval()
        for step in range(args.num_steps):
            # Sample actions
            #print("rollouts_agent.obs[step]:",rollouts_agent.obs[step].shape)
            with torch.no_grad():
                #print("rollouts_agent.obs[step]:",rollouts_agent.obs[step].shape)
                pro_value, pro_action, pro_action_log_prob, pro_recurrent_hidden_states = pro_actor_critic.act(     #动作#
                    rollouts_agent.obs[step], rollouts_agent.recurrent_hidden_states[step],
                    rollouts_agent.masks[step])
                # if j%10 == 0:
                adv_value, adv_action, adv_action_log_prob, adv_recurrent_hidden_states = adv_actor_critic.act(     #动作#
                    rollouts_adv.obs[step], rollouts_adv.recurrent_hidden_states[step],
                    rollouts_adv.masks[step])

            adv_action = torch.clamp(adv_action, min=-0.3, max=0.3)
            action = pro_action + adv_action
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    rewards_tracker.append(info['episode']['r'])
                    if len(rewards_tracker) < 10:
                        avg_rewards_tracker.append(np.average(np.array(rewards_tracker)))
                    else:
                        avg_rewards_tracker.append(np.average(np.array(rewards_tracker[-10:])))

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                for info in infos])
            rollouts_agent.insert(obs, pro_recurrent_hidden_states, pro_action,
                            pro_action_log_prob, pro_value, reward, masks, bad_masks)
                
        with torch.no_grad():
            pro_next_value = pro_actor_critic.get_value(
                rollouts_agent.obs[-1], rollouts_agent.recurrent_hidden_states[-1],
                rollouts_agent.masks[-1]).detach()    

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts_agent,
                            utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts_agent.rewards[step] = discr.predict_reward(
                    rollouts_agent.obs[step], rollouts_agent.actions[step], args.gamma,
                    rollouts_agent.masks[step])

        rollouts_agent.compute_returns(pro_next_value, args.use_gae, args.gamma,
                                args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = pro_agent.update(rollouts_agent)
        rollouts_agent.after_update()

        adv_actor_critic.train()
        pro_actor_critic.eval()
        for step in range(args.num_steps):
            with torch.no_grad():
                _, pro_action, _, _ = pro_actor_critic.act(
                    rollouts_agent.obs[step], rollouts_agent.recurrent_hidden_states[step],
                    rollouts_agent.masks[step])
                
                adv_value, adv_action, adv_action_log_prob, adv_recurrent_hidden_states = adv_actor_critic.act(
                    rollouts_adv.obs[step], rollouts_adv.recurrent_hidden_states[step],
                    rollouts_adv.masks[step])
                
            adv_action = torch.clamp(adv_action, min=-0.3, max=0.3)
            action = pro_action + adv_action
            obs, reward, done, infos = envs.step(action)

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            
            rollouts_adv.insert(obs, adv_recurrent_hidden_states, adv_action,
                                adv_action_log_prob, adv_value, -reward, masks, bad_masks)

        with torch.no_grad():
            adv_next_value = adv_actor_critic.get_value(
                rollouts_adv.obs[-1], rollouts_adv.recurrent_hidden_states[-1],
                rollouts_adv.masks[-1]).detach()

        rollouts_adv.compute_returns(adv_next_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

        adv_value_loss, adv_action_loss, adv_dist_entropy = adv_agent.update(rollouts_adv)
        rollouts_adv.after_update()

        # print status
        if j % args.log_interval == 0 and len(episode_rewards) > 1 and verbose:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))
        
        # evaluate the controller and save it if it does the best so far
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            determ_avg_reward = evaluate(args.num_evals, pro_actor_critic, obs_rms, args.env_name, structure,args.seed,
                    args.num_processes, eval_log_dir, device)

            if verbose:
                if saving_convention != None:
                    print(f'Evaluated {saving_convention[1]} using {args.num_evals} episodes. Mean reward: {np.mean(determ_avg_reward)}\n')
                else:
                    print(f'Evaluated using {args.num_evals} episodes. Mean reward: {np.mean(determ_avg_reward)}\n')

            if determ_avg_reward > max_determ_avg_reward:
                max_determ_avg_reward = determ_avg_reward

                temp_path = os.path.join(args.save_dir, args.algo, args.env_name + ".pt")
                if saving_convention != None:
                    temp_path = os.path.join(saving_convention[0], "robot_" + str(saving_convention[1]) + "_controller" + ".pt")
                
                if verbose:
                    print(f'Saving {temp_path} with avg reward {max_determ_avg_reward}\n')
                torch.save([     #此处即为控制器保存张量#
                    pro_actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], temp_path)
        
        # return upon reaching the termination condition
        if not termination_condition == None:
            if termination_condition(j):
                # file_path = os.path.join(origin_dir,"metamorph-change_1",str(saving_convention[1]))  
                # os.makedirs(file_path, exist_ok=True)
                # f = os.path.join(file_path, "robot_" + str(saving_convention[1]) + "final_controller" + ".pt")
                # torch.save(actor_critic.state_dict(), f)                
                if verbose:
                    print(f'{saving_convention} has met termination condition ({j})...terminating...\n')
                return max_determ_avg_reward
