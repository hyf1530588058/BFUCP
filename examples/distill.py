import os
import numpy as np
import torch
import pickle
from torch.optim import Adam
from torch.nn import MSELoss
from ppo.metamorphmodel import ImitationNet
# from ppo.myPPOmodel2 import Policy
import random
import datetime
import matplotlib.pyplot as plt

curr_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(curr_dir, "AAA_datasets")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_dataset(dataset_path):
    """
    加载数据集
    """
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def train_imitation_learning(dataset, model, batch_size=128, num_steps=100000, val_every=100):
    """
    改进后的模仿学习训练函数
    """
    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = MSELoss()
    
    losses = []
    val_losses = []
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    # 确保需要训练的参数启用梯度
    for param in model.parameters():
        param.requires_grad = True    

    for step in range(num_steps):
        idx = np.random.randint(0, len(dataset['obs_train']), batch_size)
        obs = torch.from_numpy(dataset['obs_train'][idx]).float().to(device)
        act = torch.from_numpy(dataset['act_train'][idx]).float().to(device)
        mask = torch.from_numpy(dataset['mask_train'][idx]).float().to(device)
        # structure = torch.from_numpy(dataset['structure_train'][idx]).float().to(device)
        structure = dataset['structure_train'][idx]

        obs = obs.squeeze(1)
        act = act.squeeze(1)
                # 检查输入数据是否保留了梯度
        obs = obs.requires_grad_(True).to(device)
        
        pred_act = model(structure, obs)

        valid_ratio = mask.sum(dim=1) / mask.size(1)
        weights = valid_ratio.unsqueeze(1).expand_as(mask)
        

        effective_lr = optimizer.param_groups[0]['lr'] * valid_ratio.mean()
        for param_group in optimizer.param_groups:
            param_group['lr'] = effective_lr
            

        masked_pred = pred_act * mask
        masked_act = act * mask
        loss = criterion(masked_pred, masked_act)
        # print("损失值梯度状态:", loss.requires_grad)
 
        invalid_output = pred_act * (1 - mask)
        reg_loss = torch.mean(invalid_output ** 2)
        total_loss = loss + 0.1 * reg_loss
        
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        
        # 记录损失
        losses.append(loss.item())
        
        # 验证
        if (step + 1) % val_every == 0:
            val_loss = validate(model, dataset, batch_size)
            val_losses.append(val_loss)
            print(f"Step {step + 1}, Loss: {np.mean(losses[-val_every:])}, Val Loss: {val_loss}")
    
    return losses, val_losses

def validate(model, dataset, batch_size):
    """
    验证模型性能
    """
    criterion = MSELoss()
    
    idx = np.random.randint(0, len(dataset['obs_val']), batch_size)
    obs = torch.from_numpy(dataset['obs_val'][idx]).float().to(device)
    act = torch.from_numpy(dataset['act_val'][idx]).float().to(device)
    mask = torch.from_numpy(dataset['mask_val'][idx]).float().to(device)
    # structure = torch.from_numpy(dataset['structure_val'][idx]).float().to(device)
    structure = dataset['structure_train'][idx]
    # 消除 obs 和 act 的中间维度
    obs = obs.squeeze(1)
    act = act.squeeze(1)

    with torch.no_grad():
        pred_act = model(structure, obs)
        val_loss = criterion(pred_act * mask, act * mask)
    
    return val_loss.item()

def plot_action_distribution(dataset):
    # 展平所有动作值
    actions = dataset['act_train'].flatten()
    mask = dataset['mask_train'].flatten()
    # 创建绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(actions[mask==1], bins=100, range=(-15, 15), color='blue', alpha=0.7)
    
    # 添加标注
    plt.axvline(x=0.6, color='red', linestyle='--', label='Lower Bound (0.6)')
    plt.axvline(x=1.6, color='green', linestyle='--', label='Upper Bound (1.6)')
    
    # 添加标签
    plt.title('Action Value Distribution')
    plt.xlabel('Action Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 保存图像
    save_path = os.path.join(save_dir, 'action_distribution.png')
    plt.savefig(save_path)
    print(f"图像已保存到：{save_path}")
    
    # 关闭图形
    plt.close()

def main(dataset_name):
    # 加载数据集
    dataset = load_dataset(os.path.join(save_dir, f'{dataset_name}.pkl'))
    
    # 绘制动作值分布
    # plot_action_distribution(dataset)
    
    # 初始化模型
    obs_dim = dataset['obs_train'].shape[1]
    act_dim = dataset['act_train'].shape[1]
    model = ImitationNet().to(device)
    
    # 训练
    losses, val_losses = train_imitation_learning(dataset, model)
    path = os.path.join(save_dir, 'distilled_controllers',dataset_name,'imitation_model.pt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 保存模型
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    print('distill start at ', datetime.datetime.now())
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    dataset_name = "NSLC_test_001_10_100"
    main(dataset_name)
    print('distill over at ', datetime.datetime.now())

# python distill.py 2>&1 | tee -a distill_test.log