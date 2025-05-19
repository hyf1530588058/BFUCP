import gym
import evogym.envs
from evogym import sample_robot
import numpy as np

def test_external_force():
    body = np.array([[1., 3., 4., 1.,],\
                        [2., 4., 3., 2., ],\
                        [2., 0., 0., 2., ],\
                        [2., 0., 0., 2.,]])
    env = gym.make('Walker-v0', body=body, render_mode='human')
    env.reset()
    pos = env.object_pos_at_time(0, 'robot')
    force_shape = pos.shape
    count = 1
    while count<1800:
        action = np.ones_like(env.action_space.sample())*1.1
        external_force = np.zeros((2, 23))
        assert np.all(external_force.shape == force_shape)
        force = 10000
        external_force[0,21:]=force*(count//100%2*2-1)
        external_force[0,17:19]=force*(count//100%2*2-1)
        external_force[0,13:15]=force*(count//100%2*2-1)

        external_force[0,19:21]=-force*(count//100%2*2-1)
        external_force[0,15:17]=-force*(count//100%2*2-1)
        external_force[0,11:13]=-force*(count//100%2*2-1)

        env.add_external_force(external_force)
        ob, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()
        count+=1
    env.close()

def test_friction():
    body = np.array([[1., 3., 4., 1.,],\
                        [2., 4., 3., 2., ],\
                        [2., 0., 0., 2., ],\
                        [2., 0., 0., 2.,]])
    env = gym.make('Walker-v0', body=body, render_mode='human')
    env.reset()
    pos = env.object_pos_at_time(0, 'robot')
    force_shape = pos.shape
    count = 1
    env.set_friction_const(100) # set low friction
    while count<1800:
        action = np.ones_like(env.action_space.sample())*1.1
        external_force = np.zeros((2, 23))
        assert np.all(external_force.shape == force_shape)
        force = 1000
        external_force[0,21:]=force
        if count==600:
            env.set_friction_const(1200) # set high friction (1200 is the default set for evogym)
        env.add_external_force(external_force)
        ob, reward, terminated, truncated, info = env.step(action)
        count+=1
        if count==1800:
            break
    env.close()

def test_mass():
    body = np.array([[1., 3., 4., 1.,],\
                        [2., 4., 3., 2., ],\
                        [4., 0., 0., 4., ],\
                        [4., 0., 0., 4.,]])
    env = gym.make('Walker-v0', body=body, render_mode='human')
    env.reset()
    pos = env.object_pos_at_time(0, 'robot')
    force_shape = pos.shape
    count = 1
    while count<1800:
        action = env.action_space.sample()
        external_force = np.zeros((2, 23))
        if count==500:
            mass = np.ones((2, 23))
            mass[0,21:]=20.
            mass[0,17:19]=20.
            mass[0,13:15]=20.
            mass[0,4:6]=10.
            mass[0,9:11]=10.
            env.set_robot_mass(mass)
        assert np.all(external_force.shape == force_shape)
        ob, reward, terminated, truncated, info = env.step(action)
        count+=1
        if count==1800:
            break
    env.close()

if __name__ == '__main__':
    while True:
        print("\nPlease choose an option:")
        print("1. Test external force")
        print("2. Test setting friction")
        print("3. Test setting mass")
        # print("3. Test setting masses")
        print("q. Quit")

        choice = input("Enter your choice (1/2/3/q): ")

        if choice == '1':
            test_external_force()
        elif choice == '2':
            test_friction()
        elif choice == '3':
            test_mass()
        elif choice.lower() == 'q':
            print("Exiting program")
            break
        else:
            print("Invalid option, please choose again.")