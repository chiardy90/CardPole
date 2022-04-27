import gym
import numpy as np

env = gym.make('CartPole-v0')

num_episodes = 5000 #要執行幾回合
max_number_of_steps = 200 #每回合最多推幾次
num_consecutive_iterations = 100 #取最後100回合來計算平均分數
last_time_steps = np.zeros(num_consecutive_iterations) #用來記錄最後100回合的分數
goal_average_steps = 195 #若平均分數達到195以上代表成功


q_table = np.random.uniform(low=-1, high=1, size=(4 ** 4, env.action_space.n))
#產生一個256*2的陣列，陣列內產生隨意數值


def bins(clip_min, clip_max, num):  #將數字clip_min ~ clip_max，區分為num個區間
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(observation): #將四個環境值轉換為狀態0~255 共256種
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),
                 np.digitize(cart_v, bins=bins(-3.0, 3.0, 4)),
                 np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),
                 np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))]
    #將四個環境值轉換為4個元素的數值list，每個元素值的範圍是0~3
    
    return sum([x * (4 ** i) for i, x in enumerate(digitized)])
    #將上面的四進位制轉為十進位制，例如：[2,3,1,0]，會轉換為180

def get_action(state, action, observation, reward):
    #依照目前的進度更新Q Table，並取得下次的動作與狀態需要傳入的值，分別為本次狀態、本次動作、動作後環境值、動作後獎勵
    next_state = digitize_state(observation) #用動作後的環境值計算出下次狀態
    next_action = np.argmax(q_table[next_state]) #從下次狀態找出價值最高的動作作為下次動作

    alpha = 0.2  #設定新值要採用的比例
    gamma = 0.99 #設定未來分數會乘上的折扣因子
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * q_table[next_state, next_action]) #更新Q Table

    return next_action, next_state

step_list = []
for episode in range(num_episodes):
    observation = env.reset() #環境初始化，並取得初始值

    state = digitize_state(observation) #用環境值取得目前狀態
    action = np.argmax(q_table[state])  #並計算目前狀態價值最高的動作(往左或往右)

    episode_reward = 0
    for t in range(max_number_of_steps):
        #env.render()  #顯示與更新遊戲畫面

        observation, reward, done, info = env.step(action) #執行動作

        action, state = get_action(state, action, observation, reward)  #取得下次動作與下次狀態
        episode_reward += reward #累計這個回合的分數，此分數代表這回合連續推幾次沒有倒下

        if done: #倒了，亦即結束
            print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1,
                last_time_steps.mean()))

            last_time_steps = np.hstack((last_time_steps[1:], [episode_reward])) #將本回合的分數放入last_time_steps
            step_list.append(last_time_steps.mean()) #將這次回合結束時的平均分數放入step_list以便繪製關係圖
            break
    
    if (last_time_steps.mean() >= goal_average_steps): #如果平均分數達到195以上則訓練成功
        print('Episode %d train agent successfully!' % episode)
        break

import matplotlib.pyplot as plt  #繪製出圖形
plt.plot(step_list)
plt.xlabel('episode')
plt.ylabel('mean_step')
plt.show()
