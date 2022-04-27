import gym
import numpy as np
import matplotlib.pyplot as plt

def do_episode(w, env):  #傳入目前的w參數與本回合的遊戲環境(env)完成一回合的遊戲
    done = False
    observation = env.reset()
    num_steps = 0

    while not done and num_steps <= max_number_of_steps:
        action = take_action(observation, w)
        observation, _, done, _ = env.step(action)
        num_steps += 1

    step_val = -1 if num_steps >= max_number_of_steps else num_steps - max_number_of_steps
    #步數小於200則獎勵為-200

    return step_val, num_steps #傳回本回合的獎勵與本回合所有的步數

def take_action(X, w): 
    action = 1 if calculate(X, w) > 0.0 else 0
    return action
#傳入四個環境值X與目前的w參數，呼叫calculate()，計算後取得動作的機率值
#如果機率值大於0就傳回1(代表往右推)，否則的話就傳回0(往左推)

def calculate(X, w): #傳入四個環境值X與目前的w參數，計算後取得動作的機率值
    result = np.dot(X, w) #計算結果是數值不是陣列
    return result

env = gym.make('CartPole-v0')

#env.render()

eta = 0.2 #學習率
sigma = 0.05 #影響參數的標準差數值

max_episodes = 5000 #執行回合
max_number_of_steps = 200 #每回合步驟，即最多推幾次
n_states = 4 # w參數的數量
num_batch = 10 #批次訓練的每批回合數
num_consecutive_iterations = 100 #取最後100回合來計算平均分數

w = np.random.randn(n_states) #初始化
reward_list = np.zeros(num_batch) #初始化
reward_h = [] #初始化
last_time_steps = np.zeros(num_consecutive_iterations) #初始化
mean_list = [] #初始化

for episode in range(max_episodes//num_batch): #開始分批訓練
    N = np.random.normal(scale=sigma,size=(num_batch, w.shape[0])) #用來改寫參數的值，視為偏差值
    for i in range(num_batch):
        w_try = w + N[i] #加入偏差值
        reward, steps = do_episode(w_try, env) #完成本回合的遊戲然後取得回合的獎勵與本回合所有的步數
        if i == num_batch-1:
            print('%d Episode finished after %d steps / mean %f' %(episode*num_batch, steps, last_time_steps.mean()))
        last_time_steps = np.hstack((last_time_steps[1:], [steps]))
        reward_list[i] = reward #將本回合的步數與獎勵放入last_time_steps與reward_list中
        mean_list.append(last_time_steps.mean()) #將這次回合結束時的平均分數放入mean_list以便繪製關係圖

    if last_time_steps.mean() >= 195: break 

    std = np.std(reward_list)
    if std == 0: std = 1
    A = (reward_list - np.mean(reward_list))/std #正規化獎勵的數值
    w_delta = eta /(num_batch*sigma) * np.dot(N.T, A) #偏微分計算
    w += w_delta #更新w參數


env.close()

plt.plot(mean_list)
plt.xlabel("episode")
plt.ylabel("mean_step")
plt.show()
