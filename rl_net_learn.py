#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import gym
import numpy as np
from random import shuffle
import time

from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten
from keras.utils import print_summary
from keras.optimizers import SGD

class NN:
    def __init__(self):
        inputs1 = Input(shape=(80,80,1))
        conv1 = Conv2D(32,(8,8), strides=4, activation='relu',\
                       kernel_initializer='he_normal')(inputs1)
        conv2 = Conv2D(64,(4,4), strides=2, activation='relu',\
                       kernel_initializer='he_normal')(conv1)
        flat = Flatten()(conv2)
        fc1 = Dense(256, activation='relu',\
                    kernel_initializer='he_normal')(flat)
        out = Dense(3, activation='linear')(fc1)
        
        model = Model(inputs=inputs1,outputs=out)
        
        opt = SGD(lr=1e-5, clipvalue=1.0)
        model.compile(loss='mse', optimizer=opt)
        self.net = model
        
    def get_action(self, x, atype='e-greedy', e_val=0.05):
        """функция возвращает номер действия а, на
        основании входного состояния x
        реализовано 2 типа: жадный, e-жадный
        TODO: softmax
        """
        EPS = e_val
        x = np.array(x)
        net_out = self.net.predict(np.array([x]), batch_size=1)
        if atype == 'greedy':
            action = np.argmax(net_out[0])
        if atype == 'e-greedy':
            rnd = np.random.uniform(0, 1)
            if rnd < EPS:
                action = np.random.randint(0,3)
            else:
                action = np.argmax(net_out[0])  
        return action, net_out[0]
    
    def learn_on_batch(self,s,a,r,tm,ns, batch_size=256):
        """формируем пакеты из текущей памяти
        и отправляем их на обучение в метод optimize_on_memory
        """
        step = np.random.randint(2,3)
        offset = np.random.randint(0,2)
        num_batch = (len(s)-offset) // (batch_size * step)
        index = range(offset, len(s), step)
        shuffle(index)
        for batch in range(num_batch):
            bs, ba, br, btm, bns = [], [], [], [], []
            for i in range(batch*batch_size, batch*batch_size+batch_size):
                bs.append(s[index[i]])
                ba.append(a[index[i]])
                br.append(r[index[i]])
                btm.append(tm[index[i]])
                bns.append(ns[index[i]])
            self.optimize_on_batch(bs,ba,br,btm,bns)
        return bs, ba, br, btm, bns
    
    def optimize_on_batch(self, s, a, r, tm, ns):
        """Обучает на введенных массивах
        [state, action, reward, terminal_marks, new_state] 
        одним пакетом через функцию fit
        """
        GAMMA = 0.99
        train_X = np.array(s)
        ns_x = np.array(ns)
        # активируем выходы, что бы по другим 
        # действиям ошибка = 0
        train_Y = self.net.predict(train_X, batch_size=len(train_X))
        # активируем выходы для новых состояний
        # для maxQ 
        nsQ = self.net.predict(ns_x, batch_size=len(ns_x))
        for i in range(len(train_X)):
            if tm[i] == 0: 
                train_Y[i][a[i]] = GAMMA*max(nsQ[i]) + r[i]
            else: 
                train_Y[i][a[i]] = r[i]
        self.net.fit(train_X, train_Y,\
                     epochs = 1,\
                     verbose = 0,\
                     batch_size = len(train_Y))

def new_rewards(rewards):
    """из вектора rewards сформируем 2 вектора
    1) Ненулевые вознаграждения: Итог / длина_игры. Нормируется
            около среднего и обрезается до -1..1
    2) Массив терминальных меток. 0 - не терминальное состояние.
            1 - терминальное состояние.
    """
    r = np.zeros_like(rewards)
    term_marks = np.zeros_like(rewards)
    start_j = 0
    for i in range(len(rewards)):
        if np.abs(rewards[i]) > 0.01:
            term_marks[i] = 1
            rew = rewards[i] 
            for j in range(start_j, i+1):
                r[j] = rew / (i+1 - start_j)
            start_j = i+1
    m = r.mean()
    std = r.std()
    r = (r-m)/std
    return  r.clip(-1.,1.), term_marks 

def get_state(observation, prev_observatin):
    """получение состояния на основании текущего
    и предыдущего кадров
    пример в файле state.csv
    """
    obs = observation 
    data1 = obs[34:194,:,0:1]
    data1 = data1[::2,::2,:]
    data1[data1==data1[0][0]] = 0
    data1[data1!=0] = 1
    
    obs = observation - prev_observation 
    data2 = obs[34:194,:,0:1]
    data2 = data2[::2,::2,:]
    data2[data2==data2[0][0]] = 0
    data2[data2!=0] = 2
    out = data1 + data2

    img1 = out.astype(np.float) / 3.0

    return img1

#################################################################
env = gym.make('Pong-v0') #action = [1 - stay , 2 - up , 3 - down]
e = 0.65 # EPS
net = NN()
print_summary(net.net)
# загрузка сетей
net.net.load_weights('./save/pong_cnn.h5')
#net.net.load_weights('./save/best_pong_net.h5')
best_perfomance = 5 # текущий лучший вариант для сохранения, если лучше
states, rewards, actions, ns = [], [], [], []
start_time = time.time()
# старт по эпизодам
for epoch in range(10000):
    num_games = 1
    action = np.random.randint(1,4)
    curr_rew = []
    for games in range(num_games):
        observation = env.reset()
        prev_observation = observation
        done = False
        frame = 0
        e = e*0.999
        # цикл для игры, записывается информация в массивы
        # states - состояния
        # actions - действие
        # rewards - вознаграждения среды
        # ns - новые состояния (new states) для maxQ
        while (not done):
#            time.sleep(0.01) #раскомментируй если хочешь смотреть игру
#            env.render() # раскомментируй если хочешь смотреть игру
            observation, reward, done, info = env.step(action+1) # 1 2 3
            if not done:
                if frame == 1:
                    rewards.append(reward)
                    curr_rew.append(reward)
                    new_state = get_state(observation, prev_observation)
                    ns.append(new_state)  
                    frame = 0         
                if frame == 0:
                    state = get_state(observation, prev_observation)
                    prev_observation = observation
                    action, n_out = net.get_action(state, "e-greedy", e)
                    states.append(state)
                    actions.append(action)
                    frame = 1
            else:
                rewards.append(reward)
                curr_rew.append(reward)
                new_state = get_state(observation, prev_observation)
                ns.append(new_state)
    # Оценка игры, вывод информации, обновление лучшей сети
    perfomance = sum(curr_rew)
    print epoch,") WINS = ",\
          sum(curr_rew),', e =',"%1.3f"%e,\
          ', memory = ', len(states),\
          " (","%5.1f"%(time.time()-start_time),"s)"
    if perfomance > best_perfomance:
        best_perfomance = perfomance
        net.net.save_weights('./save/best_pong_net.h5')
        print 'Best Net Saved ...'
    # Обучение, если память >110000
    if len(states) > 50000:
        assert len(rewards) == len(states) == len(actions) == len(ns)
        new_rew, term_marks = new_rewards(rewards)
        assert len(new_rew)==len(term_marks)==len(states)
        net.learn_on_batch(states,actions,new_rew,term_marks,ns) 
    
    # Очистка памяти если превысили 250000 (~25Gb)
    # удаляю старые 10000 элементов
    if len(states) > 150000:
        states, actions, rewards,\
        term_marks, ns = states[-140000:], actions[-140000:], \
                    rewards[-140000:], term_marks[-140000:], ns[-140000:]
    # сохраним текущую сеть
    net.net.save_weights('./save/pong_cnn.h5')
    env.render()
    env.close() 
