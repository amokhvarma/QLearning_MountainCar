import gym
import numpy as np
env = gym.make("MountainCar-v0")
# print(numpy.zeros(9))
learning_rate = 0.1
discount = 0.95
epoch = 25000
show_every = 2000
epsilon = 0.5
start_epsilon_decaying = 1
end_epsilon_decaying = epoch//2
epsilon_decay_value = epsilon/(end_epsilon_decaying-start_epsilon_decaying)

#print(env.reset())
#env.reset()


# print(type(env.observation_space.high))
# print(env.observation_space.low.shape)

discrete_os = [20] * len(env.observation_space.high)
# print(discrete_os)
discrete_os_size = (env.observation_space.high - env.observation_space.low)/discrete_os

q_table = np.random.uniform(low=-2, high=0, size=discrete_os + [env.action_space.n])
# print(discrete_os_size)

def get_disc_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_size
    return tuple(discrete_state.astype(np.int))


#print(discrete_state)


done = False

for epi in range(epoch):
    discrete_state = get_disc_state(env.reset())
    render=False
    if (epi % show_every == 0):
        print("Episode : ", epi)
        render=True
    done=False
    steps=0
    while not done:
        steps=steps+1
        if(np.random.random()>epsilon):
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)


        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_disc_state(new_state)
        #print(epi)
        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1-learning_rate)*current_q+learning_rate*(reward+discount*max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state+(action,)]=0
            print("Reached at episode :", epi,"steps :",steps)

        discrete_state = new_discrete_state

    if end_epsilon_decaying >= epi >= start_epsilon_decaying:
        epsilon -= epsilon_decay_value





#print(new_state)
#env.render()




env.close()
