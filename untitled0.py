import gym
import random
env = gym.make('LunarLander-v2', render_mode = 'human')
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = env.action_space.sample()
        _, reward, done,_ = env.step(action)
        score+=reward
        env.render()
    print(f'Episode {episode}, Score {score}')
env.close()