import numpy as np
import gymnasium as gym

class RLModel:
    def __init__(self, env_name=None):
        self.learning_rate = 0.9
        self.gamma = 0.95
        self.epsilon = 0.8
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.episodes = 10000
        
        self.env, self.n_states, self.n_actions = self.setup_env()
        self.q_table = np.zeros((self.n_states, self.n_actions))


    def setup_env(self, is_slippery = False):
        self.env = gym.make('FrozenLake-v1', desc = None, map_name = '4x4', is_slippery=is_slippery)

        return self.env, self.env.observation_space.n, self.env.action_space.n

    def choose_action(self, state):
        random_n = np.random.rand()
        if random_n < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state,done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] = self.q_table[state][action] + self.learning_rate * (target - self.q_table[state][action])

    def train(self):
        count = 0
        for episode in range(self.episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                if reward == 1.0:
                    count += 1
                self.update_q_table(state, action, reward, next_state, terminated)
                state = next_state
                done = terminated or truncated
            self.epsilon = self.epsilon * self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
            if (episode % 200 == 0):
                print(f'Episode: {episode} && Win rate: {count/(episode +1)*100}%')

if __name__ == "__main__":
    model = RLModel()
    model.train()
