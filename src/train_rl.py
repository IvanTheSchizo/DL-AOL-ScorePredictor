import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.student_env import StudentOptimizationEnv
from src.dqn_agent import Agent
from src.utils import seed_everything

seed_everything(42)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "student_habits_performance.csv")
PREDICTOR_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "final_predictor.pth")
DQN_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "dqn_agent.pth")

def train_rl():
    print(f"Loading Environment with data: {DATA_PATH}")
    env = StudentOptimizationEnv(DATA_PATH, PREDICTOR_PATH)
    
    num_features = env.students.shape[1]
    action_dim = num_features * 2
    
    agent = Agent(state_dim=num_features, action_dim=action_dim)
    
    print("Starting RL Training")
    
    episodes = 500
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.train_step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        if (e+1) % 50 == 0:
            print(f"Episode {e+1}/{episodes} | Total Reward: {total_reward:.4f} | Epsilon: {agent.epsilon:.2f}")

    # Save
    os.makedirs(os.path.dirname(DQN_PATH), exist_ok=True)
    torch.save(agent.dqn.state_dict(), DQN_PATH)
    print(f"DQN Agent saved in {DQN_PATH}")

if __name__ == "__main__":
    train_rl()