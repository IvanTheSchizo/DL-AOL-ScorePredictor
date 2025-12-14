import torch
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import StudentPredictor
from src.dqn_agent import DQN

def load_models():
    base_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_path, '..')
    
    data_path = os.path.join(project_root, 'data', 'raw', 'student_habits_performance.csv')
    predictor_path = os.path.join(project_root, 'outputs', 'models', 'final_predictor.pth')
    dqn_path = os.path.join(project_root, 'outputs', 'models', 'dqn_agent.pth')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, nrows=1)
        
        if 'student_id' in df.columns:
            df = df.drop(columns=['student_id'])
        if 'exam_score' in df.columns:
            df = df.drop(columns=['exam_score'])
            
        input_dim = df.shape[1]
        print(f"Input Dimension: {input_dim}")
    else:
        print("Defaulting to 10")
        input_dim = 10 
    
    predictor = StudentPredictor(input_dim=input_dim)
    try:
        predictor.load_state_dict(torch.load(predictor_path, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Could not load predictor weights: {e}")
    predictor.eval()
    
    action_dim = input_dim * 2
    agent = DQN(state_dim=input_dim, action_dim=action_dim)
    try:
        agent.net.load_state_dict(torch.load(dqn_path, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Could not load DQN weights: {e}")
    agent.eval()
    
    return predictor, agent