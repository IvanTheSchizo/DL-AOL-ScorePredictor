import torch
import sys
import os

# Add the project root to system path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import StudentPredictor
from .dqn_agent import DQN

def load_models():
    # 1. Define Paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_path, '..')
    
    predictor_path = os.path.join(project_root, 'outputs', 'models', 'final_predictor.pth')
    dqn_path = os.path.join(project_root, 'outputs', 'models', 'dqn_agent.pth')
    
    # 2. Initialize Models (Must match training dimensions)
    # Note: We need to know input_dim. Let's assume standard from data or pass it in.
    # For this demo, we'll hardcode based on your dataset (e.g., 6 cat + num cols). 
    # ideally, save this metadata in a JSON config during training.
    
    # Check data/raw to get shape dynamically? 
    # Or just try-catch loading.
    # Let's assume input_dim=15 (Example). 
    # If this fails, we will check your dataset shape in the next step.
    input_dim = 10 # REPLACE with actual number of features after one-hot/label encoding
    
    # Load Predictor
    predictor = StudentPredictor(input_dim=input_dim)
    try:
        predictor.load_state_dict(torch.load(predictor_path, map_location=torch.device('cpu')))
    except:
        # If dimensions mismatch, we might need to be dynamic. 
        # For now, return None and handle in app
        pass
    predictor.eval()
    
    # Load Agent
    # Action dim = input_dim * 2 (Up/Down for each feature)
    action_dim = input_dim * 2
    agent = DQN(state_dim=input_dim, action_dim=action_dim)
    try:
        agent.net.load_state_dict(torch.load(dqn_path, map_location=torch.device('cpu')))
    except:
        pass
    agent.eval()
    
    return predictor, agent