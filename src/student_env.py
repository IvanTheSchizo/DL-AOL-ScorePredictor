import torch
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import StudentPredictor
from src.data_loader import load_and_process_data

class StudentOptimizationEnv:
    def __init__(self, data_path, model_path):
        self.X, _, self.scaler = load_and_process_data(data_path)
        self.students = torch.tensor(self.X, dtype=torch.float32)
        
        input_dim = self.students.shape[1]
        self.model = StudentPredictor(input_dim=input_dim) 
        
        try:
            self.model.load_state_dict(torch.load(model_path))
        except RuntimeError:
            self.model.load_state_dict(torch.load(model_path), strict=False)
            
        self.model.eval()
        
        self.current_student_idx = 0
        self.current_state = None
        self.current_score = 0
        self.steps = 0
        self.max_steps = 5

    def reset(self):
        self.current_student_idx = np.random.randint(0, len(self.students))
        self.current_state = self.students[self.current_student_idx].clone()
        
        with torch.no_grad():
            self.current_score = self.model(self.current_state.unsqueeze(0)).item()
            
        self.steps = 0
        return self.current_state

    def step(self, action):
        feature_idx = action // 2
        direction = 1 if action % 2 == 0 else -1

        self.current_state[feature_idx] += direction * 0.1 
        self.steps += 1
        
        with torch.no_grad():
            new_score = self.model(self.current_state.unsqueeze(0)).item()
        
        reward = new_score - self.current_score
        self.current_score = new_score
        
        done = self.steps >= self.max_steps
        
        return self.current_state, reward, done