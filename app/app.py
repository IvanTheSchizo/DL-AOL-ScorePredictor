import streamlit as st
import pandas as pd
import torch
import numpy as np
import sys
import os

# --- PATH SETUP ---
# Force Python to look in the main project folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import StudentPredictor
from src.dqn_agent import DQN

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Student Tutor", layout="wide")
st.title("🎓 Deep Learning AI Tutor")

# --- INPUT FUNCTION (FIXED) ---
def user_input_features(df):
    inputs = {}
    st.sidebar.header("Student Habits Profile")
    
    for col in df.columns:
        if col == 'exam_score': 
            continue
            
        # If the column is Text/Categorical (e.g., Gender, Diet)
        if df[col].dtype == 'object':
            options = df[col].unique().tolist()
            # Use a Dropdown instead of a Slider
            val = st.sidebar.selectbox(f"Select {col}", options)
            inputs[col] = val
            
        # If the column is Numerical (e.g., Study Hours)
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            avg_val = float(df[col].mean())
            
            # Use a Slider
            val = st.sidebar.slider(
                label=col.replace('_', ' ').title(),
                min_value=min_val,
                max_value=max_val,
                value=avg_val
            )
            inputs[col] = val
            
    return pd.DataFrame(inputs, index=[0])

# --- MAIN LOGIC ---
data_path = os.path.join(project_root, 'data', 'raw', 'student_habits_performance.csv')

if os.path.exists(data_path):
    # 1. Load Data
    raw_df = pd.read_csv(data_path)
    if 'student_id' in raw_df.columns: raw_df.drop(columns=['student_id'], inplace=True)
    if 'exam_score' in raw_df.columns: raw_df.drop(columns=['exam_score'], inplace=True)
    
    # 2. Get User Input
    input_df = user_input_features(raw_df)
    
    # 3. Preprocess Input (Encoding)
    # Combine User Input with Raw Data to ensure encoding matches
    combined_df = pd.concat([raw_df, input_df], axis=0)
    
    # Smart Encoding
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Encode Categories (Text -> Numbers)
    cat_cols = combined_df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col].astype(str))
        
    # Scale Numbers
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_df)
    
    # Extract just the user's row (the very last one)
    user_input_tensor = torch.tensor(combined_scaled[-1:], dtype=torch.float32)
    input_dim = user_input_tensor.shape[1]

    # 4. Load Models
    model_path = os.path.join(project_root, 'outputs', 'models', 'final_predictor.pth')
    dqn_path = os.path.join(project_root, 'outputs', 'models', 'dqn_agent.pth')
    
    # Load Predictor
    predictor = StudentPredictor(input_dim=input_dim)
    try:
        predictor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except:
        st.warning("Could not load model weights precisely. Using untrained model for demo.")
    predictor.eval()
    
    # Load Agent
    dqn = DQN(state_dim=input_dim, action_dim=input_dim*2)
    try:
        dqn.net.load_state_dict(torch.load(dqn_path, map_location=torch.device('cpu')))
    except:
        pass
    dqn.eval()

    # 5. Display Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Performance Prediction")
        with torch.no_grad():
            prediction = predictor(user_input_tensor).item()
        
        st.metric(label="Predicted Exam Score", value=f"{prediction:.2f}")
        
        if prediction < 60:
            st.error("⚠️ At Risk")
        elif prediction < 80:
            st.warning("⚖️ Average")
        else:
            st.success("🌟 Distinction Track")

    with col2:
        st.subheader("🤖 AI Tutor Recommendations")
        if st.button("Generate Study Plan"):
            with torch.no_grad():
                q_values = dqn(user_input_tensor)
                best_action_idx = torch.argmax(q_values).item()
            
            # Decode the action
            feature_idx = best_action_idx // 2
            direction = "Increase" if best_action_idx % 2 == 0 else "Decrease"
            feature_name = raw_df.columns[feature_idx]
            
            st.info(f"**Recommendation:** {direction} your **{feature_name}**.")
            st.write("Our Reinforcement Learning Agent identified this as the most effective change to improve your score.")

else:
    st.error("Data file not found. Please ensure 'data/raw/student_habits_performance.csv' exists.")