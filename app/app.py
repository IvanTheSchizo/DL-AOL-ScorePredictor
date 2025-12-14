import streamlit as st
import pandas as pd
import torch
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import StudentPredictor
from src.dqn_agent import DQN

st.set_page_config(
    page_title="Deep Learning Habit-Based Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    
    /* Titles */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        color: #FFFFFF;
    }
    
    /* Success/Warning Messages */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Custom Button */
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        border: none;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

col_head1, col_head2 = st.columns([1, 4])
with col_head1:
    st.image("https://img.icons8.com/fluency/96/student-male.png", width=80)
with col_head2:
    st.title("Deep Learning Habit-Based Score Predictor")
    st.caption("A Deep Learning AOL Project")

st.markdown("---")

def user_input_features(df):
    inputs = {}
    st.sidebar.header("📝 Student Profile")
    st.sidebar.markdown("Adjust the habits below to see how they impact performance.")
    
    manual_defaults = {
        'age': 20,
        'study_hours': 3.5,
        'sleep_hours': 7,
        'social_media_hours': 2,
        'attendance_rate': 85,
    }
    
    for col in df.columns:
        if col == 'exam_score': continue
            
        if df[col].dtype == 'object':
            options = df[col].unique().tolist()
            val = st.sidebar.selectbox(f"{col.replace('_', ' ').title()}", options)
            inputs[col] = val
        else:
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            avg_val = int(df[col].mean())

            if 'age' or 'mental' or 'exercise' in col.lower():
                step_val = 1
                format_str = "%d"
            elif 'study' in col.lower():
                step_val = 0.5
                format_str = "%.1f"
            elif 'attendance' in col.lower():
                step_val = 1
                format_str = "%d"
            else:
                step_val = 0.5
                format_str = "%.1f"
            
            if col in manual_defaults:
                default_val = float(manual_defaults[col])
            else:
                default_val = float(avg_val)
            
            default_val = max(float(min_val), min(float(max_val), default_val))
            
            val = st.sidebar.slider(
                label=col.replace('_', ' ').title(),
                min_value=float(min_val),
                max_value=float(max_val),
                value=default_val,
                step=float(step_val),
                format=format_str
            )
            inputs[col] = val
            
    return pd.DataFrame(inputs, index=[0])

data_path = os.path.join(project_root, 'data', 'raw', 'student_habits_performance.csv')

if os.path.exists(data_path):
    raw_df = pd.read_csv(data_path)
    if 'student_id' in raw_df.columns: raw_df.drop(columns=['student_id'], inplace=True)
    if 'exam_score' in raw_df.columns: raw_df.drop(columns=['exam_score'], inplace=True)
    
    input_df = user_input_features(raw_df)
    
    combined_df = pd.concat([raw_df, input_df], axis=0)
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    cat_cols = combined_df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col].astype(str))
        
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_df)
    user_input_tensor = torch.tensor(combined_scaled[-1:], dtype=torch.float32)
    input_dim = user_input_tensor.shape[1]

    model_path = os.path.join(project_root, 'outputs', 'models', 'final_predictor.pth')
    dqn_path = os.path.join(project_root, 'outputs', 'models', 'dqn_agent.pth')
    
    predictor = StudentPredictor(input_dim=input_dim)
    try:
        predictor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except:
        st.error("Model weights not found. Please train the model first.")
    predictor.eval()
    
    dqn = DQN(state_dim=input_dim, action_dim=input_dim*2)

    if os.path.exists(dqn_path):
        dqn.load_state_dict(torch.load(dqn_path, map_location=torch.device('cpu')))
    else:
        st.error(f"RL Agent model file not found at: {dqn_path}")
        
    dqn.eval()

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Performance Prediction")
        
        with torch.no_grad():
            raw_prediction = predictor(user_input_tensor).item()
            prediction = max(0.0, min(100.0, raw_prediction))
        
        st.progress(min(prediction/100, 1.0))
        
        st.metric(label="Predicted Final Score", value=f"{prediction:.1f}/100")
        
        if prediction < 60:
            st.error("**At risk, needs major improvements**.")
        elif prediction < 80:
            st.warning("**Average, room for improvement**")
        else:
            st.success("**Great, excellent habits**")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Analysis")
        
        if st.button("Get Suggestion"):
            with torch.no_grad():
                q_values = dqn(user_input_tensor)
                best_action_idx = torch.argmax(q_values).item()
            
            feature_idx = best_action_idx // 2
            direction = "INCREASE" if best_action_idx % 2 == 0 else "DECREASE"
            feature_name = raw_df.columns[feature_idx].replace('_', ' ').upper()
            
            st.markdown(f"""
            ### Recommendation
            To maximize your exam score, you can:
            
            # **{direction} {feature_name}**
            
            """)
            
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("Data file not found. Please ensure 'data/raw/student_habits_performance.csv' exists.")