# Deep Learning Student Performance Optimizer 🎓🚀

A comprehensive Deep Learning system that moves beyond passive grade prediction to active lifestyle optimization. This project uses a hybrid architecture of **Unsupervised**, **Supervised**, and **Reinforcement Learning** to predict student exam scores and act as an "AI Tutor" to suggest personalized habit improvements.

## 🧠 Model Architecture

The system utilizes an end-to-end pipeline consisting of three core algorithms:

1.  **Denoising Autoencoder (DAE):**
    * *Unsupervised Learning.*
    * Extracts robust, latent features from noisy student data by learning to reconstruct clean inputs from corrupted versions.
2.  **Attention-Based Predictor:**
    * *Supervised Learning.*
    * Dynamically weights input features (e.g., focusing more on Sleep vs. Study depending on the student) to predict the final **Exam Score**.
3.  **Deep Q-Network (DQN) Agent:**
    * *Reinforcement Learning.*
    * Acts as an "AI Tutor" that explores simulated scenarios to find the optimal sequence of lifestyle changes to maximize academic performance.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/student-performance-dl.git](https://github.com/yourusername/student-performance-dl.git)
    cd student-performance-dl
    ```

2.  **Create a virtual environment (Optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Run the Web Application (Demo)
To launch the interactive dashboard with the AI Tutor:
```bash
streamlit run app/app.py
