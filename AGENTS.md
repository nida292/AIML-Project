# AGENTS.md

## Project: Machine Learning-Based Network Intrusion Detection

This project uses the RT-IoT2022 dataset to classify network traffic into
normal or specific attack categories using supervised machine learning.

### Key Files
- `src/train.py` — Main training pipeline
- `src/evaluate.py` — Model evaluation
- `src/predict.py` — Prediction on new data
- `app.py` — Streamlit web interface

### Commands
```bash
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
streamlit run app.py
```

### Important Rules
- Do not modify the ML pipeline to leak test data into training.
- Preprocessing must be fitted only on training data.
- Do not fabricate results or metrics.
- Keep random_state=42 for reproducibility.
