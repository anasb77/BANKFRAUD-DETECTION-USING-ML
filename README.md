# FraudShield Pro

AI-powered bank fraud detection system with a professional Windows desktop application.

## Features

- **Real-time Analysis**: Analyze transactions instantly
- **4 ML Models**: Random Forest, XGBoost, LightGBM, CatBoost
- **99.99% AUC-ROC**: State-of-the-art detection accuracy
- **Live Simulation**: Watch fraud detection in action
- **Sound Feedback**: Audio cues for interactions

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (first time only)
python train_models.py

# Run the app
python app_desktop.py
```

## Build Executable

```bash
pyinstaller build.spec --clean
```

## Project Structure

```
├── app_desktop.py      # Main application
├── train_models.py     # Model training script
├── generate_dataset.py # Dataset generator
├── test_qa.py          # QA test suite
├── build.spec          # PyInstaller config
├── requirements.txt    # Dependencies
├── icon.png/ico        # App logo
├── *.wav               # Sound effects
├── models/             # Trained models
├── data/               # Dataset
└── notebooks/          # Jupyter notebook
```

## Tech Stack

- Python 3.10+
- CustomTkinter (UI)
- scikit-learn, XGBoost, LightGBM, CatBoost (ML)
- Optuna (Hyperparameter tuning)
- SMOTE (Class balancing)

## Performance

| Model | AUC-ROC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| LightGBM | 99.99% | 96.59% | 97.33% | 96.96% |
| CatBoost | 99.98% | 92.42% | 97.71% | 94.99% |
| XGBoost | 99.97% | 90.49% | 98.09% | 94.14% |
| Random Forest | 99.94% | 89.44% | 96.95% | 93.04% |

## License

MIT
