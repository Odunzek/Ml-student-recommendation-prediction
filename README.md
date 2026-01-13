# Student Success Predictor

An AI-powered early warning system that predicts student exam scores and identifies at-risk students for timely intervention.

## Overview

This project uses machine learning to predict student exam performance based on behavioral, environmental, and academic factors. It features an interactive dashboard with explainability, batch processing, and prediction history tracking.

### Key Features

- **Accurate Predictions**: XGBoost regression model with MAE of 2.7 points
- **SHAP Explainability**: Understand which factors influence each prediction
- **Batch Processing**: Analyze entire classrooms at once via CSV upload
- **Prediction History**: Track and analyze predictions over time
- **Flexible Risk Thresholds**: Adjust at-risk cutoffs dynamically
- **MLflow Integration**: Complete experiment tracking and model versioning

## Demo

### Single Student Prediction
![Dashboard Screenshot](docs/dashboard_preview.png)

**Features:**
- Real-time prediction with interactive input form
- Risk classification with visual gauge chart
- SHAP feature importance visualization
- Personalized recommendations

### Batch Upload
Upload CSV files to analyze multiple students simultaneously and download results.

### Prediction History
View all past predictions with summary statistics and export capabilities.

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/student-success-predictor.git
   cd student-success-predictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd project/dashboard
   pip install -r requirements.txt
   ```

4. **Create the ML pipeline**
   ```bash
   cd ../notebooks
   python create_pipeline_mlflow.py
   ```

5. **Run the dashboard**
   ```bash
   cd ../dashboard
   streamlit run app_improved.py
   ```

6. **Open your browser**
   ```
   http://localhost:8502
   ```

## Project Structure

```
student-success-predictor/
├── README.md                          # This file
├── QUICK_START.md                     # Detailed setup guide
├── IMPROVEMENTS_SUMMARY.md            # Technical documentation
├── MLFLOW_GUIDE.md                    # MLflow usage guide
├── BATCH_TESTING_GUIDE.md            # Batch upload documentation
├── GIT_SETUP_GUIDE.md                # Git configuration guide
│
├── sample_students_batch.csv          # Sample data (20 students)
├── sample_at_risk_students.csv        # Sample at-risk data (10 students)
│
└── project/
    ├── data/
    │   └── StudentPerformanceFactors.csv    # Original dataset
    │
    ├── notebooks/
    │   ├── student_recommendation.ipynb     # ML training notebook
    │   └── create_pipeline_mlflow.py        # Pipeline creation script
    │
    └── dashboard/
        ├── app_improved.py                  # Main Streamlit dashboard
        └── requirements.txt                 # Python dependencies
```

## Usage

### Single Student Prediction

1. Navigate to the dashboard (http://localhost:8502)
2. Fill in student information:
   - Academic metrics (hours studied, attendance, previous scores)
   - Family & support (parental involvement, resources, income)
   - School environment (teacher quality, peer influence)
3. Adjust risk threshold (default: 65)
4. Click "Analyze Student Risk"
5. View:
   - Predicted exam score
   - Risk classification
   - SHAP feature importance
   - Personalized recommendations

### Batch Processing

1. Prepare CSV file with required columns (see `sample_students_batch.csv`)
2. Navigate to "Batch Upload" page
3. Upload CSV file
4. Click "Run Batch Prediction"
5. Download results with predictions

### Required CSV Format

```csv
Hours_Studied,Attendance,Parental_Involvement,Access_to_Resources,...
20,85,Medium,High,...
15,70,Low,Medium,...
```

See `BATCH_TESTING_GUIDE.md` for complete column specifications.

## Model Performance

| Metric | Train | Test |
|--------|-------|------|
| **MAE** | 2.12 | 2.71 |
| **RMSE** | 3.0 | 3.7 |
| **R²** | 0.89 | 0.79 |

**Interpretation:** Predictions are within ±2.7 points on average, explaining 79% of variance in exam scores.

## Technology Stack

- **ML Framework**: scikit-learn, XGBoost
- **Experiment Tracking**: MLflow
- **Dashboard**: Streamlit
- **Explainability**: SHAP
- **Visualization**: Plotly, Matplotlib
- **Data Processing**: Pandas, NumPy

## Model Architecture

```
Input (Student Features)
    ↓
Categorical Encoding
    ↓
sklearn Pipeline
├── StandardScaler (numeric features)
└── XGBoost Regressor
    ↓
Predicted Exam Score (0-100)
    ↓
Risk Classification (threshold-based)
```

## Features Used

**High Impact:**
- Attendance
- Hours Studied
- Previous Scores
- Sleep Hours
- Physical Activity

**Medium Impact:**
- Parental Involvement
- Access to Resources
- Tutoring Sessions
- Parental Education Level

## Dataset

**Source:** Synthetic educational data
**Size:** 7,907 students (6,607 original + 1,300 synthetic at-risk cases)
**Features:** 19 behavioral, environmental, and academic factors
**Target:** Exam scores (0-100)

### Data Augmentation

To address the lack of failing students in the original dataset, we generated 1,300 synthetic at-risk students with:
- Lower attendance and study hours
- Weaker support systems
- Realistic variability in risk factors

This ensures the model learns patterns associated with academic risk.

## Documentation

- **[QUICK_START.md](QUICK_START.md)** - Quick reference guide
- **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** - Technical details & architecture
- **[MLFLOW_GUIDE.md](MLFLOW_GUIDE.md)** - MLflow tracking and versioning
- **[BATCH_TESTING_GUIDE.md](BATCH_TESTING_GUIDE.md)** - Batch upload feature guide
- **[GIT_SETUP_GUIDE.md](GIT_SETUP_GUIDE.md)** - Git configuration and file management

## MLflow Tracking

View experiment tracking:

```bash
cd project/notebooks
mlflow ui
# Open http://localhost:5000
```

**What's Tracked:**
- Model hyperparameters
- Training/test metrics
- Model artifacts (full pipeline)
- Dataset information

**Model Registry:** `StudentExamScorePredictor`

## Why Regression Instead of Classification?

This project uses regression (predicting exact scores) rather than classification (at-risk/not at-risk) because:

✅ **More Informative**: Know exact predicted score (58 vs 62 vs 45)
✅ **Flexible Thresholds**: Schools can adjust risk cutoffs without retraining
✅ **Prioritization**: Identify severity (score 30 needs more help than score 59)
✅ **Actionable**: Track progress over time with continuous values

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Deploy to cloud (Streamlit Cloud, Heroku, AWS)
- [ ] Add user authentication
- [ ] Intervention tracking system
- [ ] Automated retraining pipeline
- [ ] Integration with student information systems
- [ ] Mobile-responsive design
- [ ] Email alerts for at-risk students

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset inspired by educational performance research
- Built with Streamlit, scikit-learn, and XGBoost
- SHAP for model explainability

## Contact

**Project Maintainer:** [Your Name]
**Email:** your.email@example.com
**GitHub:** [@yourusername](https://github.com/yourusername)

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{student_success_predictor,
  author = {Your Name},
  title = {Student Success Predictor: AI-Powered Early Warning System},
  year = {2025},
  url = {https://github.com/yourusername/student-success-predictor}
}
```

---

**Built with ❤️ for educational success**
