# 🏦 Credit Approval Web Application

An AI-powered web application for credit card and loan approval decisions using advanced machine learning models with a conservative, risk-focused approach.

## 🎯 Features

### 📱 **Single Application Processing**
- Interactive web form for individual credit applications
- Instant risk assessment and approval decision
- Detailed risk analysis with 5-tier classification
- Business recommendations for each decision

### 📊 **Batch Processing**
- Upload CSV files with multiple applications
- Process hundreds of applications in seconds
- Comprehensive results dashboard with charts
- Export results to CSV format

### 🧠 **Advanced AI Models**
- Ensemble machine learning approach
- Conservative threshold (0.90) for risk minimization
- Real-time probability scoring
- Risk categorization: Very Low → Very High

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required packages (see requirements.txt)

### Installation

1. **Clone or download this project**
   ```bash
   cd credit-card-approval-app
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```

3. **Train models (if not already done)**
   - Open `notebooks/preprocessing-feature-engineering.ipynb`
   - Run all cells to train the models
   - Run the last cell to save models

4. **Start the web application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   ```
   http://localhost:5000
   ```

## 📋 Usage

### Single Application
1. Navigate to "Single Application"
2. Fill out the applicant information form
3. Click "Calculate Risk & Get Decision"
4. View detailed results and recommendations

### Batch Processing
1. Navigate to "Batch Processing"
2. Download the CSV template (optional)
3. Upload your CSV file with applications
4. View comprehensive results dashboard
5. Export results if needed

### Required CSV Columns
For batch processing, your CSV must include these columns:

**Personal Information:**
- `CODE_GENDER` (M/F)
- `AGE_YEARS` (18-100)
- `CNT_CHILDREN` (0+)
- `CNT_FAM_MEMBERS` (1+)
- `NAME_FAMILY_STATUS`

**Financial Information:**
- `AMT_INCOME_TOTAL` (annual income)
- `NAME_INCOME_TYPE`
- `NAME_EDUCATION_TYPE`
- `OCCUPATION_TYPE`
- `EMPLOYED_YEARS`

**Assets:**
- `FLAG_OWN_CAR` (Y/N)
- `FLAG_OWN_REALTY` (Y/N)
- `NAME_HOUSING_TYPE`

## 🎯 Model Performance

Our conservative risk-focused model achieves:
- **92.0%** Overall Accuracy
- **34.1%** Risk Recall (catching risky customers)
- **7.7%** Risk Precision (minimizing false alarms)
- **0.90** Conservative Threshold
- **$1,312** Business Cost (vs $1,230 baseline)

## 🏗️ Technical Architecture

### Backend
- **Flask** web framework
- **scikit-learn** & **XGBoost** for ML models
- **pandas** for data processing
- **joblib** for model serialization

### Frontend
- **Bootstrap 5** for responsive UI
- **Chart.js** for data visualization
- **Font Awesome** for icons
- Vanilla JavaScript for interactions

### Models
- **Ensemble Approach**: 3 XGBoost models
- **ADASYN Sampling**: Handles class imbalance
- **Conservative Threshold**: 0.90 for risk minimization
- **Risk Classification**: 5-tier system

## 📊 Risk Assessment Levels

| Level | Threshold | Recommendation | Action |
|-------|-----------|----------------|---------|
| **Very Low** | < 20% | ✅ Auto-Approve | Standard terms |
| **Low** | 20-40% | ✅ Standard Approval | Regular monitoring |
| **Medium** | 40-60% | ⚠️ Enhanced Monitoring | Lower limits |
| **High** | 60-90% | ❌ Manual Review | Additional docs |
| **Very High** | > 90% | 🚫 Auto-Reject | High default risk |

## 🔧 Configuration

### Business Cost Matrix
Edit in `app.py`:
```python
COST_FALSE_POSITIVE = 1   # Cost of rejecting good customer
COST_FALSE_NEGATIVE = 10  # Cost of approving risky customer
```

### Risk Threshold
Adjust conservative threshold:
```python
optimal_threshold = 0.90  # Higher = more conservative
```

## 📁 Project Structure

```
credit-card-approval-app/
├── app.py                 # Main Flask application
├── setup.py              # Setup script
├── requirements.txt      # Python dependencies
├── models/               # Saved ML models
│   ├── risk_ensemble_model.pkl
│   ├── preprocessor.pkl
│   └── model_parameters.pkl
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── single_prediction.html
│   ├── batch_prediction.html
│   ├── result_single.html
│   └── result_batch.html
├── uploads/              # Temporary file uploads
├── notebooks/            # Jupyter notebooks
│   └── preprocessing-feature-engineering.ipynb
└── data/                 # Training data
    └── cleaned_credit_data.csv
```

## 🛡️ Security Features

- File upload validation (CSV only, 16MB max)
- Secure filename handling
- Input sanitization
- No data persistence (files deleted after processing)
- CSRF protection with secret key

## 🔄 API Endpoints

### REST API
```bash
POST /api/predict
Content-Type: application/json

{
  "CODE_GENDER": "M",
  "AGE_YEARS": 35,
  "AMT_INCOME_TOTAL": 50000,
  ...
}
```

Response:
```json
{
  "success": true,
  "result": {
    "risk_probability": 25.3,
    "risk_level": "LOW",
    "recommendation": {
      "decision": "APPROVED",
      "reason": "Standard approval with good terms"
    }
  }
}
```

## 🚨 Troubleshooting

### Models Not Found
1. Ensure you've run the notebook to train models
2. Check that `models/` directory contains all .pkl files
3. Run the model saving cell in the notebook

### Installation Issues
1. Check Python version (3.8+ required)
2. Try: `pip install --upgrade pip`
3. Install packages individually if needed

### Performance Issues
1. Use smaller batch sizes for large CSV files
2. Consider upgrading to more powerful hardware
3. Optimize model parameters if needed

## 📈 Business Integration

### For Production Use:
1. **Database Integration**: Connect to your customer database
2. **Authentication**: Add user login/permissions
3. **Audit Logging**: Track all decisions
4. **API Integration**: Connect to loan origination systems
5. **Monitoring**: Set up model performance tracking

### Customization:
- Adjust risk thresholds based on business needs
- Modify cost matrix for your specific use case
- Add additional features/data sources
- Implement A/B testing for model updates

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the notebook for model training details
3. Ensure all requirements are installed correctly

## 🎉 Success Metrics

After deployment, monitor:
- **Approval Rate**: Target ~7-10% for conservative approach
- **Default Rate**: Should be minimized with 0.90 threshold
- **Customer Satisfaction**: Balance with business risk
- **Processing Time**: Should be < 2 seconds per application

---

**Ready to make intelligent credit decisions with AI! 🚀**
