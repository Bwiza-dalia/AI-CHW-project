# 🚀 Streamlit Cloud Deployment Guide

## Prerequisites
- GitHub repository with the project code
- Streamlit Cloud account (free)

## Deployment Steps

### 1. Repository Setup
- Ensure all files are committed to GitHub
- Make sure `requirements.txt` is in the root directory
- Verify `.streamlit/config.toml` exists

### 2. Streamlit Cloud Deployment
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `Bwiza-dalia/AI-CHW-project`
5. Set main file path: `app/streamlit_app.py`
6. Click "Deploy"

### 3. Configuration
- **App URL**: Will be provided after deployment
- **Repository**: `Bwiza-dalia/AI-CHW-project`
- **Branch**: `main`
- **Main file**: `app/streamlit_app.py`

## Troubleshooting

### Common Issues:
1. **ModuleNotFoundError**: Fixed with path configuration in `streamlit_app.py`
2. **Memory issues**: Streamlit Cloud provides 1GB RAM
3. **Large files**: Use Git LFS for files >50MB

### File Structure:
```
AI-CHW-project/
├── app/
│   └── streamlit_app.py
├── src/
│   └── models/
│       ├── grading.py
│       ├── content.py
│       ├── analytics.py
│       └── recommend.py
├── data/
│   └── *.csv
├── requirements.txt
└── .streamlit/
    └── config.toml
```

## Features Available in Cloud:
- ✅ AI Q&A Assistant
- ✅ AI Grading System
- ✅ Content Management
- ✅ Analytics Dashboard
- ✅ Visual Diagrams
- ✅ Module Recommendations

## Performance Notes:
- First load may take 30-60 seconds (model loading)
- Subsequent loads are faster
- Data is loaded from CSV files (no database required)
