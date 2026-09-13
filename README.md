# Customer Churn Prediction

A Streamlit application that predicts customer churn probability and assigns a customer segment using trained machine-learning artifacts.

## Project Files

- `app.py` - Streamlit user interface and prediction logic.
- `customer_churn_artifacts.pkl` - Trained models, preprocessing objects, scalers, and feature definitions required by the app.
- `requirements.txt` - Python dependencies.
- `Procfile` - Start command for hosts that use Procfile-based deployments.
- `runtime.txt` - Python runtime version for compatible hosts.
- `.streamlit/config.toml` - Headless Streamlit server configuration.

## Requirements

- Python 3.9 or newer
- The included `customer_churn_artifacts.pkl` file

## Installation

From the project directory, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit app with:

```powershell
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in a browser.

## Deployment

For Streamlit Community Cloud, create a new app from this repository and select `app.py` as the main file. The platform will install `requirements.txt` automatically. Keep `customer_churn_artifacts.pkl` in the project root because the app loads it with a relative path.

For Procfile-based hosts, use the included `Procfile`. The command binds Streamlit to `0.0.0.0` and uses the host-provided `PORT` value.

## How It Works

1. Enter the customer's profile, service, plan, financial, and value information.
2. Select **Predict Customer**.
3. The app preprocesses the inputs using the saved preprocessor.
4. The logistic regression model returns the churn prediction and probability.
5. The saved scaler and K-Means model assign the customer to a segment.

## Notes

The application expects `customer_churn_artifacts.pkl` to be in the same directory as `app.py`. The artifact file must contain the following keys:

`logistic_model`, `preprocessor`, `kmeans_model`, `scaler`, `numerical_features`, `categorical_features`, and `segmentation_features`.