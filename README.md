# Car Price Predictor (Flask Web App)

This is a Flask web application that predicts car prices based on user input using a pre-trained linear regression model.

## Features

- User-friendly web form for car attributes
- Predicts car price using a machine learning model
- Uses pandas for data processing and scikit-learn for the model

## Setup Instructions

1. **Clone the repository** and navigate to the `6 - Flask Web` directory.

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```sh
    python app.py
    ```

4. **Open your browser** and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to use the app.

## Files

- `app.py`: Main Flask application.
- `clean_df.csv`: Cleaned dataset for input structure.
- `linear_regression_model.joblib`: Pre-trained regression model.
- `templates/predict.html`: HTML template for the web form.
- `requirements.txt`: Python dependencies.

## Notes

- Make sure you have Python 3.7+ installed.
- The model and data files must be present in the same directory as `app.py`.
