# Hotel Booking Cancellation Prediction and Reservation Analysis System

## Project Overview

The **Hotel Booking Cancellation Prediction and Reservation Analysis System** is a Machine Learning project that predicts whether a hotel reservation is likely to be cancelled or not cancelled.

The system also provides a reservation analysis dashboard to explore booking patterns, cancellation trends, hotel types, market segments, and customer information.

This project uses the Hotel Booking Demand dataset and a Random Forest Classification model to analyse booking details and make predictions.

## Project Objectives

- Predict whether a hotel booking will be cancelled.
- Analyse hotel reservation patterns and trends.
- Identify booking cancellation rates.
- Visualize important reservation statistics.
- Provide an interactive web application for prediction and analysis.

## Dataset Description

**Dataset:** Hotel Booking Demand Dataset

**Source:** Kaggle – Hotel Booking Demand

The dataset contains hotel reservation information from City Hotels and Resort Hotels.

**Target Variable:** `is_canceled`

- `0` – Not Cancelled
- `1` – Cancelled

The dataset includes information such as:
- Hotel type
- Lead time
- Arrival month
- Number of adults and children
- Meal type
- Country
- Market segment
- Distribution channel
- Previous cancellations
- Booking changes
- Deposit type
- Average Daily Rate (ADR)
- Special requests

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Project Workflow

1. **Data Collection** – Import the Hotel Booking Demand dataset.
2. **Data Cleaning** – Handle missing values, duplicate records, and unnecessary columns.
3. **Exploratory Data Analysis (EDA)** – Analyse booking patterns and cancellation trends using visualizations.
4. **Train-Test Split** – Separate the dataset into training and testing sets.
5. **Data Preprocessing** – Perform categorical encoding and other required transformations.
6. **Model Training** – Train a Random Forest Classification model.
7. **Model Evaluation** – Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix.
8. **Model Saving** – Save the trained model and preprocessing objects using Joblib.
9. **Web Application** – Develop an interactive Streamlit application for cancellation prediction and reservation analysis.

## Data Preprocessing

The following preprocessing steps are used:

- Handling missing values
- Removing duplicate records
- Removing unnecessary columns
- Encoding categorical variables
- Preparing training and testing datasets

**Preventing Data Leakage:**

The dataset should be split into training and testing sets before fitting preprocessing objects.

- Fit the encoder using training data.
- Transform training and testing data using the fitted encoder.
- Fit the scaler using training data only, when scaling is required.
- Transform testing data using the fitted scaler.

This ensures that information from the test dataset does not influence preprocessing during model evaluation.

## Machine Learning Model

**Algorithm:** Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to perform classification.

The model predicts whether a hotel reservation is likely to be cancelled or not cancelled based on the booking information.

## Streamlit Web Application

The application contains three main sections:

### 1. Welcome

- Project introduction
- Project objectives
- Overview of the system workflow

### 2. Cancellation Prediction

- Enter hotel reservation details.
- Predict booking cancellation.
- Display cancellation probability.
- Display not-cancelled probability.
- Show the prediction result.

### 3. Reservation Analysis

- Total bookings
- Cancelled bookings
- Not-cancelled bookings
- Cancellation rate
- Hotel type analysis
- Market segment analysis
- Customer type distribution
- Booking cancellation distribution
- Cancellation rate by hotel type
- Monthly booking trends

## Project Structure

```text
Hotel-booking-prediction-and-reservation-analysis-1/
│
├── app.py
├── hotel booking 1.ipynb
├── hotel_bookings.csv
├── hotel_cancellation_model.pkl
├── hotel_encoder.pkl
├── feature_columns.pkl
├── categorical_cols.pkl
├── requirements.txt
└── README.md
```

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/kabisha009-debug/Hotel-booking-prediction-and-reservation-analysis-1.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Hotel-booking-prediction-and-reservation-analysis-1
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your web browser.

## Future Improvements

- Hyperparameter optimization using RandomizedSearchCV.
- Further improvement of classification performance.
- Additional reservation analysis visualizations.
- Improved model performance and user experience.

## Author

**Abisha K**

B.Sc. Computer Science

GitHub: [kabisha009-debug](https://github.com/kabisha009-debug)

---

**Project:** Hotel Booking Cancellation Prediction and Reservation Analysis System
