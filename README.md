# Privacy-Preserving-Stroke-Prediction-using-Differential-Privacy
# Privacy-Preserving Stroke Prediction using Differential Privacy

## Overview

This project implements a Privacy-Preserving Machine Learning system in the healthcare domain to predict the risk of stroke while protecting sensitive patient data.

Differential Privacy (DP) is integrated into the model training process by adding Gaussian noise to model parameters. This ensures that individual data points cannot be easily reconstructed or identified, while maintaining acceptable model performance.

The project also analyzes the trade-off between privacy and accuracy using different epsilon (ε) values and visualizes the results.

---

## Objectives

* Protect sensitive healthcare data using Differential Privacy
* Train a machine learning model for stroke prediction
* Analyze the impact of privacy on model accuracy
* Visualize the privacy–utility trade-off
* Provide an interactive interface for predictions

---

## Domain

Healthcare – Stroke Prediction

---

## Dataset

* Name: Stroke Prediction Dataset
* Source: Kaggle
* Description: The dataset contains medical and demographic information used to predict the likelihood of stroke.

### Features:

* Age
* Gender
* Hypertension
* Heart Disease
* Ever Married
* Work Type
* Residence Type
* Average Glucose Level
* BMI
* Smoking Status
* Stroke (Target Variable)

---

## Tech Stack

* Python
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Streamlit

---

## Differential Privacy Implementation

This project uses the Gaussian noise mechanism to ensure privacy.

* Noise is added to model weights after training
* Privacy is controlled using epsilon (ε) values
* Lower epsilon provides stronger privacy but reduces accuracy
* Higher epsilon improves accuracy but weakens privacy guarantees

---

## Key Features

* Baseline model without privacy
* Differential Privacy-enabled model
* Before and after accuracy comparison
* Epsilon versus accuracy visualization
* Interactive prediction interface

---

## Results and Analysis

### Before vs After Privacy

The model accuracy is evaluated before applying Differential Privacy and after applying noise to the model parameters.

### Privacy vs Accuracy Trade-off

* Lower epsilon increases privacy but decreases accuracy
* Higher epsilon decreases privacy but improves accuracy

This demonstrates the trade-off between privacy and model utility.

---

## Conclusion

The project demonstrates that Differential Privacy can effectively protect sensitive data in machine learning models. However, it introduces a trade-off between privacy and performance. Selecting an appropriate epsilon value is important to balance both aspects.

---

## Project Structure

```
dp_stroke_project/
│
├── app.py
├── data/
│   └── healthcare-dataset-stroke-data.csv
│
├── src/
│   ├── preprocess.py
│   ├── dp_model.py
│   ├── train.py
│   └── visualize.py
│
└── requirements.txt
```

---

## How to Run

### Install Dependencies

```
pip install -r requirements.txt
```

### Run the Application

```
streamlit run app.py
```

---

## Output

* Accuracy comparison table
* Before vs After visualization
* Privacy vs Accuracy graph
* Stroke prediction interface

---

## Future Improvements

* Use advanced Differential Privacy frameworks such as TensorFlow Privacy
* Improve model performance using advanced algorithms
* Deploy the application on cloud platforms
* Add authentication and security layers

---

## Author

Gopika Ponnusamy

---

## Acknowledgment

Dataset sourced from Kaggle and implemented for academic purposes in Privacy and Security Analytics.
