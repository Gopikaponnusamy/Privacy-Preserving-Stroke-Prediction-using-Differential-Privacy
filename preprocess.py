import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df.drop(['id'], axis=1, inplace=True)
    df.replace("Unknown", pd.NA, inplace=True)

    cat_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
    num_cols = ['age','hypertension','heart_disease','avg_glucose_level','bmi']

    df[num_cols] = SimpleImputer(strategy='mean').fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])

    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop('stroke', axis=1)
    y = df['stroke']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, le_dict