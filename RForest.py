import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import gradio as gr
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('train.csv')

# Columns to exclude
exclude_columns = ['Patient_Id', 'Patient_First_Name', 'Family_Name', 'Father\'s_name',
                   'Location_of_Institute', 'Status', 'Test_1', 'Test_2', 'Test_3',
                   'Test_4', 'Test_5', 'Parental_consent', 'Follow-up',
                   'H/O_radiation_exposure_(x-ray)', 'Folic_acid_details_(peri-conceptional)',
                   'Place_of_birth', 'Institute_Name']

data = data.drop(columns=exclude_columns)

# Separate features and targets
X = data.drop(['Genetic_Disorder', 'Disorder_Subclass'], axis=1)
y_disorder = data['Genetic_Disorder']
y_subclass = data['Disorder_Subclass']

# Drop rows with missing values
df = pd.concat([X, y_disorder, y_subclass], axis=1).dropna()
X = df.drop(['Genetic_Disorder', 'Disorder_Subclass'], axis=1)
y_disorder = df['Genetic_Disorder']
y_subclass = df['Disorder_Subclass']

# Encode features
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode targets
le_disorder = LabelEncoder()
le_subclass = LabelEncoder()
y_disorder = le_disorder.fit_transform(y_disorder)
y_subclass = le_subclass.fit_transform(y_subclass)

# Train models with Random Forest
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_subclass, test_size=0.2, random_state=42)
model_subclass = RandomForestClassifier(class_weight='balanced', random_state=42)
model_subclass.fit(X_train_s, y_train_s)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_disorder, test_size=0.2, random_state=42)
model_disorder = RandomForestClassifier(class_weight='balanced', random_state=42)
model_disorder.fit(X_train_d, y_train_d)

# Predictions and accuracy calculation
subclass_train_pred = model_subclass.predict(X_train_s)
subclass_test_pred = model_subclass.predict(X_test_s)

disorder_train_pred = model_disorder.predict(X_train_d)
disorder_test_pred = model_disorder.predict(X_test_d)

# Calculate and log accuracy
train_accuracy_subclass = accuracy_score(y_train_s, subclass_train_pred)
test_accuracy_subclass = accuracy_score(y_test_s, subclass_test_pred)

train_accuracy_disorder = accuracy_score(y_train_d, disorder_train_pred)
test_accuracy_disorder = accuracy_score(y_test_d, disorder_test_pred)

print(f"Subclass Model - Train Accuracy: {train_accuracy_subclass:.2f}, Test Accuracy: {test_accuracy_subclass:.2f}")
print(f"Disorder Model - Train Accuracy: {train_accuracy_disorder:.2f}, Test Accuracy: {test_accuracy_disorder:.2f}")

# Prediction function for Gradio interface
def predict_disorder(*inputs):
    input_dict = dict(zip(X.columns, inputs))
    input_df = pd.DataFrame([input_dict])

    for col in input_df.columns:
        if col in label_encoders:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except ValueError:
                return f"Invalid input for {col}. Must be one of: {label_encoders[col].classes_.tolist()}"

    subclass_pred = model_subclass.predict(input_df)[0]
    disorder_pred = model_disorder.predict(input_df)[0]

    return f"Predicted Subclass: {le_subclass.inverse_transform([subclass_pred])[0]}\nPredicted Genetic Disorder: {le_disorder.inverse_transform([disorder_pred])[0]}"

# Gradio inputs
gr_inputs = []
for col in X.columns:
    choices = sorted(data[col].dropna().unique().astype(str).tolist())
    gr_inputs.append(gr.Dropdown(choices=choices, label=col))

# Launch Gradio interface
gr.Interface(
    fn=predict_disorder,
    inputs=gr_inputs,
    outputs="text",
    title="Genetic Disorder Prediction",
    description="Fill in the features to predict subclass and disorder."
).launch()
