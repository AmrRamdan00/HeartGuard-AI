
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('heart.csv')
print("Shape of data:", data.shape)
print("\nInfo:")
print(data.info())
print("\nDescribe:")
print(data.describe())
print("\nMissing values:")
print(data.isnull().sum())
cols_to_convert = ['FastingBS', 'MaxHR', 'Oldpeak']
for col in cols_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()
binary_cols = ['Sex', 'ExerciseAngina', 'FastingBS']
le = LabelEncoder()
for col in binary_cols:
    data[col] = le.fit_transform(data[col])
multi_cols = ['ChestPainType', 'RestingECG', 'ST_Slope']
data = pd.get_dummies(data, columns=multi_cols)
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    verbose=1,
    callbacks=[early_stop]
)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\n=== Test Set Evaluation ===")
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)
y_pred_classes = (model.predict(X_test) >= 0.5).astype(int)
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_classes))
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred_classes))
def get_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Error! Please enter a valid integer.")
def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Error! Please enter a valid float.")
def get_choice(prompt, choices):
    choices_str = "/".join(choices)
    while True:
        value = input(f"{prompt} ({choices_str}): ").upper()
        if value in [c.upper() for c in choices]:
            return value
        print(f"Error! Please choose one of: {choices_str}")
print("\n=== Heart Disease Prediction for New Person ===")
age = get_int("Enter Age: ")
sex = get_choice("Enter Sex", ["M", "F"])
resting_bp = get_int("Enter Resting Blood Pressure: ")
cholesterol = get_int("Enter Cholesterol: ")
fasting_bs = get_choice("Enter Fasting Blood Sugar", ["0", "1"])
max_hr = get_int("Enter Max Heart Rate: ")
exercise_angina = get_choice("Exercise Angina", ["Y", "N"])
oldpeak = get_float("Enter Oldpeak: ")
chest_pain = get_choice("Enter Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_ecg = get_choice("Enter Resting ECG type", ["Normal", "LVH", "ST"])
st_slope = get_choice("Enter ST Slope type", ["Up", "Flat", "Down"])
sex = 1 if sex.upper() == 'M' else 0
exercise_angina = 1 if exercise_angina.upper() == 'Y' else 0
fasting_bs = int(fasting_bs)
chest_pain_cols = ['ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_ASY', 'ChestPainType_TA']
resting_ecg_cols = ['RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST']
st_slope_cols = ['ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']
chest_pain_encoded = [1 if chest_pain.upper() == cp.split('_')[1] else 0 for cp in chest_pain_cols]
resting_ecg_encoded = [1 if resting_ecg.upper() == ecg.split('_')[1] else 0 for ecg in resting_ecg_cols]
st_slope_encoded = [1 if st_slope.upper() == slope.split('_')[1] else 0 for slope in st_slope_cols]
X_new = np.array([[age, sex, resting_bp, cholesterol, fasting_bs, max_hr, exercise_angina, oldpeak] +
                  chest_pain_encoded + resting_ecg_encoded + st_slope_encoded])
X_new_scaled = scaler.transform(X_new)
prediction_prob = model.predict(X_new_scaled)[0][0]
if prediction_prob >= 0.7:
    risk_category = "High Risk"
elif prediction_prob >= 0.5:
    risk_category = "Medium Risk"
else:
    risk_category = "Low Risk"
print(f"\nPredicted Risk Probability: {prediction_prob*100:.2f}%")
print("Risk Category:", risk_category)
print("\n=== Health Recommendations ===")
if risk_category == "High Risk":
    print("- Exercise regularly")
    print("- Monitor blood pressure and cholesterol")
    print("- Consult a doctor immediately")
elif risk_category == "Medium Risk":
    print("- Improve lifestyle (exercise / healthy diet)")
    print("- Monitor cholesterol and blood pressure periodically")
else:
    print("- Maintain a healthy lifestyle")
    print("- Regular checkups for prevention")
plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1,3,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
pred_probs_test = model.predict(X_test)
risk_labels_test = []
for prob in pred_probs_test:
    if prob >= 0.7:
        risk_labels_test.append("High Risk")
    elif prob >= 0.5:
        risk_labels_test.append("Medium Risk")
    else:
        risk_labels_test.append("Low Risk")
risk_labels_test.append(risk_category)
plt.subplot(1,3,3)
sns.countplot(x=risk_labels_test)
plt.title("Risk Category Distribution (Test Set + New Person)")
plt.tight_layout()
plt.show()






