import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

filename = "Pickle_RL_Model.pkl"
model = joblib.load(filename)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Glucose = float(request.form['Glucose'])
        Cholesterol = float(request.form['Cholesterol'])
        Hemoglobin = float(request.form['Hemoglobin'])
        Platelets = float(request.form['Platelets'])
        White_Blood_Cells = float(request.form['White Blood Cells'])
        Red_Blood_Cells = float(request.form['Red Blood Cells'])
        Hematocrit = float(request.form['Hematocrit'])
        Mean_Corpuscular_Volume = float(request.form['Mean Corpuscular Volume'])
        Mean_Corpuscular_Hemoglobin = float(request.form['Mean Corpuscular Hemoglobin'])
        Mean_Corpuscular_Hemoglobin_Concentration = float(request.form['Mean Corpuscular Hemoglobin Concentration'])
        Insulin = float(request.form['Insulin'])
        Bmi = float(request.form['BMI'])
        Systolic_Blood_Pressure = float(request.form['Systolic Blood Pressure'])
        Diastolic_Blood_Pressure = float(request.form['Diastolic Blood Pressure'])
        Triglycerides = float(request.form['Triglycerides'])
        HbA1c = float(request.form['HbA1c'])
        LDL_Cholesterol = float(request.form['LDL Cholesterol'])
        HDL_Cholesterol = float(request.form['HDL Cholesterol'])
        Alt = float(request.form['ALT'])
        Ast = float(request.form['AST'])
        Heart_Rate = float(request.form['Heart Rate'])
        Creatinine = float(request.form['Creatinine'])
        Troponin = float(request.form['Troponin'])
        C_reactive_Protein = float(request.form['C-reactive Protein'])

        input_features = [[Glucose,Cholesterol,Hemoglobin,Platelets,White_Blood_Cells,Red_Blood_Cells,Hematocrit,
                           Mean_Corpuscular_Volume,Mean_Corpuscular_Hemoglobin,
                           Mean_Corpuscular_Hemoglobin_Concentration,Insulin,Bmi,Systolic_Blood_Pressure,
                           Diastolic_Blood_Pressure,Triglycerides,HbA1c,LDL_Cholesterol,HDL_Cholesterol,Alt,Ast,
                           Heart_Rate,Creatinine,Troponin,C_reactive_Protein]]

        prediction = model.predict(input_features)

        return render_template('pred.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
