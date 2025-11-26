Last login: Wed Nov 19 12:19:36 on ttys003
ahmedmohamedshams@MacBook-Pro-Ahmed ~ % cd path/to/neural_project/heart_project

cd: no such file or directory: path/to/neural_project/heart_project
ahmedmohamedshams@MacBook-Pro-Ahmed ~ % cd Desktop/neural_project\ ١٥.٥٨.٠٠/heart_project

ahmedmohamedshams@MacBook-Pro-Ahmed heart_project % nano app.py
















  UW PICO 5.09                      File: app.py                      Modified  

        chest_pain_encoded = [1 if chest_pain == cp else 0 for cp in chest_pain$
        resting_ecg_encoded = [1 if resting_ecg == ecg else 0 for ecg in restin$
        st_slope_encoded = [1 if st_slope == slope else 0 for slope in st_slope$
        
        X_new = np.array([[age, sex, resting_bp, cholesterol, fasting_bs, max_h$
                           exercise_angina, oldpeak] + chest_pain_encoded +
                          resting_ecg_encoded + st_slope_encoded])
        
        X_new_scaled = scaler.transform(X_new)
        prediction = model.predict(X_new_scaled)
        result = "Risk of Heart Disease" if prediction[0]==1 else "No Risk Dete$
        return f"<h2>{result}</h2>"
        
    return render_template('index.html')
        
if __name__ == '__main__': 
    app.run(debug=True)   
        
        

^G Get Help  ^O WriteOut  ^R Read File ^Y Prev Pg   ^K Cut Text  ^C Cur Pos   
^X Exit      ^J Justify   ^W Where is  ^V Next Pg   ^U UnCut Text^T To Spell  
