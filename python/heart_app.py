Last login: Wed Nov 19 20:54:22 on ttys002
ahmedmohamedshams@MacBook-Pro-Ahmed ~ % return render_template('index.html', result=result)

zsh: unknown file attribute: i
ahmedmohamedshams@MacBook-Pro-Ahmed ~ % 
ahmedmohamedshams@MacBook-Pro-Ahmed ~ % return render_template('heart_index.html', result=result)

zsh: unknown file attribute: h
ahmedmohamedshams@MacBook-Pro-Ahmed ~ % cd ~/Desktop/heart_project
source venv/bin/activate
python heart_app.py

  File "/Users/ahmedmohamedshams/Desktop/heart_project/heart_app.py", line 1
    Last login: Wed Nov 19 20:51:37 on ttys002
         ^
SyntaxError: invalid syntax
(venv) ahmedmohamedshams@MacBook-Pro-Ahmed heart_project % nano ~/Desktop/heart_project/heart_app.py




  File: /Users/ahmedmohamedshams/Desktop/heart_project/heart_app.py   Modified  

        chest_pain_encoded = [1 if chest_pain == cp else 0 for cp in chest_pain$
        resting_ecg_encoded = [1 if resting_ecg == ecg else 0 for ecg in restin$
        st_slope_encoded = [1 if st_slope == slope else 0 for slope in st_slope$
        
        X_new = np.array([[age, sex, resting_bp, cholesterol, fasting_bs, max_h$
                          chest_pain_encoded + resting_ecg_encoded + st_slope_e$
        
        X_new_scaled = scaler.transform(X_new)
        prediction = model.predict(X_new_scaled)

        result = "Risk of Heart Disease" if prediction[0] == 1 else "No Risk De$

    return render_template('heart_index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
Last login: Wed Nov 19 20:51:37 on ttys002
ahmedmohamedshams@MacBook-Pro-Ahmed ~ % nano app.py
        

^G Get Help  ^O WriteOut  ^R Read File ^Y Prev Pg   ^K Cut Text  ^C Cur Pos   
^X Exit      ^J Justify   ^W Where is  ^V Next Pg   ^U UnCut Text^T To Spell  
