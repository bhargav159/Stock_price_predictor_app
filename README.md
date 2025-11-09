# Stock_price_predictor_app
A Streamlit app for predicting stock prices using LSTM(Long-Short Term Memory ) Deep Learning model 


Execution methods:
  1. Own Device:
         Firstly you can use this app if you have python and other requirements libraries  
  2. Google collab(⚠️⚠️⚠️ In Update):
    
========= Own device ===========

A. You should install Python (old versions only 3.10 is suggested) in your device.

B. Any coding platform like VSCode, Pycharm etc 

**** tip - Create a VIRTUAL ENVIRONMENT for not interfering with your existing files 

-----virtual environment creation -----

    1st method : Execute the command for virtual environment in VSCode terminal(ctrl + `) " python -m venv venvname " 
    
    2nd method :  
        step1 : command pallete " Ctrl + Shift + P " 
        step2 : click on " + create virtual environment "
        step3 : select venv 
        step4 : select python path
    
C. Activate the created python virtual enviromnent by using " venvname\Scripts\activate ".

D. Select the interpreter in VSCode by using  
    step1 : Command Pallete " Ctrl + Shift + P "
    step2 : Enter " Python:Select Interpreter" 
    step3 : Select the python env 

E. Install requirements by Executing the following command " pip install -r requirements.txt "

F. Checking all libraries are working fine 

    method 1: Do " step C" then exec [text](Libraries_func.py) by using " python Libraries_func.py "

    method 2: Terminal Approach  
    Step1 : Open VSCode terminal(ctrl + `)
    Step2 : enter " python --version " displays your python version Ex: Python 3.10.10 
    to ensure python working fine 

    Step3 : 1. Enter " python " to enter python in terminal 
            2. Enter " import numpy as np " 
            3. then " print(np.__version__) " displays numpy version 
            4. enter " exit() " to exit python in terminal 

G. Ensure you have all files including [text](Latest_stock_price_model.keras), [text](Web_App.py) and [text](requirements.txt)

H. To run the web Application Enter " streamlit run C:\path\of\your\file\location\StreamlitApp.py "

I. You can view by automatic browser opening or through localhost 8051 or
   Simply paste http://localhost:8501 in browser(Chrome,Linux,Firefox,Arc etc) 

Thank you for visiting the project I hope it helps
This is just an first version 

🔥🔥🔥 Updates are on the way including features like Stock database , Fixed Bugs , Faster Prediction any many more. 

