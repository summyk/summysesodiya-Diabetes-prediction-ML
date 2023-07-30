# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 16:55:28 2023

@author: KIIT
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model

loaded_model =pickle.load(open('C:/Users/KIIT/Desktop/dia_proj/trained_model.sav','rb'))

# creating function for prediction

def diabetes_prediction(input_data):
    
  

    #Changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we predicting  for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction =loaded_model.predict(input_data_reshaped)
    print(prediction)


    if (prediction[0]==0):
      return 'The person is not diabatic'
    else:
      return 'The person is diabatic'
  
    

def main():
    
    
    # giving a title
    st.title('Diabatic prediction Web App')
    
    
    # getting the input data from user
   
    
    
    Pregnancies = st.text_input('Number of pregancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood pressure level')
    SkinThickness = st.text_input('skin thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree value')
    Age = st.text_input('Age value')
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    
    if st.button('Diabaties test result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
    