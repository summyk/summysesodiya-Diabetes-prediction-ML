# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 14:47:21 2023

@author: KIIT
"""

import numpy as np
import pickle

#loading the saved model

loaded_model =pickle.load(open('C:/Users\KIIT/Desktop/dia_proj/trained_model.sav','rb'))


input_data =(4,110,92,0,0,37.6,0.191,30)

#Changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we predicting  for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction =loaded_model.predict(input_data_reshaped)
print(prediction)


if (prediction[0]==0):
  print('The person is not diabatic')
else:
  print('The person is diabatic')