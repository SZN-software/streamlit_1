import streamlit as st
import numpy as np
import pickle
import pandas as pd
from pycaret.classification import load_model, predict_model


header = st.container()
dataset = st.container()
# features = st.container()
# model_prediction = st.container()
loaded_model = load_model("my_best_pipline_1")
# print(loaded_model)


# st.cache
# def get_data(filename):
# 	ml_service = pickle.load(open(file_name,'rb'))
# 	return ml_service



with header:
	st.title('Human data')	


with dataset:
	st.header('Model has been Trained on linear Regression')
	st.text('dataset was given by SIR')

	

# with features:
# 	st.header('The features I created')

# 	st.markdown('* **first feature:** Predict Weight based on Height in inches')
	
# age	height_cm	weight_kg	body_fat	diastolic	systolic	gripForce	sit_bend_cm	sit_ups_counts	broad_jump_cm	gender_F	gender_M	class_A	class_B	class_C	class_D
with st.form(key='my_form'):
    age = st.number_input('Age')
    height_cm = st.number_input('height_cm')
    weight_kg = st.number_input('weight_kg')
    body_fat = st.number_input('body_fat')
    diastolic = st.number_input('diastolic')
    systolic = st.number_input('systolic')
    gripForce = st.number_input('gripForce')
    sit_bend_cm = st.number_input('sit_bend_cm')
    sit_ups_counts = st.number_input('sit_ups_counts')
    broad_jump_cm = st.number_input('broad_jump_cm')
    gender_txt = st.text_input('M or F')
    predict = st.form_submit_button('Predict')
    
    if predict:
        # creating data frame for input
        data = dict(age=age,	gender=gender_txt,	height_cm=height_cm,weight_kg=weight_kg,body_fat=body_fat,diastolic=diastolic,
        systolic=systolic,gripForce=gripForce,sit_bend_cm=sit_bend_cm,sit_ups_counts=sit_ups_counts,broad_jump_cm=broad_jump_cm, class_rated='C',)
        df_3 = pd.DataFrame(data, index=[0])   
        # printing data entered     
        st.write("Class",age,height_cm,weight_kg,body_fat,diastolic,systolic,gripForce,sit_bend_cm,sit_ups_counts,broad_jump_cm,gender_txt)
        # model preedict
        df1 = predict_model(loaded_model,df_3)
        # removing one of the column
        df1.drop('class_rated',inplace=True, axis=1)
        st.write(df1)        
        st.header("Result")
        st.write(df1.loc[0, 'Label'])

    