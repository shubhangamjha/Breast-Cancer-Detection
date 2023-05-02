import pickle
import streamlit as slt
from streamlit_option_menu import option_menu
import tensorrt
from keras import models
from keras.engine.functional import Functional
import tensorflow as tf



#loading Saved Files


breast_model = pickle.load(open('Breast_Cancer.sav', 'rb'))


#slider bar

with slt.sidebar:
    selected = option_menu('Breast Cancer Prediction',
                           ['Breast Cancer'],
                           icons=['person'],
                           default_index=0)


#Breast Cancer Page:

if selected == "Breast Cancer Prediction":
    #page title

    slt.title("Breast Cancer Prediction Using Ml")

    # getting the input data from the user
    col1, col2, col3, col4, col5 = slt.columns(5)

    with col1:
        Mean_Radius = slt.text_input('mean radius ')

    with col2:
        Mean_Texture = slt.text_input('mean texture')

    with col3:
        Mean_Perimeter = slt.text_input('mean perimeter')

    with col4:
        Mean_Area = slt.text_input('mean area')

    with col5:
        Mean_Smoothness = slt.text_input('mean smoothness')

    with col1:
        Mean_Compactness = slt.text_input('mean compactness')

    with col2:
        Mean_Concavity = slt.text_input('mean concavity')

    with col3:
        Mean_Concave_Points = slt.text_input('mean concave points ')

    with col4:
        Mean_Symmetry = slt.text_input('mean symmetry')

    with col5:
        Mean_Fractional_Dimension = slt.text_input('mean fractal dimension')

    with col1:
        Radius_Error = slt.text_input('radius error')

    with col2:
        Texture_Error = slt.text_input('texture error')

    with col3:
        Perimeter_Error = slt.text_input('perimeter error')

    with col3:
        Area_Error = slt.text_input('area error')

    with col4:
        Smoothness_Error = slt.text_input('smoothness error')

    with col5:
        Compactness_Error = slt.text_input('compactness error')

    with col1:
        Concavity_Error = slt.text_input('concavity error')

    with col2:
        Concave_Point_Error = slt.text_input('concave point error')

    with col3:
        Symmetry_Error = slt.text_input('symmetry error')

    with col4:
        Fractal_Dimension_Error = slt.text_input('fractal dimension error')

    with col5:
        Worst_Radius = slt.text_input('worst radius')


    with col1:
        Worst_Texture = slt.text_input('worst texture')

    with col2:
        Worst_Perimeter = slt.text_input('worst perimeter')

    with col3:
        Worst_Area = slt.text_input('worst area')

    with col4:
        Worst_Smoothness = slt.text_input('worst smoothness')

    with col5:
        Worst_Compactness = slt.text_input('worst compactness')

    with col1:
        Worst_Concavity = slt.text_input('worst concavity')

    with col2:
        Worst_Concave_Points = slt.text_input('worst concave points')

    with col3:
        Worst_Symmetry = slt.text_input('worst symmetry')

    with col4:
        Worst_Fractal_Dimension = slt.text_input('worst fractal dimension')

    with col5:
        Label = slt.text_input('label')


    #code for BC Prediction

    breast_cancer_diagnosis = ''


    #Breast Cancer Prediction Page


    if slt.button('Breast Cancer Test Result'):
        breast_cancer_prediction = breast_model.predict([[Mean_Radius, Mean_Texture, Mean_Perimeter, Mean_Area, Mean_Smoothness, Mean_Compactness, Mean_Concavity, Mean_Concave_Points, Mean_Fractional_Dimension, Radius_Error, Texture_Error, Perimeter_Error, Area_Error, Smoothness_Error, Compactness_Error, Concavity_Error, Concave_Point_Error, Symmetry_Error, Fractal_Dimension_Error, Worst_Radius, Worst_Texture, Worst_Perimeter, Worst_Area, Worst_Smoothness, Worst_Compactness, Worst_Concavity, Worst_Concave_Points, Worst_Symmetry, Worst_Fractal_Dimension, Label]])

        if breast_cancer_prediction[0] == 1:
            """The Person has Breast Cancer"""

        else:
            """The Person is Cancer Free"""

    slt.success(breast_cancer_diagnosis)
