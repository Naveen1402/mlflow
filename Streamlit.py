import pickle
import pandas as pd
import streamlit as st
import Flask


loaded_model = pickle.load(open("PredMaintainence_model.pickle", 'rb'))


def main():
    st.title("Machine Air Temperature")
    #Type = st.number_input("Type")
    ProcessTemperature = st.number_input("ProcessTemperature")
    #RotationalSpeed = st.number_input("RotationalSpeed")
    #Torque = st.number_input("Torque")
    #ToolWear = st.number_input("ToolWear")
    #MachineFailure = st.number_input("MachineFailure")
    #TWF = st.number_input("TWF")
    HDF = st.number_input("HDF")
    #PWF = st.number_input("PWF")
    #OSF= st.number_input("OSF")
    #RNF = st.number_input("RNF")


    cols = [ProcessTemperature, HDF]
    #['Type', 'ProcessTemperature', 'RotationalSpeed','Torque', 'ToolWear','MachineFailure', 'TWF', 'HDF', 'PWF', 'OSF', "RNF"]
    df = pd.DataFrame([cols], columns = ['ProcessTemperature', 'HDF'])
    #df.drop(columns = ['Type', 'RotationalSpeed',
                                         #'Torque', 'ToolWear','MachineFailure', 'TWF', 'PWF', 'OSF', "RNF"], inplace = True)
    #st.write(df)

    if st.button("Predict"):
        prediction = loaded_model.predict(df)
        st.write("Air Temperature for this set of data is:: ")
        st.write(round(prediction[0], 3))



if __name__ == '__main__':
    main()