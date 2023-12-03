import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Function to predict selling price
def predict_selling_price(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

def main():
    st.title('Car Selling Price Prediction')

    # Sidebar with input fields
    st.sidebar.header('Enter Car Details')
    model = st.sidebar.selectbox('Model', options=['Model1', 'Model2', 'Model3'])  # Add your model choices
    kilometers_driven = st.sidebar.number_input('Kilometers Driven')
    year = st.sidebar.number_input('Year')
    owner = st.sidebar.selectbox('Owner', options=['First Owner', 'Second Owner', 'Third Owner'])  # Add owner options
    fuel_type = st.sidebar.selectbox('Fuel Type', options=['Petrol', 'Diesel', 'CNG'])  # Add fuel type options
    transmission = st.sidebar.selectbox('Transmission', options=['Manual', 'Automatic'])  # Add transmission options
    car_condition = st.sidebar.number_input('Car Condition')  # Adjust the range as needed
    current_price = st.sidebar.number_input('Current Price')

    # Prepare features for prediction
    features = [model, kilometers_driven, year, owner, fuel_type, transmission, car_condition, current_price]

    # Make prediction
    if st.sidebar.button('Predict'):
        prediction = predict_selling_price(features)
        st.success(f'Predicted Selling Price: ${prediction:.2f}')

if __name__ == "__main__":
    main()
