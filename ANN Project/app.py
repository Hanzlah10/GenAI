# import streamlit as st
# import tensorflow as tf
# import pandas as pd
# import pickle

# # Load the trained model
# model = tf.keras.models.load_model('model.h5')

# # Load the encoders and scaler
# with open('label_encoder_gender.pkl','rb') as file:
#     label_encoder_gender = pickle.load(file)

# with open('onehotencoder_geography.pkl','rb') as file:
#     onehot_encoder_geo = pickle.load(file)

# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)

# # Page configuration
# st.set_page_config(page_title="Churn Prediction AI", layout="centered")

# # Custom CSS styling
# st.markdown("""
#     <style>
#     .main {
#         background-color: #F5F5F5;
#     }
#     h1 {
#         color: #2F4F4F;
#         text-align: center;
#     }
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 24px;
#     }
#     .stNumberInput, .stSelectbox, .stSlider {
#         background-color: white;
#         border-radius: 5px;
#         padding: 10px;
#     }
#     .prediction-card {
#         background-color: white;
#         border-radius: 10px;
#         padding: 20px;
#         margin: 20px 0;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Header Section
# st.title("🏦 Customer Churn Prediction AI")
# st.markdown("""
#     Predict customer churn probability using advanced artificial neural networks. 
#     This tool helps businesses identify at-risk customers and take proactive retention measures.
#     """)

# # Input Section
# with st.form("customer_info"):
#     st.header("📋 Customer Information")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
#         age = st.slider('Age', 18, 92, 30, help="Customer's age in years")
#         balance = st.number_input('Balance', value=0.0, format="%.2f", help="Account balance in USD")
#         credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
        
#     with col2:
#         gender = st.selectbox('Gender', label_encoder_gender.classes_)
#         tenure = st.slider('Tenure (Years)', 0, 10, 2, help="Number of years as customer")
#         estimated_salary = st.number_input('Estimated Salary', value=50000.0, format="%.2f")
#         num_of_products = st.slider('Number of Products', 1, 4, 1)
    
#     st.subheader("Additional Information")
#     col3, col4 = st.columns(2)
#     with col3:
#         has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
#     with col4:
#         is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    
#     submitted = st.form_submit_button("Predict Churn Probability")

# # Prediction Section
# if submitted:
#     # Prepare the input data
#     input_data = pd.DataFrame({
#         'CreditScore': [credit_score],
#         'Gender': [label_encoder_gender.transform([gender])[0]],
#         'Age': [age],
#         'Tenure': [tenure],
#         'Balance': [balance],
#         'NumOfProducts': [num_of_products],
#         'HasCrCard': [has_cr_card],
#         'IsActiveMember': [is_active_member],
#         'EstimatedSalary': [estimated_salary]
#     })

#     # One-hot encode 'Geography'
#     geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
#     geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#     # Combine one-hot encoded columns with input data
#     input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#     # Scale the input data
#     input_data_scaled = scaler.transform(input_data)

#     # Predict churn
#     prediction = model.predict(input_data_scaled)
#     prediction_proba = prediction[0][0]
    
#     # Display results
#     st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
#     st.subheader("📊 Prediction Results")
    
#     # Create a progress bar visualization
#     progress_value = prediction_proba * 100
#     st.progress(int(progress_value))
    
#     col_result, col_prob = st.columns(2)
#     with col_result:
#         st.markdown(f"**Churn Probability:** {prediction_proba:.2%}")
#     with col_prob:
#         if prediction_proba > 0.5:
#             st.error('High Risk: Likely to Churn 🚨')
#         else:
#             st.success('Low Risk: Likely to Stay ✅')
    
#     # Add explanation
#     st.markdown("""
#         **Interpretation:**  
#         - Probability below 50%: Customer is likely to stay  
#         - Probability above 50%: Customer is at risk of churning  
#         """)
    
#     st.markdown("</div>", unsafe_allow_html=True)

# # Footer
# st.markdown("---")
# st.markdown("© 2025 Customer Retention AI | Developed by Hanzala")
import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

@st.cache_resource
def load_encoders():
    with open('label_encoder_gender.pkl','rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('onehotencoder_geography.pkl','rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return label_encoder_gender, onehot_encoder_geo, scaler

# Load resources
model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_encoders()

# Header
st.title("🔄 Customer Churn Prediction")
st.markdown("---")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Customer Demographics")
    geography = st.selectbox('📍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 30)
    tenure = st.slider('⏳ Tenure (years)', 0, 10, 2)

with col2:
    st.subheader("💰 Financial Information")
    balance = st.number_input('💳 Balance ($)', min_value=0.0, value=0.0, step=1000.0)
    credit_score = st.number_input('📊 Credit Score', min_value=300, max_value=850, value=650)
    estimated_salary = st.number_input('💵 Estimated Salary ($)', min_value=0.0, value=50000.0, step=5000.0)
    num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)

# Create two columns for binary inputs
col3, col4 = st.columns(2)

with col3:
    has_cr_card = st.selectbox('💳 Has Credit Card', ['Yes', 'No']) == 'Yes'

with col4:
    is_active_member = st.selectbox('✅ Is Active Member', ['Yes', 'No']) == 'Yes'

# Add a predict button
if st.button('🔍 Predict Churn Probability'):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [int(has_cr_card)],
        'IsActiveMember': [int(is_active_member)],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine and scale data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction_proba * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    # Display results
    st.plotly_chart(fig, use_container_width=True)
    
    # Display prediction message
    if prediction_proba > 0.5:
        st.error("⚠️ High Risk: The customer is likely to churn!")
        st.markdown("""
            ### Recommendations:
            - Consider reaching out to the customer proactively
            - Review their product portfolio
            - Offer personalized retention incentives
        """)
    else:
        st.success("✅ Low Risk: The customer is likely to stay!")
        st.markdown("""
            ### Recommendations:
            - Continue monitoring customer satisfaction
            - Look for opportunities to increase engagement
            - Consider cross-selling additional products
        """)

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and TensorFlow</p>
    </div>
""", unsafe_allow_html=True)