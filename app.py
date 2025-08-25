
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
# Removed shap import
from tensorflow.keras.models import load_model

# Define image dimensions and class labels (these were defined in the notebook)
target_size = (224, 224)
class_labels = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis']

# Load the trained model
# Assuming the model is saved as 'lung_disease_model.h5'
model = load_model('lung_disease_model.h5')

# Removed SHAP explainer instantiation

# Set the title of the Streamlit application
st.title('Lung Disease Prediction and Explanation')

# Add a file uploader widget for image upload
uploaded_file = st.file_uploader("Upload a chest X-ray image...", type=['jpg', 'jpeg', 'png'])

# Add a text input field for patient information
patient_info = st.text_input("Enter Patient Information (Optional)")

# Include a button to trigger the prediction process
if st.button('Predict and Explain'):
    # 1. Check if an image was uploaded
    if uploaded_file is None:
        st.warning("Please upload an image before predicting.")
    else:
        # Display the uploaded image
        st.subheader("Uploaded Image:")
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded X-ray Image', use_column_width=True)

        # Display patient information if entered
        if patient_info:
            st.subheader("Patient Information:")
            st.write(patient_info)

        # 2. Read the image file into a format suitable for the model
        image = image.convert('RGB') # Ensure image is in RGB format

        # 3. Preprocess the image
        # Resize the image
        image = image.resize(target_size)
        # Convert image to numpy array and normalize
        image_array = np.array(image) / 255.0
        # Expand dimensions to match model input shape (batch size, height, width, channels)
        image_array = np.expand_dims(image_array, axis=0)

        # 4. Use the loaded model to make a prediction
        st.subheader("Prediction Results:")
        st.write("Making prediction...")
        predictions = model.predict(image_array)
        # Get the predicted class index and probability
        predicted_class_index = np.argmax(predictions)
        predicted_probability = predictions[0][predicted_class_index]
        predicted_class_label = class_labels[predicted_class_index]

        st.write(f"Predicted Class: **{predicted_class_label}**")
        st.write(f"Predicted Probability: **{predicted_probability:.4f}**")

        # 5. Handle uncertain predictions with a pie chart
        uncertainty_threshold = 0.7 # Define your uncertainty threshold
        if predicted_probability < uncertainty_threshold:
            st.warning("Prediction is uncertain. Probability distribution:")
            fig, ax = plt.subplots()
            ax.pie(predictions[0], labels=class_labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)
            plt.close(fig) # Close the figure to prevent it from displaying automatically later

        # Removed SHAP explanation section

        # 7. Include WHO guidelines
        st.subheader("WHO Care and Safety Guidelines:")

        # General guidelines applicable to all lung diseases
        st.markdown("""
        **General Guidelines:**
        * **Practice good respiratory hygiene:** Cover your mouth and nose with your elbow or a tissue when you cough or sneeze.
        * **Wash your hands frequently:** Use soap and water or an alcohol-based hand rub.
        * **Avoid close contact** with people who are sick.
        * **Stay home when you are sick.**
        * **Follow recommended vaccination schedules.**
        * **Seek medical care** if you have symptoms of lung disease.
        * **Follow your doctor's advice** for managing existing lung conditions.
        """)

        # Specific guidelines based on predicted diagnosis
        if predicted_class_label == 'Bacterial Pneumonia':
            st.markdown("""
            **Specific Guidelines for Bacterial Pneumonia:**
            * **Complete the full course of antibiotics** prescribed by your doctor, even if you start feeling better.
            * **Get plenty of rest.**
            * **Drink plenty of fluids.**
            * **Avoid smoking and exposure to secondhand smoke.**
            """)
        elif predicted_class_label == 'Corona Virus Disease':
            st.markdown("""
            **Specific Guidelines for Corona Virus Disease (COVID-19):**
            * **Isolate yourself** to prevent spreading the virus.
            * **Wear a mask** if you must be around others.
            * **Monitor your symptoms** and seek medical attention if they worsen.
            * **Follow local public health guidelines** regarding testing and isolation.
            """)
        elif predicted_class_label == 'Tuberculosis':
            st.markdown("""
            **Specific Guidelines for Tuberculosis (TB):**
            * **Complete the full course of prescribed medication** as directed by your doctor. This is crucial to cure TB and prevent drug resistance.
            * **Ensure proper ventilation** in your living spaces.
            * **Cover your mouth and nose** when coughing or sneezing, especially during the initial infectious period.
            * **Attend all scheduled medical appointments.**
            """)
        elif predicted_class_label == 'Normal':
             st.markdown("""
            **Specific Guidelines for Normal Lung Health:**
            * **Maintain a healthy lifestyle** with regular exercise and a balanced diet.
            * **Avoid smoking and air pollution.**
            * **Get vaccinated** against respiratory infections like influenza and pneumonia as recommended.
            * **If you experience new or worsening respiratory symptoms,** consult a healthcare professional.
            """)


else:
    # This block is executed when the button is not clicked
    st.info("Upload an image and click 'Predict and Explain' to see the results and guidelines.")
