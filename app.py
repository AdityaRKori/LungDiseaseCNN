import streamlit as st\nfrom PIL import Image\nimport numpy as np\nimport tensorflow as tf\nimport matplotlib.pyplot as plt\nimport os\n\nimport logging\n\n# Configure logging\nlogging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')\n\n\n\n# Load the pre-trained model\n# Assuming the model is saved in the current directory\nmodel_path = 'lung_disease_classification_model.h5'\nmodel = tf.keras.models.load_model(model_path)\n\nst.title('Lung Disease Classification')\nst.write('Upload a lung X-ray image for classification.')\nlogging.info('Streamlit app started and ready for image upload.')\n\nuploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])\n\n# Define disease information based on WHO (simplified for example)\ndisease_info = {\n    'Bacterial Pneumonia': 'Bacterial pneumonia is an infection of the lungs caused by bacteria. Symptoms often include cough with phlegm, fever, chills, and difficulty breathing. WHO emphasizes vaccination and proper hygiene for prevention.',\n    'Corona Virus Disease': 'Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus. Most people infected with the virus will experience mild to moderate respiratory illness. WHO provides extensive information on prevention, symptoms, and treatment.',\n    'Normal': 'A normal chest X-ray shows healthy lungs without signs of acute disease or abnormalities.',\n    'Tuberculosis': 'Tuberculosis (TB) is an infectious disease usually caused by Mycobacterium tuberculosis bacteria. TB most commonly affects the lungs. WHO highlights the importance of early diagnosis and treatment for preventing the spread of TB.'\n}\n\nif uploaded_file is not None:\n    # Display the uploaded image\n    image = Image.open(uploaded_file)\n    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)\nlogging.info('Image uploaded and displayed.')\n\n    # Preprocess the image\n    img_array = np.array(image.resize((224, 224))) / 255.0\n    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension\n\n    # Make predictions\n    predictions = model.predict(img_array)\nlogging.info('Predictions made.')\n    probabilities = predictions[0]\n\n    # Get class labels from the model (assuming the model has them or you define them)\n    # If you trained with ImageDataGenerator, you can get them from the generator's class_indices\n    # For this example, we'll use the class labels defined in the evaluation step\n    class_labels = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis']\n\n    # Create a pie chart of the probabilities\n    fig, ax = plt.subplots()\n    ax.pie(probabilities, labels=class_labels, autopct='%1.1f%%', startangle=90)\n    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.\n    st.pyplot(fig)\n\n    # Display the probabilities as text as well\n    st.write("Predicted Probabilities:")\n    for i, label in enumerate(class_labels):\n        st.write(f"{label}: {probabilities[i]:.4f}")\n\n    # Determine the predicted class with the highest probability\n    predicted_class_index = np.argmax(probabilities)\n    predicted_class_label = class_labels[predicted_class_index]\n\n    # Display information about the predicted disease\n    st.subheader(f"Information about {predicted_class_label}:")\n    st.info(disease_info.get(predicted_class_label, "Information not available for this class."))\n\n    # --- SHAP Integration Placeholder ---\n    # Implementing SHAP requires a background dataset and can be computationally intensive.\n    # You would typically calculate SHAP values here and visualize them.\n    # Example (requires shap library and a background dataset):\n    # import shap\n    # # Choose a background dataset (e.g., a few representative images from your training set)\n    # # explainer = shap.DeepExplainer(model, background_dataset)\n    # # shap_values = explainer.shap_values(img_array)\n    # # st.subheader("SHAP Explanations:")\n    # # shap.image_plot(shap_values, -img_array) # Or use other SHAP plotting functions\n    # st.write("SHAP integration can be added here for explainability.")\n    # --- End of SHAP Integration Placeholder ---\n\n\n# Save the modified code back to app.py\nwith open('app.py', 'w') as f:\n    f.write(streamlit_app_code)\n\nprint("app.py modified successfully to include disease information.")\n
    # --- SHAP Integration ---
    if explainer is not None:
        st.subheader("SHAP Explanations:")
        try:
            # Reshape img_array for SHAP (remove batch dimension for plotting if needed, or handle batching)
            # shap.image_plot expects (N, H, W, C) or (N, C, H, W)
            # Our img_array is (1, H, W, C)
            shap_values = explainer.shap_values(img_array)

            # SHAP image_plot expects a list of arrays for multi-output models
            # And the images need to be relative to the background
            # For plotting, display original image as background
            st.write("Feature importance for each class:")
            shap.image_plot(shap_values, -img_array, show=False) # show=False prevents auto-display
            st.pyplot(plt.gcf()) # Display the plot in Streamlit
            plt.close(fig) # Close the figure to free memory

            logging.info('SHAP values calculated and plot displayed.')
        except Exception as e:
            logging.error(f'Error during SHAP calculation or plotting: {e}')
            st.error(f"Could not generate SHAP explanations: {e}")
    else:
        st.info("SHAP explanations are not available because the explainer could not be created.")
    # --- End of SHAP Integration ---
\n
# Choose a small, representative background dataset for SHAP
# Using a batch from the training generator
try:
    background_dataset, _ = next(train_generator)
    # Limit the background dataset size for performance
    background_dataset = background_dataset[:50] # Use a smaller subset

    # Create a SHAP explainer object
    explainer = shap.DeepExplainer(model, background_dataset)
    logging.info('SHAP explainer created successfully.')
except NameError:
    logging.warning('train_generator not found. SHAP explainer could not be created.')
    explainer = None # Set explainer to None if train_generator is not available

\n\nimport shap