import streamlit as st
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="Defect Detection", layout="centered")

st.title("⚙️ Aircraft Component Defect Detection (Keras)")
st.write("Upload an image of an aerospace component to check for surface defects (e.g. scratches, bending, color anomalies).")

@st.cache_resource
def load_model_keras(model_name="resnet"):
    model_path = f"saved_models/{model_name}_best.keras"
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model, True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, False
    else:
        return None, False

model_choice = st.sidebar.selectbox("Select Model", ["resnet", "baseline"], index=0)
st.sidebar.markdown("""
**Models:**
- `resnet`: Transfer learning from EfficientNetB0 (Industry Standard)
- `baseline`: Simple 3-layer CNN from scratch
""")

model, is_loaded = load_model_keras(model_choice)

if not is_loaded:
    st.error(f"⚠️ Model weights not found for '{model_choice}'. Please train the model first by running `python train.py --model {model_choice}`.")
else:
    st.success(f"✅ {model_choice.capitalize()} model loaded successfully!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and is_loaded:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
    with col2:
        with st.spinner("Analyzing component..."):

            image_resized = image.resize((224, 224))
            img_array = tf.keras.utils.img_to_array(image_resized)
            img_tensor = tf.expand_dims(img_array, 0)
            

            predictions = model.predict(img_tensor, verbose=0)
            probability_defect = float(predictions[0][0])
            
            st.subheader("Analysis Results:")
            
            if probability_defect >= 0.5:
                st.error("🚨 DEFECT DETECTED")
                st.write("The model identified anomalies indicative of a defect.")
            else:
                st.success("✅ PASSED INSPECTION")
                st.write("The component appears normal.")
                
            st.write("---")
            st.write(f"**Confidence Scores:**")
            st.progress(probability_defect)
            st.write(f"Probability of Defect: {probability_defect:.2%}")
            
            st.info(f"**How it works:** This prediction was made using the Keras {model_choice} architecture. The model was trained specifically to handle class imbalance, focusing on maximizing recall for the rare defect class.")
