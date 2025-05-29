import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
import logging

# Configure page
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .normal { border-left-color: #28a745; }
    .mild { border-left-color: #ffc107; }
    .moderate { border-left-color: #fd7e14; }
    .severe { border-left-color: #dc3545; }
    .proliferative { border-left-color: #6f42c1; }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetModel(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=5):
        super(ResNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Transform for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels and descriptions
class_info = {
    0: {
        "name": "No DR (Normal)",
        "description": "No signs of diabetic retinopathy detected.",
        "color": "#28a745",
        "recommendation": "Continue regular eye checkups and maintain good blood sugar control."
    },
    1: {
        "name": "Mild DR",
        "description": "Early signs of diabetic retinopathy with microaneurysms.",
        "color": "#ffc107",
        "recommendation": "Monitor closely and consult with eye specialist within 6-12 months."
    },
    2: {
        "name": "Moderate DR",
        "description": "More advanced changes with blocked blood vessels.",
        "color": "#fd7e14",
        "recommendation": "Require regular monitoring and possible treatment. Consult specialist within 3-6 months."
    },
    3: {
        "name": "Severe DR",
        "description": "Many blocked blood vessels depriving retina of blood supply.",
        "color": "#dc3545",
        "recommendation": "Immediate medical attention required. High risk of vision loss."
    },
    4: {
        "name": "Proliferative DR",
        "description": "Advanced stage with new blood vessel growth.",
        "color": "#6f42c1",
        "recommendation": "Urgent treatment needed to prevent severe vision loss or blindness."
    }
}

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = ResNetModel('resnet50', 5).to(device)
        
        # Try to load from different possible paths
        model_paths = [
            'best_model.pth',
            'model/best_model.pth',
            'checkpoints/best_model.pth'
        ]
        
        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                model_loaded = True
                st.success(f"✅ Model loaded successfully from {path}")
                return model
        
        if not model_loaded:
            st.warning("⚠️ No trained model file found. Using pretrained ResNet50.")
            model.eval()
            return model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def predict_image(image, model):
    """Predict diabetic retinopathy for a single image"""
    try:
        if model is None:
            return None, None, "Model not loaded"
        
        # Process image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted_class = torch.max(outputs, 1)
        
        predicted_class = predicted_class.item()
        
        return predicted_class, probabilities, None
        
    except Exception as e:
        return None, None, str(e)

def create_probability_chart(probabilities):
    """Create an interactive probability chart using Plotly"""
    probs = probabilities[0].cpu().numpy()
    classes = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1']
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probs,
            marker_color=colors,
            text=[f'{p*100:.1f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="DR Severity Level",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig

def create_gauge_chart(confidence):
    """Create a confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level (%)"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ Diabetic Retinopathy Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About This App")
        st.write("""
        This application uses a deep learning model (ResNet50) to detect and classify diabetic retinopathy in retinal images.
        
        **Severity Levels:**
        - 🟢 **Normal**: No DR detected
        - 🟡 **Mild**: Early stage DR
        - 🟠 **Moderate**: Moderate DR
        - 🔴 **Severe**: Advanced DR
        - 🟣 **Proliferative**: Most severe stage
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for educational/research purposes only. 
        Always consult with qualified healthcare professionals for medical diagnosis and treatment.
        """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please check if the model file exists.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Retinal Image")
        uploaded_file = st.file_uploader(
            "Choose a retinal image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a clear retinal photograph for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add prediction button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    predicted_class, probabilities, error = predict_image(image, model)
                
                if error:
                    st.error(f"❌ Prediction error: {error}")
                else:
                    # Store results in session state
                    st.session_state.prediction_results = {
                        'predicted_class': predicted_class,
                        'probabilities': probabilities,
                        'confidence': probabilities[0][predicted_class].item()
                    }
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            predicted_class = results['predicted_class']
            probabilities = results['probabilities']
            confidence = results['confidence']
            
            # Display prediction
            class_name = class_info[predicted_class]['name']
            class_color = class_info[predicted_class]['color']
            
            st.markdown(f"""
            <div class="prediction-box {class_name.lower().replace(' ', '-').replace('(', '').replace(')', '')}">
                <h3>🎯 Prediction: {class_name}</h3>
                <p><strong>Confidence:</strong> {confidence*100:.1f}%</p>
                <p><strong>Description:</strong> {class_info[predicted_class]['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display recommendation
            st.info(f"💡 **Recommendation:** {class_info[predicted_class]['recommendation']}")
            
            # Show detailed metrics
            st.subheader("📈 Detailed Analysis")
            
            # Create two columns for charts
            chart_col1, chart_col2 = st.columns([2, 1])
            
            with chart_col1:
                # Probability chart
                prob_fig = create_probability_chart(probabilities)
                st.plotly_chart(prob_fig, use_container_width=True)
            
            with chart_col2:
                # Confidence gauge
                gauge_fig = create_gauge_chart(confidence)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Probability table
            st.subheader("📋 Probability Breakdown")
            prob_data = []
            probs = probabilities[0].cpu().numpy()
            
            for i, (class_id, info) in enumerate(class_info.items()):
                prob_data.append({
                    'Severity Level': info['name'],
                    'Probability': f"{probs[i]*100:.2f}%",
                    'Confidence Bar': probs[i]
                })
            
            df = pd.DataFrame(prob_data)
            st.dataframe(
                df,
                column_config={
                    "Confidence Bar": st.column_config.ProgressColumn(
                        "Confidence",
                        help="Prediction confidence",
                        min_value=0,
                        max_value=1,
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Export results
            st.subheader("💾 Export Results")
            result_text = f"""
Diabetic Retinopathy Analysis Report
===================================
Prediction: {class_name}
Confidence: {confidence*100:.1f}%
Description: {class_info[predicted_class]['description']}
Recommendation: {class_info[predicted_class]['recommendation']}

Detailed Probabilities:
{chr(10).join([f"- {info['name']}: {probs[i]*100:.2f}%" for i, info in class_info.items()])}

Disclaimer: This analysis is for educational/research purposes only.
Always consult healthcare professionals for medical advice.
            """
            
            st.download_button(
                label="📄 Download Report",
                data=result_text,
                file_name=f"dr_analysis_{uploaded_file.name if uploaded_file else 'report'}.txt",
                mime="text/plain"
            )
        
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏥 Diabetic Retinopathy Detection System | Built with Streamlit & PyTorch</p>
        <p>⚕️ For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()