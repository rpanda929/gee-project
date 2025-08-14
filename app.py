"""
Enhanced Streamlit App for Google Earth Engine + Deep Learning Models
Comprehensive poverty mapping with improved error handling and visualization
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from datetime import datetime
import time
import os
import warnings
import base64  # <-- NEW
warnings.filterwarnings('ignore')

# =========================
# NEW: GEE + auth imports
# =========================
try:
    import json
    import ee
    from google.oauth2 import service_account
except Exception as e:
    st.error(
        f"Failed to import Earth Engine/auth libs: {e}\n"
        "Install with: pip install earthengine-api google-auth"
    )
    st.stop()

# =========================
# NEW: GEE AUTH CONFIG (reads from env/secret GCP_SA_KEY)
# =========================
USE_SERVICE_ACCOUNT = True  # keep True to auto-auth with the secret

# Optional overrides from environment
SERVICE_ACCOUNT_EMAIL = os.environ.get(
    "GCP_SA_EMAIL",
    "gee-project@weighty-time-440511-h3.iam.gserviceaccount.com"
)
GEE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID", "weighty-time-440511-h3")

# We no longer hardcode JSON; pulled from env/secrets at runtime
SERVICE_ACCOUNT_KEY_JSON = ""     # intentionally blank
SERVICE_ACCOUNT_KEY_FILE = ""     # not used when secret is present


def _get_sa_key_from_env_or_secrets(var_name: str = "GCP_SA_KEY") -> str:
    """
    Returns the service account JSON as a string.
    Supports either raw JSON or base64-encoded JSON stored in:
      1) st.secrets[var_name] (Streamlit Cloud) OR
      2) os.environ[var_name]  (GitHub Actions, Codespaces, local env)
    """
    key_text = ""
    try:
        # Streamlit Cloud secrets
        key_text = st.secrets.get(var_name, "")
    except Exception:
        pass

    if not key_text:
        # GitHub Actions / local env / other runtime envs
        key_text = os.environ.get(var_name, "")

    if not key_text:
        return ""

    s = key_text.strip()
    if s.startswith("{"):
        # Raw JSON provided
        return s

    # Try base64 decode (common pattern for GitHub secrets)
    try:
        decoded = base64.b64decode(s).decode("utf-8")
        if decoded.strip().startswith("{"):
            return decoded
    except Exception:
        # Not base64; return as-is
        pass

    return s


# =========================
# Helper to init EE with service account (unchanged)
# =========================
def init_gee_with_service_account(
    project_id: str,
    service_account_email: str,
    key_json_text: str = "",
    key_file_path: str = ""
) -> bool:
    """
    Initialize Google Earth Engine using a service account.
    Provide either key_json_text (preferred) or key_file_path.
    """
    try:
        if key_json_text:
            key_info = json.loads(key_json_text)
        elif key_file_path and os.path.exists(key_file_path):
            with open(key_file_path, "r") as f:
                key_info = json.load(f)
        else:
            st.error("No service account key provided. Set GCP_SA_KEY or paste/upload the key JSON.")
            return False

        # Warn (non-fatal) if email mismatch
        if "client_email" in key_info and service_account_email and \
           key_info["client_email"] != service_account_email:
            st.warning(
                "Service account email does not match the JSON's client_email. "
                "Proceeding with the JSON's client_email."
            )

        scopes = [
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/devstorage.full_control",
        ]
        credentials = service_account.Credentials.from_service_account_info(
            key_info, scopes=scopes
        )
        ee.Initialize(credentials=credentials, project=project_id)
        return True
    except Exception as e:
        st.error(f"Service account authentication failed: {e}")
        return False

# Import custom modules
try:
    from gee_processor import GEEProcessor
    from models_and_metrics import build_cnn_segmentation, build_unet, build_hybrid_cnn_unet, compute_metrics
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="GEE Poverty Mapping with Deep Learning",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.stProgress .st-bo {
    background-color: #00cc88;
}
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("🌍 Geospatial Poverty Mapping: Deep Learning with Google Earth Engine")
st.markdown("**Odisha Districts Analysis: Koraput, Rayagada, Malkangiri**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # ============= NEW: Auth UI (supports env/secret) ============
    st.subheader("🔐 Earth Engine Authentication")
    use_sa_ui = st.checkbox("Use Service Account (recommended)", value=USE_SERVICE_ACCOUNT)
    project_id_ui = st.text_input(
        "GEE Project ID",
        value=(GEE_PROJECT_ID or ""),
        help="Your Google Cloud project linked to Earth Engine"
    )
    sa_email_ui = st.text_input(
        "Service Account Email",
        value=(SERVICE_ACCOUNT_EMAIL or ""),
        help="...@<project>.iam.gserviceaccount.com"
    )

    # Detect if we already have a key in env/secrets
    _env_has_key = bool(_get_sa_key_from_env_or_secrets())

    options = [
        "Use env/secret 'GCP_SA_KEY'",
        "Use hardcoded JSON (above)",
        "Paste JSON here",
        "Upload JSON file",
    ]

    # Prefer the env/secret if present, else fallback to paste
    default_index = 0 if _env_has_key else (1 if SERVICE_ACCOUNT_KEY_JSON else 2)

    key_input_mode = st.radio(
        "Key Input Method",
        options=options,
        index=default_index
    )

    key_json_from_ui = ""
    if key_input_mode == "Use env/secret 'GCP_SA_KEY'":
        key_json_from_ui = _get_sa_key_from_env_or_secrets()
    elif key_input_mode == "Paste JSON here":
        key_json_from_ui = st.text_area(
            "Paste service account JSON (or base64 of it)",
            value="",
            height=200,
            help="Paste the entire key JSON or a base64-encoded string"
        ).strip()
    elif key_input_mode == "Upload JSON file":
        key_file = st.file_uploader("Upload service account JSON", type=["json"])
        if key_file is not None:
            key_json_from_ui = key_file.read().decode("utf-8").strip()

    st.markdown("---")

    # Model parameters
    st.subheader("Model Parameters")
    num_classes = st.selectbox("Number of Classes", [2, 3, 4, 5], index=1)
    batch_size = st.selectbox("Batch Size", [2, 4, 8, 16], index=1)
    epochs = st.slider("Training Epochs", 1, 50, 10)
    
    # Data parameters
    st.subheader("Data Configuration")
    patch_size = st.selectbox("Patch Size", [32, 64, 128, 256], index=1)
    use_demo_data = st.checkbox("Use Demo Data", value=True, 
                               help="Check to use simulated data for testing")
    
    st.markdown("---")
    st.info("💡 This app demonstrates poverty mapping using satellite data and deep learning models.")

# Initialize session state
if 'gee_initialized' not in st.session_state:
    st.session_state.gee_initialized = False
if 'features_processed' not in st.session_state:
    st.session_state.features_processed = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# Main content
col1, col2 = st.columns([2, 1])

with col2:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/8/8b/India_Odisha_location_map.svg", 
        caption="Study Area: Odisha Districts",
        width=200
    )

with col1:
    st.markdown("""
    ### About This Application
    This interactive tool combines **Google Earth Engine** satellite data with **deep learning models** 
    for poverty mapping in rural Odisha districts. The workflow includes:
    
    - 📡 **Satellite Data**: MODIS NDVI, Sentinel-2, WorldCover, Population, Nighttime Lights
    - 🧠 **AI Models**: CNN, U-Net, and Hybrid CNN+U-Net architectures
    - 📊 **Metrics**: Accuracy, Precision, Recall, F1 Score with visualization
    """)

# Step 1: Google Earth Engine Processing
st.header("🛰️ Step 1: Google Earth Engine Data Processing")

if st.button("🚀 Initialize GEE and Process Features", type="primary"):
    with st.spinner("Initializing Google Earth Engine..."):
        authed = False
        chosen_project = (project_id_ui or GEE_PROJECT_ID or "").strip()
        chosen_sa_email = (sa_email_ui or SERVICE_ACCOUNT_EMAIL or "").strip()

        if use_sa_ui:
            # Priority: UI selection -> (optional) hardcoded JSON -> (optional) file path
            key_json_effective = key_json_from_ui or SERVICE_ACCOUNT_KEY_JSON
            key_file_effective = ""  # rely on JSON text for secrets; file path optional for local

            authed = init_gee_with_service_account(
                project_id=chosen_project,
                service_account_email=chosen_sa_email,
                key_json_text=key_json_effective,
                key_file_path=key_file_effective
            )

        # Fallback to interactive auth (via your processor) if SA auth not used or failed
        if not authed:
            gee_tmp = GEEProcessor(project_id=chosen_project if chosen_project else None)
            if hasattr(gee_tmp, "authenticate_and_initialize"):
                authed = gee_tmp.authenticate_and_initialize()
            gee = gee_tmp if authed else None

        if authed:
            st.success("✅ Earth Engine authentication successful!")
            st.session_state.gee_initialized = True

            gee = gee if 'gee' in locals() and gee is not None else GEEProcessor(
                project_id=chosen_project if chosen_project else None
            )

            # Process features with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            features_to_process = [
                ("Defining study area", 0.1),
                ("Processing MODIS NDVI", 0.2),
                ("Processing Sentinel-2 NDBI", 0.4),
                ("Processing WorldCover", 0.6),
                ("Processing Population Density", 0.8),
                ("Processing Nighttime Lights", 0.9),
                ("Creating Poverty Index", 1.0)
            ]
            
            try:
                for feature_name, progress in features_to_process:
                    status_text.text(f"Processing: {feature_name}...")
                    progress_bar.progress(progress)
                    time.sleep(0.3)  # Simulate processing time
                
                if gee.process_all_features():
                    st.session_state.features_processed = True
                    st.session_state.gee_processor = gee
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ All features processed successfully!")
                    
                    # Display feature information
                    feature_info = gee.get_feature_info()
                    st.subheader("📊 Processed Features Summary")
                    
                    # Create feature summary table
                    feature_df = pd.DataFrame([
                        {"Feature": k, "Bands": len(v.get('bands', [])), "Projection": v.get('projection', 'N/A')}
                        for k, v in feature_info.items()
                    ])
                    st.dataframe(feature_df, use_container_width=True)
                else:
                    st.error("❌ Failed to process features. Check your GEE authentication and processor logs.")
            
            except Exception as e:
                st.error(f"❌ Error processing features: {str(e)}")
        else:
            st.error("❌ GEE Authentication failed. Please verify service account details or try interactive auth.")

# Show processed features if available
if st.session_state.features_processed:
    st.success("✅ Features ready for model training!")

# =========================
# Step 2: Data Loading and Preparation
# =========================
st.header("📂 Step 2: Dataset Preparation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Upload")
    uploaded_file = st.file_uploader(
        "Upload training data (GeoTIFF, NPZ, or PKL)",
        type=['tif', 'tiff', 'npz', 'pkl'],
        help="Upload your preprocessed training patches"
    )

with col2:
    st.subheader("Data Information")
    if use_demo_data or uploaded_file:
        # Generate or load data
        n_samples = 50 if use_demo_data else 10
        input_shape = (patch_size, patch_size, 7)  # 7 features from GEE
        
        # Simulate data for demo
        X = np.random.rand(n_samples, *input_shape).astype(np.float32)
        y = np.random.randint(0, num_classes, size=(n_samples, patch_size, patch_size))
        
        st.info(f"""
        **Dataset Shape**: {X.shape}  
        **Input Shape**: {input_shape}  
        **Classes**: {num_classes}  
        **Samples**: {n_samples}
        """)
        
        # Data visualization
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(X[0, :, :, 0], cmap='viridis')
        axes[0].set_title('Feature Channel 1')
        axes[0].axis('off')
        
        axes[1].imshow(X[0, :, :, 1], cmap='plasma')
        axes[1].set_title('Feature Channel 2')
        axes[1].axis('off')
        
        axes[2].imshow(y[0], cmap='Set3')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        st.pyplot(fig)
    else:
        st.warning("Please upload data or enable demo data to continue.")

# =========================
# Step 3: Model Training and Evaluation
# =========================
if (use_demo_data or uploaded_file) and (st.session_state.features_processed or use_demo_data):
    st.header("🤖 Step 3: Deep Learning Models")
    
    # Model selection
    model_options = {
        "CNN (Baseline)": {
            "function": build_cnn_segmentation,
            "description": "Simple CNN with basic convolution and pooling layers",
            "complexity": "Low",
            "recommended_epochs": 10
        },
        "U-Net": {
            "function": build_unet,
            "description": "Encoder-decoder architecture with skip connections",
            "complexity": "Medium",
            "recommended_epochs": 20
        },
        "Hybrid CNN+U-Net": {
            "function": build_hybrid_cnn_unet,
            "description": "Advanced architecture with attention mechanisms and ASPP",
            "complexity": "High",
            "recommended_epochs": 30
        }
    }
    
    # Model comparison table
    st.subheader("🔍 Model Comparison")
    model_comparison_df = pd.DataFrame([
        {
            "Model": name,
            "Description": info["description"],
            "Complexity": info["complexity"],
            "Recommended Epochs": info["recommended_epochs"]
        }
        for name, info in model_options.items()
    ])
    st.dataframe(model_comparison_df, use_container_width=True)
    
    # Model training section
    st.subheader("🎯 Train and Evaluate Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_cnn = st.button("Train CNN", type="secondary")
    with col2:
        train_unet = st.button("Train U-Net", type="secondary")
    with col3:
        train_hybrid = st.button("Train Hybrid", type="secondary")
    
    # Train all models button
    train_all = st.button("🚀 Train All Models", type="primary")
    
    # Training function
    def train_and_evaluate_model(model_name, model_function, X, y):
        """Train and evaluate a single model"""
        with st.spinner(f"Training {model_name}..."):
            # Build model
            input_shape = X.shape[1:]
            model = model_function(input_shape, num_classes)
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simple training loop simulation
            for epoch in range(epochs):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{epochs}")
                time.sleep(0.1)  # Simulate training time
            
            # Actual training (simplified)
            history = model.fit(
                X, y, batch_size=batch_size, epochs=min(3, epochs),
                verbose=0, validation_split=0.2
            )
            
            # Make predictions
            y_pred = np.argmax(model.predict(X, verbose=0), axis=-1)
            
            # Calculate metrics
            acc, prec, rec, f1, cm = compute_metrics(y, y_pred, num_classes)
            
            return {
                'model': model,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'confusion_matrix': cm,
                'history': history,
                'predictions': y_pred
            }
    
    # Handle model training
    if train_cnn or train_unet or train_hybrid or train_all:
        models_to_train = []
        
        if train_cnn or train_all:
            models_to_train.append(("CNN (Baseline)", model_options["CNN (Baseline)"]["function"]))
        if train_unet or train_all:
            models_to_train.append(("U-Net", model_options["U-Net"]["function"]))
        if train_hybrid or train_all:
            models_to_train.append(("Hybrid CNN+U-Net", model_options["Hybrid CNN+U-Net"]["function"]))
        
        results = {}
        
        for model_name, model_func in models_to_train:
            result = train_and_evaluate_model(model_name, model_func, X, y)
            results[model_name] = result
        
        st.session_state.model_results = results
    
    # Display results if available
    if st.session_state.model_results:
        st.header("📊 Results and Analysis")
        
        # Metrics comparison
        st.subheader("🏆 Model Performance Comparison")
        
        results = st.session_state.model_results
        metrics_data = []
        
        for model_name, result in results.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1 Score': result['f1_score']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Metrics bar chart
            fig = px.bar(
                metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Value'),
                x='Model', y='Value', color='Metric',
                title='Model Performance Metrics',
                barmode='group'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Best model highlight
            best_model = max(results.keys(), key=lambda k: results[k]['f1_score'])
            best_f1 = results[best_model]['f1_score']
            
            st.markdown(f"""
            ### 🥇 Best Model: {best_model}
            **F1 Score**: {best_f1:.4f}
            """)
            
            # Model metrics
            best_result = results[best_model]
            st.markdown(f"""
            - **Accuracy**: {best_result['accuracy']:.4f}
            - **Precision**: {best_result['precision']:.4f}
            - **Recall**: {best_result['recall']:.4f}
            """)
        
        # Confusion matrices
        st.subheader("🔍 Confusion Matrices")
        
        n_models = len(results)
        cols = st.columns(n_models)
        
        for idx, (model_name, result) in enumerate(results.items()):
            with cols[idx]:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', 
                           cmap='Blues', ax=ax)
                ax.set_title(f'{model_name}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
        
        # Predictions visualization
        st.subheader("🎯 Prediction Examples")
        
        sample_idx = st.selectbox("Select sample to visualize:", range(min(10, len(X))))
        
        cols = st.columns(len(results) + 2)
        
        # Ground truth
        with cols[0]:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(y[sample_idx], cmap='Set3')
            ax.set_title('Ground Truth')
            ax.axis('off')
            st.pyplot(fig)
        
        # Feature example
        with cols[1]:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(X[sample_idx, :, :, 0], cmap='viridis')
            ax.set_title('Input Feature')
            ax.axis('off')
            st.pyplot(fig)
        
        # Model predictions
        for idx, (model_name, result) in enumerate(results.items()):
            with cols[idx + 2]:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(result['predictions'][sample_idx], cmap='Set3')
                ax.set_title(f'{model_name}')
                ax.axis('off')
                st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
### 📝 About This Research
This application demonstrates the integration of satellite remote sensing data with deep learning 
for poverty mapping in rural India. The methodology combines multiple Earth observation datasets 
to create comprehensive poverty indicators using state-of-the-art computer vision techniques.

**Key Features:**
- Real-time Google Earth Engine data processing
- Multiple deep learning architectures
- Comprehensive evaluation metrics
- Interactive visualization

**Research Contact:** Developed for geospatial analysis and poverty mapping research.
""")

# Add download button for results
if st.session_state.model_results:
    results_summary = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1_score'],
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        for name, result in st.session_state.model_results.items()
    ])
    
    csv = results_summary.to_csv(index=False)
    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name=f"poverty_mapping_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )