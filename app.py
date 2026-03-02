# app.py - Blood Pressure Classification Prediction Application
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

# Page configuration
st.set_page_config(
    page_title="Blood Pressure Prediction Tool",
    page_icon="❤️",
    layout="wide"
)

# Application title
st.title("❤️ Blood Pressure Classification Prediction")
st.markdown("### Machine Learning-Based Blood Pressure Anomaly Detection System")


# Time conversion function
def calculate_time_features(date_input, hour_input):
    """
    Calculate sine and cosine values for month and hour from date and hour

    Parameters:
        date_input: datetime.date object
        hour_input: int (0-23)

    Returns:
        month_sin, month_cos, hour_sin, hour_cos
    """
    # Extract month (1-12)
    month = date_input.month

    # Calculate sine and cosine values for hour (0-23)
    hour_sin = math.sin(2 * math.pi * hour_input / 24)
    hour_cos = math.cos(2 * math.pi * hour_input / 24)

    # Calculate sine and cosine values for month (1-12)
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)

    return month_sin, month_cos, hour_sin, hour_cos


# Load model and configuration
@st.cache_resource
def load_model_and_config():
    """Load model and configuration"""
    try:
        # Load Streamlit configuration
        with open('streamlit_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Load model
        with open(config['model_file'], 'rb') as f:
            model = pickle.load(f)

        # Load scaler
        with open(config['scaler_file'], 'rb') as f:
            scaler = pickle.load(f)

        # Load feature names
        feature_names = config['feature_names']
        optimal_threshold = config['optimal_threshold']

        return model, scaler, feature_names, optimal_threshold, config

    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None, None, None, None


# Load model
model, scaler, feature_names, optimal_threshold, config = load_model_and_config()

if model is None:
    st.warning("⚠️ Unable to load model, please ensure model files exist.")
    st.stop()

# Display model information
with st.sidebar:
    st.header("📊 Model Information")
    st.markdown(f"**Model Type**: {config.get('model_name', 'XGBoost')}")
    st.markdown(f"**Number of Features**: {len(feature_names)}")
    st.markdown(f"**Optimal Threshold**: {optimal_threshold:.3f}")

    # Display performance metrics (if available)
    if 'performance_metrics' in config:
        st.markdown("---")
        st.subheader("Model Performance")
        metrics = config['performance_metrics']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("AUC", f"{metrics.get('AUC', 0):.3f}")
            st.metric("Accuracy", f"{metrics.get('Accuracy', 0):.3f}")
        with col2:
            st.metric("F1 Score", f"{metrics.get('F1_score', 0):.3f}")
            st.metric("Recall", f"{metrics.get('Recall', 0):.3f}")

    st.markdown("---")
    st.info("💡 **Instructions**: Fill all parameters and click Predict button")

# Main content area - Input form
st.header("📝 Input Parameters")

# Create input columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Physiological Parameters")
    age = st.slider("**Age**", 18, 90, 50, help="Select age (18-90 years)")
    weight = st.slider("**Weight (kg)**", 40.0, 120.0, 60.0, 0.1, help="Enter weight (40-120 kg)")
    height = st.slider("**Height (cm)**", 140.0, 200.0, 165.0, 0.1, help="Enter height (140-200 cm)")
    bmi = st.slider("**BMI**", 15.0, 40.0, 22.0, 0.1, help="Body Mass Index (15-40)")

with col2:
    st.subheader("Medical Parameters")
    alt = st.slider("**ALT (U/L)**", 0.0, 100.0, 20.0, 0.1, help="Alanine aminotransferase (0-100 U/L)")
    pulse = st.slider("**Pulse (bpm)**", 40, 120, 72, help="Pulse rate (40-120 beats per minute)")

    st.subheader("Time Parameters")
    st.markdown("Select the date and hour for prediction")

    # Get current date and time as default
    current_date = datetime.date.today()
    current_time = datetime.datetime.now()

    # Date picker
    date_input = st.date_input(
        "**Date**",
        value=current_date,
        help="Select prediction date"
    )

    # Hour selector (0-23) with hour format display
    hour_input = st.slider(
        "**Hour (0-23)**",
        min_value=0,
        max_value=23,
        value=current_time.hour,
        help="Select hour of day (0 = midnight, 12 = noon, 23 = 11 PM)"
    )

    # Display selected hour in 24-hour format
    hour_display = f"{hour_input:02d}:00"
    st.markdown(f"**Selected time:** {hour_display}")

    # Calculate time features
    month_sin, month_cos, hour_sin, hour_cos = calculate_time_features(date_input, hour_input)

    # Display conversion results (only sin values)
    st.markdown("**Time Feature Conversion:**")
    col_time1, col_time2 = st.columns(2)
    with col_time1:
        st.write(f"Month: {date_input.month}")
        st.write(f"month_sin: {month_sin:.4f}")
    with col_time2:
        st.write(f"Hour: {hour_display}")
        st.write(f"hour_sin: {hour_sin:.4f}")

# Prediction button and result area
st.markdown("---")
st.header("🔮 Prediction Results")

# Create prediction button column
predict_col, result_col = st.columns([1, 2])

with predict_col:
    predict_button = st.button("Start Prediction", type="primary", use_container_width=True)

# Initialize result state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
    st.session_state.prediction_prob = None
    st.session_state.feature_values = None
    st.session_state.time_features = None

# Handle prediction
if predict_button:
    with st.spinner("Calculating..."):
        try:
            # Recalculate time features
            month_sin, month_cos, hour_sin, hour_cos = calculate_time_features(date_input, hour_input)

            # Prepare feature data (must follow training feature order)
            feature_values = {
                'age': age,
                'alt': alt,
                'height': height,
                'weight': weight,
                'pulse': pulse,
                'BMI': bmi,
                'month_sin': month_sin,
                'month_cos': month_cos,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos
            }

            # Store time features separately for display
            st.session_state.time_features = {
                'date': date_input.strftime("%Y-%m-%d"),
                'time': f"{hour_input:02d}:00",
                'month': date_input.month,
                'hour': hour_input,
                'month_sin': month_sin,
                'hour_sin': hour_sin
            }

            # Ensure all features exist
            input_features = []
            for feature in feature_names:
                if feature in feature_values:
                    input_features.append(feature_values[feature])
                else:
                    # If feature doesn't exist, use default value
                    st.warning(f"Feature '{feature}' not in input, using default value 0")
                    input_features.append(0)

            # Convert to numpy array
            X_input = np.array([input_features])

            # Data standardization
            X_scaled = scaler.transform(X_input)

            # Prediction
            probability = model.predict_proba(X_scaled)[0, 1]
            prediction = 1 if probability >= optimal_threshold else 0

            # Save results to session state
            st.session_state.prediction_result = prediction
            st.session_state.prediction_prob = probability
            st.session_state.feature_values = feature_values

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Display results
with result_col:
    if st.session_state.prediction_result is not None:
        probability = st.session_state.prediction_prob
        prediction = st.session_state.prediction_result

        # Display probability gauge
        st.subheader("Prediction Result")

        # Create two columns for results
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            # Blood pressure status indicator
            if prediction == 0:
                st.success("✅ **Normal Blood Pressure**")
                st.metric("Abnormal Probability", f"{probability:.3f}")
            else:
                st.error("⚠️ **Abnormal Blood Pressure**")
                st.metric("Abnormal Probability", f"{probability:.3f}")

            # Threshold indicator
            st.progress(float(probability))
            st.caption(f"Threshold: {optimal_threshold:.3f}")

        with result_col2:
            # Display detailed analysis
            st.markdown("#### Detailed Analysis")

            # Calculate risk level
            if probability < 0.3:
                risk_level = "Low Risk"
                risk_color = "green"
            elif probability < 0.7:
                risk_level = "Medium Risk"
                risk_color = "orange"
            else:
                risk_level = "High Risk"
                risk_color = "red"

            st.markdown(f"**Risk Level**: :{risk_color}[{risk_level}]")

            # Recommendations
            if prediction == 0:
                st.markdown("**Recommendation**: Maintain healthy lifestyle, monitor blood pressure regularly")
            else:
                st.markdown("**Recommendation**: Consider further examination and consult a doctor")

        # Display input parameters review
        st.markdown("---")
        with st.expander("📋 View Input Parameters"):
            if st.session_state.feature_values:
                # Create display dictionary without cos values
                display_features = {
                    'age': st.session_state.feature_values.get('age'),
                    'alt': st.session_state.feature_values.get('alt'),
                    'height': st.session_state.feature_values.get('height'),
                    'weight': st.session_state.feature_values.get('weight'),
                    'pulse': st.session_state.feature_values.get('pulse'),
                    'BMI': st.session_state.feature_values.get('BMI'),
                    'month_sin': st.session_state.feature_values.get('month_sin'),
                    'hour_sin': st.session_state.feature_values.get('hour_sin')
                }

                param_df = pd.DataFrame(
                    list(display_features.items()),
                    columns=['Parameter', 'Value']
                )
                st.table(param_df)

            if st.session_state.time_features:
                st.markdown("**Time Features:**")
                time_df = pd.DataFrame(
                    list(st.session_state.time_features.items()),
                    columns=['Time Parameter', 'Value']
                )
                st.table(time_df)

    else:
        st.info("👆 Click 'Start Prediction' button to see results")

# Feature importance visualization with force diagram
try:
    if hasattr(model, 'feature_importances_') and st.session_state.prediction_result is not None:
        st.markdown("---")
        st.header("📊 Feature Contribution Analysis (Force Diagram)")

        # Get feature importances
        importances = model.feature_importances_

        # Get current feature values
        if st.session_state.feature_values:
            # Create feature contribution visualization
            feature_contributions = []

            for i, feature_name in enumerate(feature_names):
                if feature_name in st.session_state.feature_values:
                    feature_value = st.session_state.feature_values[feature_name]
                    # Calculate contribution (importance * normalized value)
                    # For XGBoost, we can use feature importances as weights
                    contribution = importances[i] * feature_value
                    feature_contributions.append({
                        'Feature': feature_name,
                        'Importance': importances[i],
                        'Value': feature_value,
                        'Contribution': contribution,
                        'AbsContribution': abs(contribution)
                    })

            # Create DataFrame
            contribution_df = pd.DataFrame(feature_contributions)
            contribution_df = contribution_df.sort_values('AbsContribution', ascending=False)

            # Display top contributing features
            st.markdown("#### Top Contributing Features")

            # Create force diagram visualization
            fig, ax = plt.subplots(figsize=(12, 8))

            # Prepare data for plotting (top 10 features)
            top_features = contribution_df.head(10)
            features = top_features['Feature'].tolist()
            contributions = top_features['Contribution'].tolist()

            # Color code: positive contributions (increase risk) in red, negative in blue
            colors = ['#FF6B6B' if c > 0 else '#4ECDC4' for c in contributions]

            # Create horizontal bar chart (force diagram style)
            y_pos = np.arange(len(features))

            # Plot bars
            bars = ax.barh(y_pos, contributions, color=colors, alpha=0.7, edgecolor='black')

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, contributions)):
                width = bar.get_width()
                label_x = width + (0.01 * max(contributions) if width >= 0 else 0.01 * min(contributions))
                ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                        f'{val:.4f}',
                        va='center', ha='left' if width >= 0 else 'right',
                        fontsize=9, fontweight='bold')

            # Set chart properties
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('Contribution Value', fontsize=12, fontweight='bold')
            ax.set_title('Feature Contribution Force Diagram', fontsize=14, fontweight='bold', pad=20)

            # Add grid lines
            ax.grid(True, alpha=0.3, linestyle='--', axis='x')

            # Add zero reference line
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

            # Add legend
            import matplotlib.patches as mpatches

            pos_patch = mpatches.Patch(color='#FF6B6B', alpha=0.7, label='Positive Contribution (Increase Risk)')
            neg_patch = mpatches.Patch(color='#4ECDC4', alpha=0.7, label='Negative Contribution (Decrease Risk)')
            ax.legend(handles=[pos_patch, neg_patch], loc='upper right', fontsize=10)

            # Add prediction probability info
            info_text = f"Prediction Probability: {probability:.3f} | Threshold: {optimal_threshold:.3f}"
            ax.text(0.5, -0.1, info_text, transform=ax.transAxes,
                    fontsize=11, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))

            plt.tight_layout()
            st.pyplot(fig)

            # Display detailed contribution table
            with st.expander("View Detailed Contribution Analysis"):
                st.dataframe(contribution_df, use_container_width=True)

                # Additional analysis
                st.markdown("##### Contribution Summary")
                total_positive = contribution_df[contribution_df['Contribution'] > 0]['Contribution'].sum()
                total_negative = contribution_df[contribution_df['Contribution'] < 0]['Contribution'].sum()
                net_contribution = total_positive + total_negative

                col_sum1, col_sum2, col_sum3 = st.columns(3)
                with col_sum1:
                    st.metric("Positive Contributions", f"{total_positive:.4f}")
                with col_sum2:
                    st.metric("Negative Contributions", f"{total_negative:.4f}")
                with col_sum3:
                    st.metric("Net Contribution", f"{net_contribution:.4f}")

except Exception as e:
    # If unable to create force diagram, skip
    st.warning(f"Unable to create force diagram: {str(e)}")

# Blood pressure classification reference
st.markdown("---")
st.header("📚 Blood Pressure Classification Reference")

col_ref1, col_ref2 = st.columns(2)

with col_ref1:
    st.markdown("""
    ### Blood Pressure Classification
    | Category | Systolic (mmHg) | Diastolic (mmHg) |
    |----------|-----------------|------------------|
    | Normal | < 120 | < 80 |
    | Elevated | 120-129 | 80-84 |
    | Stage 1 Hypertension | 130-139 | 85-89 |
    | Stage 2 Hypertension | 140-159 | 90-99 |
    | Stage 3 Hypertension | ≥ 160 | ≥ 100 |
    """)

with col_ref2:
    st.markdown("""
    ### Important Notes
    1. **This is a prediction tool** - results are for reference only
    2. **Actual diagnosis** should be based on medical institution measurements
    3. **Recommend** regular blood pressure monitoring
    4. **Maintain** a healthy lifestyle
    5. **If abnormal** consult a doctor promptly
    """)

# Footer
st.markdown("---")
st.caption("💡 Note: This tool is based on machine learning models, accuracy depends on training data")
st.caption("🔒 Privacy Protection: All calculations are done on the server, no user data is saved")
st.caption(f"🔄 Last Updated: {config.get('training_info', {}).get('deployment_date', 'Unknown')}")

# Debug information (only shown in development mode)
if st.sidebar.checkbox("Show Debug Information", False):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Debug Information")
    st.sidebar.json(config)
    st.sidebar.write(f"Feature Names: {feature_names}")
    st.sidebar.write(f"Number of Features: {len(feature_names)}")
