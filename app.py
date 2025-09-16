from ContentGeneration import chat_completion, add_citations
from GradCAMVisualizer import GradCAMVisualizer
from Prompt import build_prompt
from datetime import datetime
import plotly.express as px
import streamlit as st
from PIL import Image
import pandas as pd
import io
import os

def load_model():
    """
    Load the trained model and initialize the GradCAM visualizer.
    """
    # Path to the trained model file
    model_path = "models/best_model2.keras"

    # Initialize the GradCAM visualizer with the model
    visualizer = GradCAMVisualizer(model_path)
    return visualizer

def plot_scores(results_dict, class_names, title="Prediction Scores"):
    """
    Plot prediction scores as a horizontal bar chart.

    Args:
        results_dict (dict): Dictionary containing prediction results.
            Must include 'predictions' (list of scores).
        class_names (list): Names of the predicted classes.
        title (str): Title of the plot.

    Returns:
        fig (plotly.graph_objects.Figure): Bar plot figure.
    """
    # Extract prediction scores
    scores = results_dict['predictions']

    # Create DataFrame for Plotly visualization
    df = pd.DataFrame({
        'Class': class_names,
        'Score': scores
    })

    # Create horizontal bar plot
    fig = px.bar(
        df,
        y='Class',
        x='Score',
        orientation='h',
        title=title,
        color='Score',
        color_continuous_scale='viridis'
    )

    # Customize plot layout
    fig.update_layout(
        xaxis_title='Prediction Score',
        yaxis_title='',
        showlegend=False,
        plot_bgcolor='white',
        height=400,
        width=800,
        title_font_size=20,
        title_font_weight='bold'
    )

    # Customize x-axis
    fig.update_xaxes(
        range=[0, max(scores) * 1.15],  # extend slightly for labels
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=False
    )

    # Customize y-axis
    fig.update_yaxes(
        showgrid=False,
        zeroline=False
    )

    # Add numeric score labels to bars
    for i, score in enumerate(scores):
        fig.add_annotation(
            x=score + 0.005,  # small offset to the right of the bar
            y=class_names[i],
            text=f'{score:.4f}',
            showarrow=False,
            font=dict(size=12, weight='bold'),
            xanchor='left',
            yanchor='middle'
        )

    return fig

def reset_controls():
    """
    Reset the input controls by clearing the case name field.
    """
    st.session_state['input_text'] = ""         

# Get current timestamp for case identification
date = datetime.now()
date = date.strftime('%Y-%m-%d %H:%M:%S')

# Load GradCAM model/visualizer once
model = load_model()

# Initialize Streamlit session state if first run
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# App title
st.title("Brain Tumor Detector Tool")

# Input field for case name
title = st.text_input(
    "Case name:", 
    key='input_text',
    help="Write your case name and press enter"
)

if len(title) > 0:
    # Build unique case identifier with timestamp
    case_name = f"{title}_{date}"
    case_name = case_name.replace(" ", "").replace("-", "").replace(":", "").lower()
    
    # File uploader for MRI brain image
    uploaded_file = st.file_uploader(
        "Upload MRI Brain Image", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Read uploaded image into bytes
        image_bytes = uploaded_file.getvalue()

        # Open image with Pillow
        image = Image.open(io.BytesIO(image_bytes))

        # Display uploaded image
        st.image(image, caption=f"Study case: {case_name}", width='stretch')

        # Action button to start inference
        make_inference = st.button("Proceed with analysis", width='stretch', type='primary')
        if make_inference:
            # Save uploaded image locally
            os.makedirs(f"cases/{case_name}")
            img_path = f"cases/{case_name}/{uploaded_file.name}"
            img_path_inf = f"cases/{case_name}/inference_{uploaded_file.name}"
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            # Run model inference and GradCAM visualization
            with st.spinner("Waiting the MRI image is being analyzed...", show_time=True):
                results = model.visualize(img_path, img_path_inf)

                # Show predicted class
                st.header(f"Predicted Class: {results['predicted_class_name']}", divider=True)

                # Show prediction scores
                scores_plot = plot_scores(results, model.config.class_names)
                st.plotly_chart(scores_plot, use_container_width=True)

                # Show GradCAM visualization
                st.image(img_path_inf, case_name)

            # Run complementary analysis with LLM + citations
            with st.spinner("Finding complementary information...", show_time=True):
                prompt = build_prompt()
                response = chat_completion(prompt, img_path_inf)
                response_formatted = add_citations(response)

                if len(response_formatted) > 0:
                    with st.container(border=True, height=800):
                        st.markdown(response_formatted)

            # Button to reset for a new case study
            new_case = st.button(
                'New case study', 
                width='stretch', 
                type='tertiary', 
                on_click=reset_controls
            )