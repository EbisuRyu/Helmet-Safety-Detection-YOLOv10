import os
import cv2
import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter
from ultralytics import YOLOv10


def find_pt_files(directory):
    """
    🔍 Find all files with .pt extension in the directory.

    Args:
        directory (str): The directory to search for .pt files.

    Returns:
        list: A list of file paths with .pt extension.
    """
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith('.pt')]


def save_uploaded_file(uploaded_file, temp_dir='./streamlit/input', file_name='input.jpg'):
    """
    💾 Save the uploaded file to a temporary path.

    Args:
        uploaded_file (UploadedFile): The file uploaded by the user.
        temp_dir (str): The directory to save the file temporarily.
        file_name (str): The name to save the file as.

    Returns:
        str: The path to the saved file.
    """
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file_name)
    with open(temp_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.getbuffer())
    return temp_path


def display_image_results(result, temp_path, result_save_path):
    """
    📊 Display the detection result and detailed statistics for images.

    Args:
        result (Results): The result from the YOLO model.
        temp_path (str): The path to the input image.
        result_save_path (str): The path to the result image.
    """
    st.markdown('**📊 Detection Result**')
    column_1, column_2 = st.columns(2)
    column_1.image(temp_path, caption='Input 🖼️')
    column_2.image(result_save_path, caption='Result 🖼️')

    st.markdown('**📈 Detection Statistics**')
    label_dict = result.names
    objects = [label_dict[int(box.cls[0])] for box in result.boxes]
    confidences = [box.conf[0].item() for box in result.boxes]
    object_counter = Counter(objects)

    # Save detailed statistics to DataFrame
    dataframe = pd.DataFrame({'Object': [f'Object {i}' for i in range(len(objects))],
                              'Label': objects, 'Confidence': confidences})
    st.dataframe(dataframe, use_container_width=True)

    # Create and display bar chart for object counts
    fig_counts = px.bar(x=object_counter.keys(), y=object_counter.values(),
                        labels={'x': 'Labels', 'y': 'Counts'}, title='Object Counting')
    st.plotly_chart(fig_counts)


def display_video_results(temp_path, result_save_path):
    """
    📊 Display the detection result for videos.

    Args:
        temp_path (str): The path to the input video.
        result_save_path (str): The path to the result video.
    """
    st.markdown('**📊 Detection Result**')
    column_1, column_2 = st.columns(2)
    column_1.write('**📥 Input**')
    column_1.video(temp_path)
    column_2.write('**📤 Output**')
    column_2.video(result_save_path)


# Sidebar Configuration
with st.sidebar:
    st.title('🛠️ Configuration for Helmet Detection')
    input_type = st.selectbox('**Type**', ['Image 🖼️', 'Video 🎥'])
    model_path = st.selectbox('**Model**', find_pt_files('./'))
    image_size = st.number_input('**Image Size**', value=640)
    confidence_threshold = st.number_input(
        '**Confidence Threshold**', value=0.5)
    save_dir = st.text_input('**Save Directory**', value='./streamlit/output')

    configuration = {
        'Input Type': input_type,
        'Model Path': model_path,
        'Image Size': image_size,
        'Confidence Threshold': confidence_threshold,
        'Save Directory': save_dir
    }

# Main title
st.title('🪖 Helmet Safety Detection using YOLOv10')
st.divider()

# File uploader
uploaded_file = st.file_uploader(
    "**📁 Choose file**", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
st.divider()
st.write('**⚙️ Configuration**')
st.write(configuration)
st.divider()

# Load model
model = YOLOv10(model_path)

# Predict button
if st.button('**🚀 Predict**'):
    if uploaded_file:
        file_name = 'input.jpg' if 'Image' in input_type else 'input.mp4'
        temp_path = save_uploaded_file(uploaded_file, file_name=file_name)
        result_save_path = os.path.join(
            save_dir, f'output.{file_name.split(".")[-1]}')
        os.makedirs(save_dir, exist_ok=True)

        if 'Image' in input_type:
            with st.status(label="**🔍 Detecting...**", expanded=True) as status:
                result = model.predict(
                    source=temp_path, imgsz=image_size, conf=confidence_threshold)[0]
                result.save(result_save_path)
                status.update(label="**🎉 Detection complete!**",
                              state="complete", expanded=True)
                display_image_results(result, temp_path, result_save_path)
        elif 'Video' in input_type:
            with st.status(label="**🔍 Detecting...**", expanded=True) as status:
                cap = cv2.VideoCapture(temp_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                codec = int(cap.get(cv2.CAP_PROP_FOURCC))
                out = cv2.VideoWriter(
                    result_save_path, codec, fps, (frame_width, frame_height))

                while cap.isOpened():
                    success, frame = cap.read()
                    if success:
                        results = model.predict(
                            source=frame, imgsz=image_size, conf=confidence_threshold)
                        annotated_frame = results[0].plot()
                        out.write(annotated_frame)
                    else:
                        break

                cap.release()
                out.release()
                status.update(label="**🎉 Detection complete!**",
                              state="complete", expanded=True)
                display_video_results(temp_path, result_save_path)
    else:
        st.info('Please upload an input file 📁.')
