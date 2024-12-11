# Face Mask Detection Application



## Project Overview



The Face Mask Detection Application is a computer vision project designed to identify and classify faces in images or video streams as "Mask" or "No Mask" using deep learning techniques. It features an interactive Streamlit interface for user interaction.



---



## Table of Contents



1. [Project Features](#project-features)

2. [System Requirements](#system-requirements)

3. [Setup Instructions](#setup-instructions)

4. [Steps to Execute](#steps-to-execute)

5. [Usage](#usage)

6. [Troubleshooting](#troubleshooting)



---



## Project Features



- **Face Detection**: Detects faces in uploaded images or live webcam feeds.

- **Mask Classification**: Classifies faces as "Mask" or "No Mask".

- **Bounding Boxes**: Draws labeled bounding boxes around detected faces.

- **Real-time Detection**: Supports live detection via webcam.

- **Visual Feedback**: Displays results with annotated images and counts.



---



## System Requirements



- **Operating System**: Windows/Linux/MacOS

- **Python Version**: Python 3.9 or higher

- **Hardware**:

  - A computer with a webcam (for live detection).

  - Minimum 4GB RAM.



---



## Setup Instructions



### 1. Clone the Repository



```bash

$ git clone https://github.com/your-repository/facemask-detection.git

$ cd facemask-detection

```



### 2. Set Up Virtual Environment



```bash

$ python -m venv venv

$ source venv/bin/activate    # For Linux/MacOS

$ venv\Scripts\activate       # For Windows

```



### 3. Install Dependencies



```bash

$ pip install -r requirements.txt

```



### 4. Download Pre-trained Model



Download the pre-trained `model2.h5` file and place it in the project root directory.



### 5. Verify Dataset



Ensure the dataset directory for training/testing is present (if needed for retraining).



---



## Steps to Execute



### 1. Launch the Application



Run the `app.py` file using Streamlit:



```bash

$ streamlit run app.py

```



### 2. Access the Application



After running the command, a local URL will be provided (e.g., `http://localhost:8501`). Open the link in a web browser.



### 3. Choose an Interaction Mode



- **Image Upload**:

  - Upload an image file (JPEG/PNG) via the upload tab.

  - The application will process the image, annotate it, and display the results.

- **Live Webcam Feed**:

  - Select the "Live Webcam" tab.

  - Choose the camera index and click "Start Webcam".



---



## Usage



### For Image Upload



1. Navigate to the *Image Upload* tab.

2. Select an image file to upload.

3. View annotated results with counts of "Mask" and "No Mask" detections.



### For Live Webcam Feed



1. Navigate to the *Live Webcam* tab.

2. Select your camera index (default is 0).

3. Click "Start Webcam" to begin real-time mask detection.



---



## Troubleshooting



1. **Webcam Not Working**:



   - Ensure your webcam is connected and enabled.

   - Check the camera index selection in the app.



2. **Module Import Errors**:



   - Verify all dependencies are installed correctly by running `pip install -r requirements.txt`.



3. **Model File Not Found**:



   - Confirm `model2.h5` is in the project root directory.



4. **Application Crashes on Launch**:



   - Check Python version compatibility.

   - Ensure you have activated the virtual environment.




