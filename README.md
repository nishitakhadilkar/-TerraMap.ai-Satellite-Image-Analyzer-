# **Project title:**
TerraMap.ai | Satellite-Image-Analyzer

# **Overview**
* **TerraMap.ai** is a deep learning-based satellite image classification system designed to identify land cover types such as forest, water, urban, smoke and fire detection and agricultural areas from Earth observation data.
The model is built using ResNet-50 in PyTorch and trained on the EuroSAT dataset. It uses transfer learning along with preprocessing and data augmentation techniques to improve generalization. The system achieves 92% classification accuracy and evaluates performance using confusion matrix analysis.
This project demonstrates how deep learning can be applied in remote sensing for automated terrain analysis.

# **Features**
*	Satellite Image Classification:
<br>Classifies and maps satellite images into land cover categories such as forest, urban, water, and agriculture using a ResNet-50 deep learning model.
*	Data Preprocessing and Augmentation:
<br>Implements image resizing, normalization, and augmentation (rotation, flipping) to improve model generalization across diverse satellite imagery.
*	Thermal Pattern Analysis (Experimental):
<br>It performs pixel-intensity-based analysis to highlight high-intensity regions, serving as a basic approach for anomaly detection.
*	Performance Evaluation Metrics:
<br>It evaluates model performance using accuracy (92%), confusion matrix, and F1-score for detailed class-wise analysis.
*	Streamlit Interface:
<br>Provides a simple web interface to upload satellite images and visualize classification results in an accessible format.
*	Modular and Extendable Design:
<br>It has structured pipeline that allows future extension to object detection models (e.g., YOLO) or real-time satellite data integration TURES.

# **Technology and tools used**
*	Programming Language: 
<br>Python 3.x
*	Frameworks:
<br>PyTorch, Torchvision → model development and training 
<br>Streamlit → user interface for interaction
*	Libraries:
<br>NumPy → numerical computations 
<br>Pandas → data handling 
<br>Matplotlib → result visualization 
<br>scikit-learn → evaluation metrics (confusion matrix, F1-score) ;
<br>Model Architecture: ResNet-50 
<br>Used for feature extraction and multi-class classification.
*	Dataset:
<br>EuroSAT- Satellite imagery dataset used for training and evaluation.
*	Hardware Acceleration:
<br>NVIDIA CUDA → accelerates training and inference
*	Execution Environment:
<br>Python IDLE → development and testing 
<br>Command Prompt (CLI) → running training and evaluation scripts

# **Steps to install and run the project**
* 1)Install Python & Dependencies
<br>Install Python 3.10+ on your system. Then install the required libraries using:
<br>[pip install streamlit torch torchvision pillow numpy pandas plotly]
* 2)Download Model Weights
<br> a stable internet connection during the first run so the ResNet-50 pretrained weights can be downloaded automatically.
* 3)Run the Application
<br>Run the project using:
<br>[streamlit run SpaceProject.py]
* 4)Use the Tool
<br>Follow the UI to upload a satellite image and view the classification results.

# **Instructions for testing**
*	Terrain Classification Test :
<br>Upload different satellite images (forest, urban, water, agriculture) and verify whether the predicted class matches the actual terrain.
*	Robustness Test :
<br>Test with low-resolution or noisy images to observe how the model handles degraded input and how predictions change.
*	Thermal Detection Test :
<br>Upload images containing fire or smoke and check if the system correctly flags high-intensity regions in the thermal status output.
*	Model Evaluation Check :
<br>Review the confusion matrix and F1-score to analyze class-wise performance and identify misclassifications.
*	Training Performance Review :
<br>Check the training history graph to ensure loss decreases over epochs and the model shows proper learning behavior.
*	Interface Testing :
<br>Verify that the Streamlit interface correctly handles image uploads and displays outputs without errors.





