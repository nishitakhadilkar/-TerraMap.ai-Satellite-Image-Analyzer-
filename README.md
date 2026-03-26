#**PROJECT TITLE:**
TerraMap.ai | Satellite-Image-Analyzer

##**OVERVIEW**
* **TerraMap.ai** is a deep learning-based satellite image classification system designed to identify land cover types such as forest, water, urban, smoke and fire detection and agricultural areas from Earth observation data.
The model is built using ResNet-50 in PyTorch and trained on the EuroSAT dataset. It uses transfer learning along with preprocessing and data augmentation techniques to improve generalization. The system achieves 92% classification accuracy and evaluates performance using confusion matrix analysis.
This project demonstrates how deep learning can be applied in remote sensing for automated terrain analysis.

##**FEATURES**
*	Satellite Image Classification
Classifies and maps satellite images into land cover categories such as forest, urban, water, and agriculture using a ResNet-50 deep learning model.
*	Data Preprocessing and Augmentation
Implements image resizing, normalization, and augmentation (rotation, flipping) to improve model generalization across diverse satellite imagery.
*	Thermal Pattern Analysis (Experimental)
It performs pixel-intensity-based analysis to highlight high-intensity regions, serving as a basic approach for anomaly detection.
*	Performance Evaluation Metrics
It evaluates model performance using accuracy (92%), confusion matrix, and F1-score for detailed class-wise analysis.
*	Streamlit Interface
Provides a simple web interface to upload satellite images and visualize classification results in an accessible format.
*	Modular and Extendable Design
It has structured pipeline that allows future extension to object detection models (e.g., YOLO) or real-time satellite data integration TURES.

##**TECHNOLOGIES AND TOOLS USED**
*	Programming Language: 
Python 3.x
*	Frameworks:
PyTorch, Torchvision → model development and training
Streamlit → user interface for interaction
*	Libraries:
NumPy → numerical computations
Pandas → data handling
Matplotlib → result visualization
scikit-learn → evaluation metrics (confusion matrix, F1-score)
Model Architecture: ResNet-50
Used for feature extraction and multi-class classification.
*	Dataset:
EuroSAT- Satellite imagery dataset used for training and evaluation.
*	Hardware Acceleration:
NVIDIA CUDA → accelerates training and inference
*	Execution Environment:
Python IDLE → development and testing
Command Prompt (CLI) → running training and evaluation scripts

##**STEPS TO INSTALL AND RUN THE PROJECT**
1)Install Python & Dependencies
Install Python 3.10+ on your system. Then install the required libraries using:
[pip install streamlit torch torchvision pillow numpy pandas plotly]
2)Download Model Weights
Ensure a stable internet connection during the first run so the ResNet-50 pretrained weights can be downloaded automatically.
3)Run the Application
Run the project using:
[streamlit run SpaceProject.py]
4)Use the Tool
Follow the UI to upload a satellite image and view the classification results.

##**INSTRUCTIONS for TESTING**
•	Terrain Classification Test
Upload different satellite images (forest, urban, water, agriculture) and verify whether the predicted class matches the actual terrain.
•	Robustness Test
Test with low-resolution or noisy images to observe how the model handles degraded input and how predictions change.
•	Thermal Detection Test
Upload images containing fire or smoke and check if the system correctly flags high-intensity regions in the thermal status output.
•	Model Evaluation Check
Review the confusion matrix and F1-score to analyze class-wise performance and identify misclassifications.
•	Training Performance Review
Check the training history graph to ensure loss decreases over epochs and the model shows proper learning behavior.
•	Interface Testing
Verify that the Streamlit interface correctly handles image uploads and displays outputs without errors.





