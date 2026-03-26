# Problem Statement
Satellite imagery is widely used in domains such as environmental monitoring, urban planning, and disaster management. However, raw satellite data is not directly usable and requires interpretation to extract meaningful information such as land cover types or potential anomalies.

<br>Traditional approaches rely on manual analysis or specialized geospatial tools, which can be time-consuming, resource-intensive, and not easily accessible to non-experts. Additionally, rapid assessment of large-scale imagery during critical situations (e.g., environmental changes or fire-affected regions) remains a challenge.

<br>There is a need for an automated system that can process satellite images efficiently and provide reliable classification of land cover, along with basic identification of high-intensity regions, using accessible machine learning techniques.

# Scope of the Project
TerraMap.ai focuses on building a satellite image analysis system that bridges the gap between raw Earth observation data and usable insights.
* <br>Primary Scope:
<br>Classification of land cover into categories such as forest, urban, water, agriculture, and arid land using deep learning.
* <br>Secondary Scope:
<br>thermal pattern analysis using pixel intensity to highlight potential high-intensity regions.
* <br>Technical Scope:
<br>Implementation of a computer vision pipeline using a ResNet-50 model in PyTorch, designed to run as a standalone application with a simple user interface.

# Target Users
* Environmental Researchers
Can use the system to analyze land use patterns, vegetation distribution, and environmental changes over time using satellite imagery.
* Disaster Management Teams
Helps in identifying regions that may indicate fire-affected areas or abnormal surface conditions, supporting faster situational assessment.
* Students and Engineers
Useful for learning and experimenting with deep learning applications in remote sensing, computer vision, and Earth observation systems.
* Urban Planners
Assists in understanding urban expansion, land distribution, and infrastructure development through automated terrain classification.

# High-Level Features
* Deep Learning-Based Classification
  <br>Implements a ResNet-50 convolutional neural network to classify satellite images into land cover categories such as forest, urban, water, agriculture, and arid land, enabling automated terrain analysis.
* Thermal Pattern Indicator (Experimental)
<br>Performs pixel-intensity-based analysis to highlight high-intensity regions in images, providing a basic indication of possible anomalies such as fire or heat-affected areas.
* Interactive Dashboard
<br>Built using Streamlit, allowing users to upload satellite images, run the details , and view outputs through a simple and accessible interface.
* Model Evaluation Metrics
<br>Displays key performance metrics including accuracy, confusion matrix, and F1-score, enabling detailed evaluation of model predictions and class-wise performance.
* Training Visualization
<br>Provides graphical representation of training loss and accuracy over epochs, helping users understand model convergence and learning behavior.
* Visual Output Comparison
<br>Shows the input satellite image alongside predicted results and confidence levels, allowing quick verification and interpretation of outputs.
