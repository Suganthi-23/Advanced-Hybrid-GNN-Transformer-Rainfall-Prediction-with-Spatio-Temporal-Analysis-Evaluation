A Hybrid Spatio-Temporal Model Using a Novel Graphformer for Rainfall Prediction

Project Overview:
This project focuses on accurate rainfall prediction by modeling both spatial and temporal dependencies in weather data. A hybrid deep learning architecture is proposed that integrates Graph Neural Networks and Transformer models to effectively learn complex patterns from multi-station meteorological time-series data.

Objective:
To forecast rainfall by capturing dynamic spatial relationships between weather stations and long-term temporal trends in weather parameters.

Methodology:
Construction of spatial graphs based on geographical proximity of weather stations
Spatial feature extraction using Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT)
Dimensionality reduction and feature enhancement using Singular Value Decomposition (SVD)
Temporal sequence modeling using Transformer architecture
Hybrid fusion of spatial and temporal representations for final rainfall prediction

Technologies Used:
Python
PyTorch
Torch-Geometric
Scikit-Learn
NumPy
Pandas

Dataset Description:
The model utilizes multi-station historical weather data containing meteorological parameters and corresponding rainfall measurements, provided in CSV or tensor-based formats.
Model Training and Evaluation
The model is trained using GPU acceleration and evaluated using standard regression metrics including RMSE, MAE, R² score, and correlation coefficient.

Results:
The proposed Graphformer-based hybrid model demonstrates improved rainfall prediction accuracy with strong correlation between predicted and actual rainfall values.

Applications:
Weather forecasting systems
Agricultural planning
Flood and disaster management
Water resource management

Conclusion:
By jointly learning spatial and temporal features through graph-based and attention-based mechanisms, the proposed model provides an effective and scalable solution for rainfall prediction using time-series weather data.
