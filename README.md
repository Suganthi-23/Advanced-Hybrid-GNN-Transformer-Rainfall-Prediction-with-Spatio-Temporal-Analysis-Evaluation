# Hybrid GNN-Transformer System for Multi-Station Rainfall Forecasting

I built this project to solve a key limitation in traditional rainfall prediction systems, where most approaches rely on single-station data or purely temporal models that fail to capture spatial dependencies between locations.

I designed and developed a hybrid spatio-temporal AI system that models both spatial relationships across weather stations and temporal weather patterns. I implemented a Graph Neural Network architecture (GCN + GAT) to learn dynamic interactions between stations and integrated it with a Transformer model to capture long-range temporal dependencies.

To improve prediction stability, I incorporated feature extraction techniques on multi-station meteorological data, enabling more consistent and reliable outputs.

Compared to conventional statistical and LSTM-based models, my approach improves predictive performance by explicitly combining spatial and temporal learning. The model achieves MAE ≈ 2.7 mm and RMSE ≈ 6.6 mm, supported by detailed evaluation including prediction vs actual analysis, residual distributions, and time-series forecasts.

This project reflects my ability to design and build end-to-end AI systems that go beyond standard approaches, delivering scalable, interpretable, and high-performance solutions for real-world climate prediction problems.

# Key Results

- MAE: 2.699 mm  
- RMSE: 6.581 mm  
- Correlation: 0.451  

The model demonstrates strong performance in rainfall magnitude prediction, with significantly improved correlation at aggregated levels (weekly and seasonal), indicating robust trend learning.


# Detailed Evaluation

# Standard Evaluation
- Samples: 12,560  
- RMSE: 6.581 mm  
- MAE: 2.699 mm  
- R²: 0.137  
- Correlation: 0.451  

# Rain Event Evaluation (≥ 1 mm threshold)
- RMSE: 10.354 mm  
- MAE: 5.580 mm  
- Correlation: 0.364  

# Weekly Aggregation
- RMSE: 33.197 mm  
- MAE: 18.942 mm  
- Correlation: 0.724  

# Seasonal Aggregation
- RMSE: 531.463 mm  
- MAE: 411.630 mm  
- Correlation: 0.946  

# Tech Stack

Python - PyTorch - Torch-Geometric - NumPy - Pandas - Scikit-learn
