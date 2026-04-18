**#Disease Predictor: Genomic Data Mining**
Overview
This project explores the intersection of bioinformatics and machine learning. I developed a prototype system that analyzes genomic sequence data to predict susceptibility to various genetic disorders and their specific subclasses. 

**The Data**

Source: Kaggle Genetic Disorders Dataset. Link: https://www.kaggle.com/datasets/aibuzz/predict-the-genetic-disorders-datasetof-genomes/data

Objective: Multi-target classification of primary disorders and secondary subclasses.

Key Technical Features


Data Processing: Implemented consistent encoding for genomic features to ensure model stability during real-time inference. 

Machine Learning: Built a predictive model focused on high-accuracy classification of genetic markers.


Interactive UI: Developed a front-end interface using Gradio, allowing users to input sequence data and receive immediate disorder predictions in plain text. 

How to Use

1.Clone the repository and ensure train.csv is in the root directory.

2. Install dependencies: pip install pandas scikit-learn gradio.

3. Run the application: python disease_predictor.py.
