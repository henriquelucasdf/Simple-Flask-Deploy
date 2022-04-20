# Simple-Flask-Deploy
A simple machine learning deploy using Flask and Heroku

## Introduction
On this project, a Machine Learning Pipeline for text classification will be deployed on Heroku. This pipeline receives a news text, performs its preprocessing and classifies it in 5 categories: business, entertainment, politics, sport and tech.

The pipeline was build using sklearn's [Pipeline Class](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and consists on the following steps:
- Text Cleaning: a Custom Sklearn transformer was developed to clean the text
- TF-IDF: The sklearn implementation of TF-IDF
- Dimensionality Reduction: the TruncatedSVD sklearn class 
- Logistic Regression model: a model trained on the BBC Dataset. 

**The training process and more details about this pipeline can be found on this repo:**

[![henriquelucasdf - BBC_MulticlassClassification](https://img.shields.io/static/v1?label=henriquelucasdf&message=BBC_MulticlassClassification&color=blue&logo=github)](https://github.com/henriquelucasdf/BBC_MulticlassClassification "Go to GitHub repo")

## The Inference Pipeline:

## The Deploy:


## References
- BBC Dataset:
  
    D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.

    Available at this [link](http://mlg.ucd.ie/datasets/bbc.html).
