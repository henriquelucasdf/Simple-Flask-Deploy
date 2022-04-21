# Simple-Flask-Deploy
A simple machine learning deploy using Flask and Heroku

## Introduction
On this project, a Machine Learning Pipeline for text classification will be deployed on Heroku. This pipeline receives a news text, performs its preprocessing and classifies it in 5 categories: business, entertainment, politics, sport and tech.

The pipeline was build using sklearn's [Pipeline Class](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and consists on the following steps:
- Text Cleaning: a Custom Sklearn transformer was developed to clean the text
- TF-IDF: The sklearn implementation of TF-IDF
- Dimensionality Reduction: the TruncatedSVD sklearn class 
- Logistic Regression model: a model trained on the BBC Dataset. 

A Webapp was created using Flask and consists in two pages, one for inserting the text to be classified and another for printing the results. The HTML script used in the web page is on the "static" folder and the CSS templates on the "templates" folder. 

**The Deployed WebApp can be accessed on this [link](https://bbc-classification-flask.herokuapp.com/)**

**The training process and more details about this pipeline can be found on this repo:**

[![henriquelucasdf - BBC_MulticlassClassification](https://img.shields.io/static/v1?label=henriquelucasdf&message=BBC_MulticlassClassification&color=blue&logo=github)](https://github.com/henriquelucasdf/BBC_MulticlassClassification "Go to GitHub repo")

## The Inference Pipeline:
Since we are using the sklearn Pipeline Class, the inference for this model became quite simple.

First, we retrieve the input text using the `request` method on Flask. We insert the text into a list, since this is the expected input type of the pipeline. 

```python
input_text = [request.form['input_text']]
```

We load the serialized pipeline using joblib (we need the preprocess_src.py script to load the Custom Sklearn transformer Class we created on the training process).

After that, we use the Pipeline to predict directly on the input text. This pipeline will perform all the necessary transformations, as done in the training. The output will be a probability array.

```python
yhat_prob = pipeline.predict_proba(X=input_text)[0]
```

We retrieve the maximum probability and its class (as an index):

```python
probability = f'{100*max(yhat_prob):.2f}%'
yhat = np.argmax(yhat_prob)
```

We format the class (label) into the respective category and print the results in the "estimate" page. 

## The Deploy:

To deploy the app on Heroku, we need 4 files: 
1. **Procfile**: to tell which commands need to be run by heroku
2. **requirements.txt**: to include the packages needed for the webapp and the inference pipeline on the deployment environment (don't forget about gunicorn)
3. **runtime.txt**: to specify the python version to be used on Heroku
4. **nltk.txt**: to install the NLTK methods used. In our case, "punkt" and "stopwords".

With the files created on our project repository (already with git), we login in on Heroku CLI:

```
heroku login
```

After the login on the Heroku CLI, we create the app with:

```
heroku apps:create bbc-classification-flask
```

Finally, we add and commit the changes on git and pushes to our app on heroku:
```
git push heroku main
```

## References
- BBC Dataset:
  
    D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.

    Available at this [link](http://mlg.ucd.ie/datasets/bbc.html).
