# A-DL-based-Sentiment-Analysis-Approach-for-online-producting-ranking-with-PLTSs
Probabilities linguistic term set (PLTS) is an efficient tool to represent sentimental intensities hidden in unstructured text reviews and the sentimental intensities are useful for multi-criteria online product ranking. We propose a deep learning-based sentiment analysis approach to produce PLTSs from online product reviews so as to ranking online products. A natural language processing-based method is firstly applied to extract product features and corresponding feature texts from online reviews. Then, the state-of-the-art deep learning-based models are implemented to conduct the sentiment classification for online product/feature review texts. To ensure classification accuracy, we propose an experimental matching mechanism to identify the level of sentiment tendency for all rating labels of a review dataset and then match each label with the most appropriate linguistic term. 

# DL based appraoch
A Pytorch implementation of deep model based approach for generating PLTSs for online reviews

# Implemented Models
TextCNN, TextRNN, CharNN, Transformers, fastText and Seq2Seq.
We also implemented some machine learning based-models as a comparison: XGBoost, SVM, KNN, Naive Bayes.

# How to use
1. Modify the data path and parameter settings as needed
2. Use this command to train and test: python3 main.py
   i.e: python3 main.py
# Dataset in experiments
The dataset we used in our experiments can be reviewed and downloaed from https://www.dropbox.com/sh/z0uajpyshew2owg/AAAb6B8D0FyM7I3K_i70LmY4a?dl=0, the origional data is derived from https://nijianmo.github.io/amazon/index.html

# Contact
zixuxilan@gmail.com  limaomao.maolin@gmail.com


