import pandas as pd
import nltk
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
porter = PorterStemmer()

#nltk.download('stopwords')
def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

stop = stopwords.words('english')
basepath='./'

train_data=pd.read_csv(os.path.join(basepath,'train.csv'), encoding='utf-8')
test_data=pd.read_csv(os.path.join(basepath,'test.csv'), encoding='utf-8')
final_test=pd.read_csv(os.path.join(basepath,'feature_texts.csv'), encoding='utf-8')

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__alpha':[0,0.5,1]
               }
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', MultinomialNB())])


gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

X_train = train_data.loc[:, 'text'].values
y_train = train_data.loc[:, 'labels'].values

X_test=test_data.loc[:, 'text'].values
y_test=test_data.loc[:, 'labels'].values

X_final = final_test.loc[:, 'Text'].values
y_final = final_test.loc[:, 'Star'].values

gs_lr_tfidf.fit(X_train, y_train)
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_

print(clf.score(X_test, y_test))

print(clf.score(X_final, y_final))