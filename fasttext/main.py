import fasttext
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
from config import Config
import scipy
import scipy.io


#cd code/review_folder/fastText/data
#cat train.train | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > train_pre.train
#cat test.test | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > test_pre.test
#cat final_test.test | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > final_test_pre.test

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

def get_pandas_df(filename):
    '''
    Load the data into Pandas.DataFrame object
    This will be used to convert data to torchtext object
    '''
    with open(filename, encoding="utf-8") as datafile:
        data = [line.strip().split(',', maxsplit=1) for line in datafile]
        data_text = list(map(lambda x: x[1], data))
        data_label = list(map(lambda x: x[0], data))

    full_df = pd.DataFrame({"text":data_text, "label":data_label})
    return full_df

def predict(test):
    a,b=model.predict(test['text'])
    c=list(a)
    return c[0].strip()[-1]

def trans_labels(test):
    y=[]
    for i in test:
      y.append(i.strip()[-1])
    return y

#train_file = './data/train.train'
#test_file = './data/test.test'
#final_test='./data/final_test.test'

train_file = '../all_data/train.train'
test_file = '../all_data/test.test'
final_test='../all_data/final_test.test'

model = fasttext.train_supervised(input=train_file, lr=1.0, epoch=30, wordNgrams=2)  # lr=1.0 epoch=25

print('train file')
df_train=get_pandas_df(train_file)
y_pre_train=df_train.apply(predict,axis=1)
y_label_train=trans_labels(df_train["label"].tolist())

print ('Test Accuracy: {:.4f}'.format(accuracy_score(y_label_train, y_pre_train)))

print('test file')
print(model.test_label(test_file))


df_test=get_pandas_df(test_file)

y_pre=df_test.apply(predict,axis=1)
y_label=trans_labels(df_test["label"].tolist())


test_acc = accuracy_score(y_label, y_pre)
test_f1=f1_score(y_label, y_pre,average=None)  #,average='macro'
test_precision=precision_score(y_label, y_pre,average=None)#,average='macro'
test_recall=recall_score(y_label, y_pre,average=None)#,average='macro'

df_final=get_pandas_df(final_test)

print ('Test Accuracy: {:.4f}'.format(test_acc))
#print (test_precision)
#print (test_recall)
#print (test_f1)
print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(test_precision)))
print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(test_recall)))
print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(test_f1)))


df_final_test=get_pandas_df(final_test)
y_pre_f=df_final_test.apply(predict,axis=1)
y_label_f=trans_labels(df_final_test["label"].tolist())

final_acc = accuracy_score(y_label_f, y_pre_f)
final_f1=f1_score(y_label_f, y_pre_f,average=None)  #,average='macro'
final_precision=precision_score(y_label_f, y_pre_f,average=None)#,average='macro'
final_recall=recall_score(y_label_f, y_pre_f,average=None)#,average='macro'

print('final TV file')
print(model.test_label(final_test))

print ('TV Test Accuracy: {:.4f}'.format(final_acc))
print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(final_precision)))
print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(final_recall)))
print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(final_f1)))
#print (final_precision)
#print (final_recall)
#print (final_f1)


def predict_pro(row):
    return model.predict(row['text'])


xxx=df_final_test.apply(predict_pro,axis=1)
all_preds=[]
for t in xxx:
    Y=list(t[0])
    X=t[1]
    Z=[x for _,x in sorted(zip(Y,X))]
    all_preds.append(Z)

config = Config()
results = list(map(int, y_pre_f))
scipy.io.savemat('./results_'+config.data_len+'_'+config.model_name+'.mat', mdict={'train_acc': 0, 
'val_acc': 0, 'test_acc': test_acc,'test_f1':test_f1,'test_precision':test_precision,'test_recall':test_recall,
'final_acc': final_acc,'final_f1':final_f1,'final_precision':final_precision,'final_recall':final_recall,'pre_probality':all_preds,'pre_label':results})
model.save_model('./'+config.data_len+'_'+config.model_name+'.ftz')


#print_results(*model.test(test_file))

print('d=o=n=e=======')

#print_results(*model.test(final_test))

