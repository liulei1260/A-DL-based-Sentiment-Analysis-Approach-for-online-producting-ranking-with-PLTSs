from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
import scipy
import scipy.io
from scipy.special import softmax



#############transformers load pre-trainned model
#model = ClassificationModel('bert', 'outputs/checkpoint-679-epoch-1', num_labels=5)

data_len='2600'
model_name='transformers'

def get_pandas_df(filename):
    '''
    Load the data into Pandas.DataFrame object
    This will be used to convert data to torchtext object
    '''
    with open(filename, encoding="utf-8") as datafile:
        data = [line.strip().split(',', maxsplit=1) for line in datafile]
        data_text = list(map(lambda x: x[1], data))
        data_label = list(map(lambda x: int(x[0].strip()[-1])-1, data))

    full_df = pd.DataFrame({"text":data_text, "labels":data_label})
    return full_df

def f1_multiclass(labels, preds):
      return f1_score(labels, preds, average=None)

def precision_multiclass(labels, preds):
      return precision_score(labels, preds, average=None)

def recall_multiclass(labels, preds):
      return recall_score(labels, preds, average=None)

def predict(test):
    predictions, raw_outputs=model.predict(test['text'])
    pre_label=predictions[0]
    pre_probality=raw_outputs[0]
    return pre_label, pre_probality

#train_file = './data/train_pre.train'
#test_file = './data/test_pre.test'
#final_test='./data/final_test_pre.test'

#train_file = './data/train.train'
#test_file = './data/test.test'
#final_test='./data/final_test.test'

train_file = '../all_data/train.train'
test_file = '../all_data/test.test'
final_test='../all_data/final_test.test'

df_train=get_pandas_df(train_file)
df_test=get_pandas_df(test_file)
df_final=get_pandas_df(final_test)

print(df_train.head())
model_args = ClassificationArgs(num_train_epochs=1,overwrite_output_dir=True)
model = ClassificationModel('bert', 'bert-base-cased',args=model_args,num_labels=3)

model.train_model(df_train)
result, model_outputs, wrong_predictions=model.eval_model(df_train,acc=accuracy_score)
train_acc=result['acc']
print("====")
print ('Test Accuracy: {:.4f}'.format(train_acc))
print("====")


result, model_outputs, wrong_predictions = model.eval_model(df_test,f1=f1_multiclass, acc=accuracy_score,recall=recall_multiclass,precision=precision_multiclass)
probabilities = softmax(model_outputs, axis=1)

test_acc = result['acc']
test_f1=result['f1']  #,average='macro'
test_precision=result['precision']#,average='macro'
test_recall=result['recall']#,average='macro'

print("====")
print ('Test Accuracy: {:.4f}'.format(test_acc))
print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(test_precision)))
print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(test_recall)))
print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(test_f1)))
#print (test_f1)
#print (test_precision)
#print (test_recall)
print("====")

results, raw_outputs = model.predict(list(df_final['text']))
all_preds = softmax(raw_outputs, axis=1)



final_acc = accuracy_score(df_final['labels'].tolist(),results)
final_f1=f1_score(df_final['labels'].tolist(),results,average=None)  #,average='macro'
final_precision=precision_score(df_final['labels'].tolist(),results,average=None)#,average='macro'
final_recall=recall_score(df_final['labels'].tolist(),results,average=None)#,average='macro'


print ('TV Test Accuracy: {:.4f}'.format(final_acc))
print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(final_precision)))
print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(final_recall)))
print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(final_f1)))
#print (final_f1)
#print (final_precision)
#print (final_recall)

scipy.io.savemat('./results_'+data_len+'_'+model_name+'.mat', mdict={'train_acc': 0, 
'val_acc': 0, 'test_acc': test_acc,'test_f1':test_f1,'test_precision':test_precision,'test_recall':test_recall,
'final_acc': final_acc,'final_f1':final_f1,'final_precision':final_precision,'final_recall':final_recall,'pre_probality':all_preds,'pre_label':results})
#predictions, raw_outputs = model.predict(["I bought this to save space in the bedroom.  Standard broadcasts look good, and with an upscaling DVD player hooked up even older TV shows look good.  For the price, a pretty good buy.  P.S. - Soyo is the manufacturer, though they are sold via GoVideo in the US."])
