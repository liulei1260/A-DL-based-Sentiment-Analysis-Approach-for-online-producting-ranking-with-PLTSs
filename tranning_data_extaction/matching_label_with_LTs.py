import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def xxx():
  file_name='allTV_review.csv'
  
  df = pd.read_csv(file_name, encoding='utf-8')
  for i, row in df.iterrows():
      ifor_val = row['Text']
      if "\n" in ifor_val:
          ifor_val=ifor_val.replace("\n", " ")
          df.loc[i, 'Text'] = ifor_val
      if i!=row['index']:
          df.loc[i, 'index']=i
  df.to_csv('extracted_data\\alldata_processed.csv')
  class_1=  df[df.Star==1]
  class_2 = df[df.Star == 2]
  class_3 = df[df.Star == 3]
  class_4 = df[df.Star == 4]
  class_5 = df[df.Star == 5]
  X, y = class_1.iloc[:, :].values, class_1.iloc[:, 1].values
  X_train0, X_test0, y_train0, y_test0 = train_test_split(X, y, test_size=1500, random_state=27)
  X, y = class_2.iloc[:, :].values, class_2.iloc[:, 1].values
  X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=1500, random_state=27)
  X, y = class_3.iloc[:, :].values, class_3.iloc[:, 1].values
  X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=1500, random_state=27)
  X, y = class_4.iloc[:, :].values, class_4.iloc[:, 1].values
  X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=1500, random_state=27)
  X, y = class_5.iloc[:, :].values, class_5.iloc[:, 1].values
  X_train4, X_test4, y_train4, y_test4 = train_test_split(X, y, test_size=1500, random_state=27)
  X_train=np.concatenate((X_train0, X_train1,X_train2,X_train3,X_train4), axis=0)
  X_test=np.concatenate((X_test0, X_test1,X_test2,X_test3,X_test4), axis=0)
  y_train=np.concatenate((y_train0, y_train1,y_train2,y_train3,y_train4), axis=0)
  y_test=np.concatenate((y_test0, y_test1,y_test2,y_test3,y_test4), axis=0)
  train_df = pd.DataFrame(X_train[:], columns=["index","Star","Label","Text","asin"])
  test_df=pd.DataFrame(X_test[:], columns=["index","Star","Label","Text","asin"])
  train_df.to_csv('extracted_data\\train_dataset.csv')
  test_df.to_csv('extracted_data\\test_dataset.csv')

file_path='alldata_processed_cleaned.csv'
df = pd.read_csv(file_path, encoding='utf-8')
for i, row in df.iterrows():
    if row['Star']==2:
        df.loc[i, 'Star'] = 2;
        df.loc[i, 'Label']='__label__2'
    elif row['Star']==3:
        df.loc[i, 'Star'] = 2;
        df.loc[i, 'Label']='__label__2'
    elif row['Star']==4:
        df.loc[i, 'Star'] = 2;
        df.loc[i, 'Label']='__label__2'
    elif row['Star'] == 5:
        df.loc[i, 'Star'] = 3;
        df.loc[i, 'Label'] = '__label__3'
df.to_csv('fangan4.csv',index=False)
class_1 = df[df.Star == 1]
class_2 = df[df.Star == 2]
class_3 = df[df.Star == 3]
len_list = [len(class_1), len(class_2), len(class_3)]
print(len_list)
X, y = class_1.iloc[:, :].values, class_1.iloc[:, 1].values
X_train0, X_test0, y_train0, y_test0 = train_test_split(X, y, test_size=1500, random_state=27)
X, y = class_2.iloc[:, :].values, class_2.iloc[:, 1].values
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=1500, random_state=27)
X, y = class_3.iloc[:, :].values, class_3.iloc[:, 1].values
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=1500, random_state=27)

X_train = np.concatenate((X_train0, X_train1, X_train2), axis=0)
X_test = np.concatenate((X_test0, X_test1, X_test2), axis=0)
#y_train = np.concatenate((y_train0, y_train1, y_train2, y_train3, y_train4), axis=0)
#y_test = np.concatenate((y_test0, y_test1, y_test2, y_test3, y_test4), axis=0)
train_df = pd.DataFrame(X_train[:], columns=["index", "Star", "Label", "Text", "asin"])
test_df = pd.DataFrame(X_test[:], columns=["index", "Star", "Label", "Text", "asin"])
train_df.to_csv('fangan4_train.csv',index=False)
test_df.to_csv( 'fangan4_test.csv',index=False)
print('done')