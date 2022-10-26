import json
import numpy as np
import pandas as pd
from types import SimpleNamespace

df = pd.read_csv('TV_info.csv',  encoding='utf-8')
df_sub=df[df.category=="ElectronicsTelevision &amp; VideoTelevisionsLED &amp; LCD TVs"]
df_sub_=df[df.category=="ElectronicsTelevision & VideoTelevisionsLED & LCD TVs"]
print(len(df_sub))
print(len(df_sub_))
xxx=[df_sub,df_sub_]
df_sub_2 = pd.concat(xxx)
print(len(df_sub_2))
all_tv_list=df_sub_2.asin.to_list()


label_all=[]
label_all_=[]
review_all=[]
asin_all=[]
time_all=[]

with open('Electronics.json',"r") as f:
    largeReviewData = f.readlines()
print('files are open')
for point in largeReviewData:    #large loop
    pp=json.loads(point)
    if "reviewText" in pp and pp['asin'] in all_tv_list and int(pp['reviewTime'].strip()[-4:])>2009:
            label_all.append(int(pp['overall']))
            label_all_.append('__label__'+str(int(pp['overall'])))
            review_all.append(pp['reviewText'])
            asin_all.append(pp['asin'])
            time_all.append(int(pp['reviewTime'].strip()[-4:]))



allTVdata=pd.DataFrame(label_all,columns=['Star'])
allTVdata['Label']=label_all_
allTVdata['Text']=review_all
allTVdata['asin']=asin_all
allTVdata['time']=time_all



result = allTVdata
print(len(result))
result.to_csv('allTV_review_2010.csv')

result=result.sample(frac=1).reset_index(drop=True)
x=result[['Label', 'Text']].to_numpy()
np.savetxt('train_all_2010.train', x,fmt='%s', delimiter=',', newline='\n',encoding="utf-8")