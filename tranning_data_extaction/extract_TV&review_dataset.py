import json
import numpy as np
import pandas as pd
from types import SimpleNamespace

df = pd.read_csv('TV_info.csv',  encoding='utf-8')
df_sub=df[df.category=="ElectronicsTelevision &amp; VideoTelevisionsLED &amp; LCD TVs"]
df_sub_2=df[df.category=="ElectronicsTelevision & VideoTelevisionsLED & LCD TVs"]
smart_tv_list=df_sub.asin.to_list()
all_tv_list=df_sub_2.asin.to_list()

label_smart=[]
label_smart_=[]
review_smart=[]
asin_smart=[]

label_all=[]
label_all_=[]
review_all=[]
asin_all=[]

with open('Electronics.json',"r") as f:
    largeReviewData = f.readlines()
print('files are open')
for point in largeReviewData:    #large loop
    pp=json.loads(point)
    if "reviewText" in pp:
        if pp['asin'] in smart_tv_list:
            print(pp['overall'])
            label_smart.append(int(pp['overall']))
            label_smart_.append('__label__'+str(int(pp['overall'])))
            review_smart.append(pp['reviewText'])
            asin_smart.append(pp['asin'])
        elif pp['asin'] in all_tv_list:
            label_all.append(int(pp['overall']))
            label_all_.append('__label__'+str(int(pp['overall'])))
            review_all.append(pp['reviewText'])
            asin_all.append(pp['asin'])

smartTVdata=pd.DataFrame(label_smart,columns=['Star'])
smartTVdata['Label']=label_smart_
smartTVdata['Text']=review_smart
smartTVdata['asin']=asin_smart

allTVdata=pd.DataFrame(label_all,columns=['Star'])
allTVdata['Label']=label_all_
allTVdata['Text']=review_all
allTVdata['asin']=asin_all

frames = [smartTVdata, allTVdata]

result = pd.concat(frames)

smartTVdata.to_csv('smartTV_review.csv')

smartTVdata = smartTVdata.sample(frac=1).reset_index(drop=True)
x=smartTVdata[['Label', 'Text']].to_numpy()
np.savetxt('train_smart.train', x,fmt='%s', delimiter=',', newline='\n',encoding="utf-8")



result.to_csv('allTV_review.csv')

result=result.sample(frac=1).reset_index(drop=True)
x=result[['Label', 'Text']].to_numpy()
np.savetxt('train_all.train', x,fmt='%s', delimiter=',', newline='\n',encoding="utf-8")


        
