import pandas as pd
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingRegressor

races_df = pd.read_csv('./data//races.csv',delimiter=",",header=0, index_col='race_id')
runs_df = pd.read_csv('./data//runs.csv', delimiter=",", header=0)

print(races_df.shape)
print(runs_df.shape)
pd.set_option('display.max_columns', None)
df=runs_df
df=df.drop('won',axis=1)
df=df.drop('lengths_behind',axis=1)
df=df.drop('horse_rating',axis=1)
df=df.drop('horse_gear',axis=1)
df=df.drop('draw',axis=1)
df=df.drop('position_sec1',axis=1)
df=df.drop('position_sec2',axis=1)
df=df.drop('position_sec3',axis=1)
df=df.drop('position_sec4',axis=1)
df=df.drop('position_sec5',axis=1)
df=df.drop('position_sec6',axis=1)
df=df.drop('behind_sec1',axis=1)
df=df.drop('behind_sec2',axis=1)
df=df.drop('behind_sec3',axis=1)
df=df.drop('behind_sec4',axis=1)
df=df.drop('behind_sec5',axis=1)
df=df.drop('behind_sec6',axis=1)
df=df.drop('time1',axis=1)
df=df.drop('time2',axis=1)
df=df.drop('time3',axis=1)
df=df.drop('time4',axis=1)
df=df.drop('time5',axis=1)
df=df.drop('time6',axis=1)
df=df.drop('win_odds',axis=1)
df=df.drop('place_odds',axis=1)
print(races_df.head(1))
print(runs_df.columns)
df = pd.merge(df,races_df[['venue','config','surface','distance','going']],on='race_id', how='left')
print(runs_df.head(3))

horse_tot_race=runs_df.groupby(['horse_id'])['result'].apply(lambda x: (x).sum()).reset_index(name='horse_tot_race')
df=pd.merge(df,horse_tot_race,on='horse_id',how='left')
horse_tot_place=runs_df.groupby(['horse_id'])['result'].apply(lambda x: (x <=3).sum()).reset_index(name='horse_tot_place')
df=pd.merge(df,horse_tot_place,on='horse_id',how='left')

jockey_tot_race=runs_df.groupby(['jockey_id'])['result'].apply(lambda x: (x).sum()).reset_index(name='jockey_tot_race')
df=pd.merge(df,jockey_tot_race,on='jockey_id',how='left')
jockey_tot_place=runs_df.groupby(['jockey_id'])['result'].apply(lambda x: (x <=3).sum()).reset_index(name='jockey_tot_place')
df=pd.merge(df,jockey_tot_place,on='jockey_id',how='left')

trainer_tot_race=runs_df.groupby(['trainer_id'])['result'].apply(lambda x: (x).sum()).reset_index(name='trainer_tot_race')
df=pd.merge(df,trainer_tot_race,on='trainer_id',how='left')
trainer_tot_place=runs_df.groupby(['trainer_id'])['result'].apply(lambda x: (x <=3).sum()).reset_index(name='trainer_tot_place')
df=pd.merge(df,trainer_tot_place,on='trainer_id',how='left')

#new horse features
df['horse_place_perc']=df['horse_tot_place']/df['horse_tot_race']

#new jockey features
df['jockey_place_perc']=df['jockey_tot_place']/df['jockey_tot_race']

#new trainer features
df['trainer_place_perc']=df['trainer_tot_place']/df['trainer_tot_race']

print(df.columns)

df=df.drop('horse_tot_place',axis=1)
df=df.drop('horse_tot_race',axis=1)
df=df.drop('horse_id',axis=1)


df=df.drop('trainer_tot_place',axis=1)
df=df.drop('trainer_tot_race',axis=1)
df=df.drop('trainer_id',axis=1)


df=df.drop('jockey_tot_place',axis=1)
df=df.drop('jockey_tot_race',axis=1)
df=df.drop('jockey_id',axis=1)

df=df.drop('horse_no',axis=1)
df = df.sort_index(axis=1, ascending=True)
temp_cols=df.columns.tolist()
index=df.columns.get_loc("finish_time")
new_cols=temp_cols[index:index+1] + temp_cols[0:index] + temp_cols[index+1:]
df=df[new_cols]

print(df.shape)
print(df.head(2))

df.isna().sum()
df=df.dropna()
print(df.info)

# encode ordinal columns: config,going
config_encoder = preprocessing.OrdinalEncoder()
df['config'] = config_encoder.fit_transform(df['config'].values.reshape(-1, 1))

going_encoder = preprocessing.OrdinalEncoder()
df['going'] = going_encoder.fit_transform(df['going'].values.reshape(-1, 1))

# encode nominal column: venue, horse_country, horse_type
venue_encoder = preprocessing.LabelEncoder()
df['venue'] = venue_encoder.fit_transform(df['venue'])

horse_country_encoder = preprocessing.LabelEncoder()
df['horse_country'] = horse_country_encoder.fit_transform(df['horse_country'])

horse_type_encoder = preprocessing.LabelEncoder()
df['horse_type'] = horse_type_encoder.fit_transform(df['horse_type'])

print(df.head(10))
print(df.shape)

corr=df.corr()
print(corr.style.background_gradient(cmap='coolwarm').set_precision(2))

#1st database with race_id=0 , 2nd race_id=3
df_0=df[df.race_id==0]
df_1=df[df.race_id==1]
df_3=df[df.race_id==3]

print(df_0.shape)
df=df[df.race_id!=0]
df=df[df.race_id!=1]
df=df[df.race_id!=3]
print(df.shape)

df=df.drop('race_id',axis=1)
df=df.drop('result',axis=1)

df_0_ML=df_0
df_0_ML=df_0_ML.drop('race_id',axis=1)
df_0_ML=df_0_ML.drop('result',axis=1)

df_1_ML=df_1
df_1_ML=df_1_ML.drop('race_id',axis=1)
df_1_ML=df_1_ML.drop('result',axis=1)

df_3_ML=df_3
df_3_ML=df_3_ML.drop('race_id',axis=1)
df_3_ML=df_3_ML.drop('result',axis=1)

X = df[df.columns[1:]]
ss = preprocessing.StandardScaler()
X = pd.DataFrame(ss.fit_transform(X),columns = X.columns)
y_time = df.finish_time
#We are not going to use those db now
X_0= df_0_ML[df_0_ML.columns[1:]]
ss = preprocessing.StandardScaler()
X_0= pd.DataFrame(ss.fit_transform(X_0),columns = X_0.columns)
X_1 = df_1_ML[df_1_ML.columns[1:]]
ss = preprocessing.StandardScaler()
X_1 = pd.DataFrame(ss.fit_transform(X_1),columns = X_1.columns)
X_3 = df_1_ML[df_1_ML.columns[1:]]
ss = preprocessing.StandardScaler()
X_3 = pd.DataFrame(ss.fit_transform(X_3),columns = X_3.columns)

# split data into train and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_time, train_size=0.75, test_size=0.25, random_state=1)
print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)
lr= linear_model.LinearRegression()
cv= cross_val_score(lr,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
knn= KNeighborsRegressor(n_neighbors=4)
cv= cross_val_score(knn,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
tree=DecisionTreeRegressor(random_state=1)
cv= cross_val_score(tree,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
rf=RandomForestRegressor(random_state=1)
cv= cross_val_score(rf,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
#VotingRegressor
voting_rg=VotingRegressor(estimators=[('lr',lr),('rf',rf)])
cv= cross_val_score(voting_rg,X_train,y_train,cv=5)
print(cv)
print(cv.mean())
#linear regression
lr.fit(X_train,y_train)
y_lr=lr.predict(X_test)
print("Linear Regression results:")
print("R^2",metrics.r2_score(y_test,y_lr))
print("Mean Absolute Error", metrics.mean_absolute_error(y_test,y_lr))
print("Mean Squared Error", metrics.mean_squared_error(y_test,y_lr))
print("Root Mean Squared Error",np.square(metrics.mean_squared_error(y_test,y_lr)))
#random forest
rf.fit(X_train,y_train)
y_rf=rf.predict(X_test)
print("Random Forest results:")
print("R^2",metrics.r2_score(y_test,y_rf))
print("Mean Absolute Error", metrics.mean_absolute_error(y_test,y_rf))
print("Mean Squared Error", metrics.mean_squared_error(y_test,y_rf))
print("Root Mean Squared Error",np.square(metrics.mean_squared_error(y_test,y_rf)))

#VotingRegressor
voting_rg.fit(X_train,y_train)
y_voting_rg=voting_rg.predict(X_test)
print("Voting Regressor results:")
print("R^2",metrics.r2_score(y_test,y_voting_rg))
print("Mean Absolute Error", metrics.mean_absolute_error(y_test,y_voting_rg))
print("Mean Squared Error", metrics.mean_squared_error(y_test,y_voting_rg))
print("Root Mean Squared Error",np.square(metrics.mean_squared_error(y_test,y_voting_rg)))

df_0['pred']=voting_rg.predict(X_0)
df_0 = df_0[['finish_time','pred','result']]
df_0['result_pred'] = df_0['pred'].rank(ascending=True).astype(int)
print(df_0)

df_3['pred']=voting_rg.predict(X_3)
df_3 = df_3[['finish_time','pred','result']]
df_3['result_pred'] = df_3['pred'].rank(ascending=True).astype(int)
print(df_3)

df_1['pred']=voting_rg.predict(X_1)
df_1 = df_1[['finish_time','pred','result']]
df_1['result_pred'] = df_1['pred'].rank(ascending=True).astype(int)
print(df_1)

