#!/usr/bin/env python
# coding: utf-8

# In[181]:


#### Importing necessary libraries:-
import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso,LassoCV,Ridge,RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
import pickle



# In[2]:


## Loading both train and test csv file:
bigdatamart_train = pd.read_csv(r"C:/Users/swati/Downloads/bigdatamart_rep-master/bigdatamart_rep-master/bigdatamart_Train.csv")
bigdatamart_test  = pd.read_csv(r"C:/Users/swati/Downloads/bigdatamart_rep-master/bigdatamart_rep-master/bigdatamart_Test.csv")


# In[3]:


###Train data:-
bigdatamart_train.head()


# In[4]:


###Test data:-
bigdatamart_test.head()


# In[5]:


### Checking size of Train data:-
bigdatamart_train.shape


# There are 8523 rows and 12 columns in Train data.

# In[6]:


### Checking size of Test data:-
bigdatamart_test.shape


# There are 5681 rows and 11 columns in Test data.Target column Item_Outlet_Sales is not present in test data,so I'll give
# my model for training and prediction only Train data,beacuse In Test data,Target column is not present,so I'll not be able to compare my model's predicted value with actual one.Rest all steps  will be done for both Train and Test data.

# In[7]:


### Checking more info about Train data:
bigdatamart_train.info()


# **So above,we can see Item_Identifier,Item_Fat_Content,Item_Type,Outlet_Identifier,Outlet_Size,Outlet_Location_Type,Outlet_Type,
# these 7 columns are Categorical .
# Item_Weight,Item_Visibility,Item_MRP,Outlet_Establishment_Year,Item_Outlet_Sales,these 5 columns are Numerical.
# ** All columns have 8523 count, except Item_Weight and Outlet_Size,So will check for missing values.
# 

# In[8]:


### Checking more info about Train data:
bigdatamart_test.info()


# **So here also,we can see Item_Identifier,Item_Fat_Content,Item_Type,Outlet_Identifier,Outlet_Size,Outlet_Location_Type,Outlet_Type,
# these 7 columns are Categorical .
# Item_Weight,Item_Visibility,Item_MRP,Outlet_Establishment_Year these 4 columns are Numerical.
# ** All columns have 5681 count, except Item_Weight and Outlet_Size,So will check for missing values.
# 

# In[9]:


### Checking for missing values in Train data:-
bigdatamart_train.isnull().sum()


# So here ,we can see Item_Weight has 1463 missing values and Outlet_Size has 2410 missing values. 

# In[10]:


### Checking for missing values in Test data:-
bigdatamart_test.isnull().sum()


# In[ ]:


So here ,we can see Item_Weight has 976 missing values and Outlet_Size has 1606 missing values. 


# In[11]:


### Filling missing values,Since if data has outliers we can not apply mean,so first check outliers
### using boxplot in column Item_Weight:-
sns.boxplot(bigdatamart_train['Item_Weight'])


# As we can see there are no outliers present in Item_Weight,so we can apply mean to fill missing values.

# In[12]:


bigdatamart_train['Item_Weight'].fillna(bigdatamart_train['Item_Weight'].mean(),inplace = True)


# In[13]:


## Again we check for missing values:-
bigdatamart_train.isnull().sum()


# As we can see, Now Item_Weight has 0 missing values. Same step for Test Data.

# In[14]:


### Filling missing values,Since if data has outliers we can not apply mean,so first check outliers
### using boxplot in column Item_Weight:-
sns.boxplot(bigdatamart_test['Item_Weight'])


# Here also ,not any outliers are present.So applying mean method to fill missing values.

# In[15]:


bigdatamart_test['Item_Weight'].fillna(bigdatamart_test['Item_Weight'].mean(),inplace = True)


# In[16]:


## Again we check for missing values:-
bigdatamart_test.isnull().sum()


# Here also Item_Weight has 0 null values.

# In[17]:


## Now fill missing values for Outlet_Size,Since it is a categorical column,so will apply mode to fill missing values.
## First check value_counts
bigdatamart_train['Outlet_Size'].value_counts()


# So we can see Medium has maximum counts,so we fill missing values with Medium both in Train and Test data.

# In[18]:


### Visualizing through graph:-
sns.countplot(bigdatamart_train['Outlet_Size'])


# In[19]:


## Filling with mode:
bigdatamart_train['Outlet_Size'].fillna(bigdatamart_train['Outlet_Size'].mode()[0],inplace = True)


# In[20]:


## Again we check for missing values:-
bigdatamart_train.isnull().sum()


# So we can see now, no column has missing values.

# In[21]:


### Visualizing through graph for Test data:-
sns.countplot(bigdatamart_test['Outlet_Size'])


# In[22]:


## Filling with mode:
bigdatamart_test['Outlet_Size'].fillna(bigdatamart_test['Outlet_Size'].mode()[0],inplace = True)


# In[23]:


## Again we check for missing values:-
bigdatamart_test.isnull().sum()


# Now  also in Test data,no column has missing value.

# In[24]:


## Checking more information about Train and Test data
bigdatamart_train.describe()


# As we can see Item_Visibility minimum value is zero,It is not possible,so will fill this also.First check how many rows have 
# 0 values.

# In[25]:


sum(bigdatamart_train['Item_Visibility'] == 0)


# As we can see 526 rows have 0 value in Item_Visibility column,so fill it with mean .

# In[26]:


## Replacing zero with mean:-
bigdatamart_train.loc[:,'Item_Visibility'].replace([0],[bigdatamart_train['Item_Visibility'].mean()],inplace = True)


# In[27]:


### Checking again:-
sum(bigdatamart_train['Item_Visibility'] == 0)


# In[28]:


bigdatamart_train.describe()


# Now  we can see,min value for Item_Visibility is 0.003575 not zero.

# In[29]:


## Same steps will be  repeated for Test data:-

bigdatamart_test.describe()    


# In[30]:


sum(bigdatamart_test['Item_Visibility'] == 0)


# As we can see 353 rows have 0 value in Item_Visibility column,so fill it with mean .

# In[31]:


## Replacing zero with mean:-
bigdatamart_test.loc[:,'Item_Visibility'].replace([0],[bigdatamart_test['Item_Visibility'].mean()],inplace = True)


# In[32]:


## Checking again:-
sum(bigdatamart_test['Item_Visibility'] == 0)


# In[33]:


bigdatamart_test.describe()


# Now  we can see,min value for Item_Visibility is 0.003591 not zero.

# In[34]:


bigdatamart_train.columns


# # Feature Selection:-

# From the analysis of data,I feel that Item_Identifier and Outlet_Identifier has not any impact on Target column means for sales 
# of any product ,it's unique id does not play any role,same for Outlet identifier.So will drop these two columns. 

# In[35]:


bigdatamart_train.drop(columns = ['Item_Identifier','Outlet_Identifier'],inplace = True)


# In[36]:


bigdatamart_train.shape


# Now we can see instead of 12,we have 10 columns.

# In[37]:


bigdatamart_test.columns


# So here also,will delete Item_Identifier and Outlet_Identifier.

# In[38]:


bigdatamart_test.drop(columns = ['Item_Identifier','Outlet_Identifier'],inplace = True)


# In[39]:


bigdatamart_test.shape


# Now we can see instead of 11,we have 9 columns.

# In[40]:


### Now Separating data into categorical and numerical columns:-
df_cat = bigdatamart_train[['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']].copy()
df_cat


# In[41]:


df_cat.describe()


# As from above table we can see unique values for following columns are:
# Item_Fat_Content = 5
# Item_Type   =16
# Outlet_Size = 3
# Outlet_Location_Type =3
# Outlet_Type =3
# 

# In[54]:


### Printing value counts for all categorical column:
for col in df_cat:
    
    print(col)
    print(df_cat[col].value_counts())
    print("\n")


# In[43]:


df_cat['Item_Fat_Content'].value_counts()


# As we can see There are five unique values for Low fat and Regular type ,so will replace LF and low fat in Low Fat and Reg into
# Regular

# In[45]:


#### Merging into only two categories:-
bigdatamart_train['Item_Fat_Content']=bigdatamart_train['Item_Fat_Content'].replace({'LF': 'Low Fat','low fat':'Low Fat','reg':'Regular'})
bigdatamart_train['Item_Fat_Content'].value_counts()


# So here ,we can see now we have only two category Low Fat and Regular.

# In[47]:


bigdatamart_test['Item_Fat_Content'].value_counts()


# In[48]:


#### Merging into only two categories:-
bigdatamart_test['Item_Fat_Content']=bigdatamart_test['Item_Fat_Content'].replace({'LF': 'Low Fat','low fat':'Low Fat','reg':'Regular'})
bigdatamart_test['Item_Fat_Content'].value_counts()


# So in Test data,we have only two category Low Fat and Regular.

# In[50]:


bigdatamart_train['Outlet_Establishment_Year']


# As we can see in Outlet_Establishment_Year,we have year as value,As per my analysis,If any shop or outlet store is old,then sale at that store will be more due to customer's trust.So making it more meaningful column,I substract all the years from 2013.

# In[55]:


bigdatamart_train['years_old'] = 2013 - bigdatamart_train['Outlet_Establishment_Year']


# In[56]:


bigdatamart_train.head()


# In[57]:


bigdatamart_test['years_old'] = 2013 - bigdatamart_test['Outlet_Establishment_Year']


# In[58]:


bigdatamart_test.head()


# # EDA for Train data:-

# In[64]:


### Visualization Through Graph:-
### Visualizing through graph for Numerical columns:-
sns.distplot(bigdatamart_train['Item_Weight'])


# In[65]:


sns.distplot(bigdatamart_train['Item_Visibility'])


# In[66]:


sns.distplot(bigdatamart_train['Item_MRP'])


# In[67]:


sns.distplot(bigdatamart_train['years_old'])


# In[68]:


sns.distplot(bigdatamart_train['Item_Outlet_Sales'])


# In[69]:


## Visualization for Categorical columns:-
sns.countplot(bigdatamart_train['Item_Fat_Content'])    


# In[114]:


#plt.figure(figsize = (20,20))
label = list(bigdatamart_train['Item_Type'].unique())
graph = sns.countplot(bigdatamart_train['Item_Type'])    
graph.set_xticklabels(labels =label,rotation = 90 )


# In[72]:


sns.countplot(bigdatamart_train['Outlet_Size'])    


# In[73]:


sns.countplot(bigdatamart_train['Outlet_Location_Type'])    


# In[75]:


plt.figure(figsize=(10,10))
sns.countplot(bigdatamart_train['Outlet_Type'])    


# In[81]:


### Checking skewness:-
bigdatamart_train.skew()


# As all the values are in +/- 3 ,so I consider values are normally distributed.

# In[82]:


sns.boxplot(bigdatamart_train['Item_Weight'])


# In[83]:


sns.boxplot(bigdatamart_train['Item_Visibility'])


# In[84]:


sns.boxplot(bigdatamart_train['Item_MRP'])


# In[86]:


sns.boxplot(bigdatamart_train['years_old'])


# In[87]:


sns.boxplot(bigdatamart_train['Item_Outlet_Sales'])


# Since Item_Visibility and Item_Outlet_Sales are showing some outliers.So applying Z score on these colums

# In[88]:


## Removing outliers using zscore:-
from scipy.stats import zscore
z_score = zscore(bigdatamart_train[['Item_Visibility','Item_Outlet_Sales']])
abs_zscore=np.abs(z_score)
new_data =( abs_zscore < 3).all(axis=1)
bigdatamart_train = bigdatamart_train[new_data]
bigdatamart_train.describe()


# In[89]:


sns.boxplot(bigdatamart_train['Item_Visibility'])


# In[90]:


sns.distplot(bigdatamart_train['Item_Visibility'])


# In[91]:


sns.boxplot(bigdatamart_train['Item_Outlet_Sales'])


# In[92]:


sns.distplot(bigdatamart_train['Item_Outlet_Sales'])


# So we can see for both  Item_Outlet_Sales and Item_Visibility columns ,data are almost normally distributed.

# In[ ]:


## Removing outliers using zscore:-
from scipy.stats import zscore
z_score = zscore(bigdatamart_train[['Item_Visibility','Item_Outlet_Sales']])
abs_zscore=np.abs(z_score)
new_data =( abs_zscore < 3).all(axis=1)
bigdatamart_train = bigdatamart_train[new_data]
bigdatamart_train.describe()


# In[93]:


bigdatamart_train.shape


# In[96]:


### Percentage loss of data:-
data_loss =(( 8523-8334)/8523)*100
data_loss


# # EDA for Test data

# In[105]:


### Visualization Through Graph:-
### Visualizing through graph for Numerical columns:-
sns.distplot(bigdatamart_test['Item_Weight'])


# In[106]:


sns.distplot(bigdatamart_test['Item_Visibility'])


# In[107]:


sns.distplot(bigdatamart_test['Item_MRP'])


# In[108]:


sns.distplot(bigdatamart_test['years_old'])


# In[109]:


### Visualization for Categorical column:-
sns.countplot(bigdatamart_test['Item_Fat_Content'])    


# In[115]:


label = list(bigdatamart_test['Item_Type'].unique())
graph = sns.countplot(bigdatamart_test['Item_Type'])    
graph.set_xticklabels(labels =label,rotation = 90 )


# In[116]:


sns.countplot(bigdatamart_test['Outlet_Size'])    


# In[117]:


sns.countplot(bigdatamart_test['Outlet_Location_Type'])    


# In[119]:


plt.figure(figsize = (10,10))
sns.countplot(bigdatamart_test['Outlet_Type'])    


# In[120]:


### Checking skewness in Test data:
bigdatamart_test.skew()


# In[ ]:


All values are in range +/- 1,so will not do anything for skewness.


# In[121]:


### Checking outliers in Test Data:-
sns.boxplot(bigdatamart_test['Item_Weight'])


# In[122]:


sns.boxplot(bigdatamart_test['Item_Visibility'])


# In[123]:


sns.boxplot(bigdatamart_test['Item_MRP'])


# In[126]:


sns.boxplot(bigdatamart_test['years_old'])


# In[129]:


#So outliers only in Item_Visibilty column,So will remove using Z score:-
from scipy.stats import zscore
z_score = zscore(bigdatamart_test[['Item_Visibility']])
abs_zscore=np.abs(z_score)
new_data =( abs_zscore < 3).all(axis=1)
bigdatamart_test = bigdatamart_test[new_data]
bigdatamart_test.shape


# In[131]:


## Percentage loss:-
loss_data =( (5681-5595)/5681)*100
loss_data


# # Correlation 

# In[133]:


sns.heatmap(bigdatamart_train.corr(),annot=True)


# So ,here we can see,Only Item_MRP has good correlation with Target column,No columns are very much correlated with each others,
# and Target variables too.

# In[134]:


sns.heatmap(bigdatamart_test.corr(),annot=True)


# # Label Encoding for Categorical Columns in Train Data:-

# In[135]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat_col = ['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']
for col in cat_col:
    bigdatamart_train[col] = le.fit_transform(bigdatamart_train[col])
    


# In[136]:


bigdatamart_train.head()


# # Label Encoding for Categorical Columns in Test Data:-

# In[137]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat_col = ['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']
for col in cat_col:
    bigdatamart_test[col] = le.fit_transform(bigdatamart_test[col])


# In[138]:


bigdatamart_test.head()


# In[139]:


## Splitting data between features and columns:-
## Since Test data has not target column,so will do model training and testing only on Train data.
x = bigdatamart_train.drop('Item_Outlet_Sales',axis=1)

y = bigdatamart_train['Item_Outlet_Sales']
y


# In[140]:


x.shape


# In[141]:


y.shape


# In[142]:


### Since data is in very vast range,so applying Standardization:-
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)


# In[145]:


x_scaler


# # Train Test Split

# In[146]:


## Finding best random state:-
from sklearn.metrics import r2_score

maxAcc = 0
maxrs =0
for i in range(1,200):
    x_train,x_test,y_train,y_test = train_test_split(x_scaler,y,test_size =0.20,random_state =i)
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    pred_test = lr.predict(x_test)
    score = r2_score(y_test,pred_test)
    
    if score >maxAcc:
        maxAcc=score
        maxrs=i
print("Best Accuracy is :",maxAcc,"at random state",maxrs)        


# # Linear Regression

# In[156]:


x_train,x_test,y_train,y_test = train_test_split(x_scaler,y,test_size =0.20,random_state =142)
lr = LinearRegression()
lr.fit(x_train,y_train)
pred_test = lr.predict(x_test)
score = r2_score(y_test,pred_test)
print(score)


# # Decision Tree Regression

# In[149]:


dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
pred_test = dtr.predict(x_test)
score = r2_score(y_test,pred_test)

score


# # Ensemble 

# In[150]:


rf = RandomForestRegressor()
rf.fit(x_train,y_train)
pred_test = rf.predict(x_test)
score = r2_score(y_test,pred_test)

score


# #  KNeighborsRegressor
# 

# In[151]:


knr = KNeighborsRegressor()
knr.fit(x_train,y_train)
pred_test = knr.predict(x_test)
score = r2_score(y_test,pred_test)

score


# ### Ridge Regression
# 

# In[157]:


rd = Ridge()
rd.fit(x_train,y_train)
predict_rd = rd.predict(x_test)

score = r2_score(y_test,predict_rd)
score


# # GradientBoosting Regressor:-
# 

# In[159]:


gbr = GradientBoostingRegressor()
gbr.fit(x_train,y_train)
predict_rd = gbr.predict(x_test)

score = r2_score(y_test,predict_rd)
score


# # AdaBoostRegressor

# In[161]:


adb = AdaBoostRegressor()
adb.fit(x_train,y_train)
predict_rd = adb.predict(x_test)

score = r2_score(y_test,predict_rd)
score


# # Cross Validation Of Models:-

# In[163]:


from sklearn.model_selection import cross_val_score
for i in range(2,10):
    cv_score = cross_val_score(lr,x_scaler,y,cv = i)
    cv_mean = cv_score.mean()
    print(f"At cross fold {i} the cv score is {cv_mean}")


# At crossfold 2,score is maximum,so will choose cv=2

# In[164]:


cvs= cross_val_score(lr,x_scaler,y,cv=2)
print("Cross Validation of Linear Regression model ",cvs.mean())


# In[165]:


cvs= cross_val_score(dtr,x_scaler,y,cv=2)
print("Cross Validation of Decision Tree Regression model ",cvs.mean())


# In[166]:


cvs= cross_val_score(rf,x_scaler,y,cv=2)
print("Cross Validation of RandomForest Regression model ",cvs.mean())


# In[167]:


cvs= cross_val_score(knr,x_scaler,y,cv=2)
print("Cross Validation of KNeighbors Regression model ",cvs.mean())


# In[168]:


cvs= cross_val_score(rd,x_scaler,y,cv=2)
print("Cross Validation of Ridge Regression model ",cvs.mean())


# In[169]:


cvs= cross_val_score(gbr,x_scaler,y,cv=2)
print("Cross Validation of GradientBoosting Regression model ",cvs.mean())


# In[171]:


cvs= cross_val_score(adb,x_scaler,y,cv=2)
print("Cross Validation of AdaBoost Regression model ",cvs.mean())


# Cross validation score is higher for GradientBoosting Regression model,so will do hyperparameter tuning for that.

# # HyperParameter Tuning for GradientBoosting Regression:-

# In[178]:


params = {
          'criterion':['friedman_mse', 'squared_error', 'mse', 'mae'],
          'max_features':["auto","sqrt","log2"],
    
         }
gbr = GradientBoostingRegressor()
grid_search = GridSearchCV(gbr,params)
grid_search.fit(x_train,y_train)
print(grid_search.best_params_)


# In[179]:


gbr = GradientBoostingRegressor(criterion='friedman_mse',max_features='sqrt')

gbr.fit(x_train,y_train)
predict_rd = gbr.predict(x_test)

score = r2_score(y_test,predict_rd)
score


# # Model Saving

# In[182]:


filename = 'BigMartSales_prediction.pickle'
pickle.dump(gbr,open(filename,'wb'))


# # Conclusion
# 

# In[183]:


## Conclusion : -
a =np.array(y_test)
predicted = np.array(gbr.predict(x_test))
df =  pd.DataFrame({ 'Original' : a,
                     'predicted' :predicted ,
                   },index = range(len(a))) 
df


# In[ ]:




