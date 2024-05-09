#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


df=pd.read_csv("C:/Users/Ravi/Downloads/lass0/Datasets_LassoRidge/Life_expectencey_LR.csv")


# In[25]:


df.head()


# In[26]:


df.info()


# In[31]:


df.isna().sum()


# In[29]:


life_mean=df['Life_expectancy'].mean()


# In[30]:


df.Life_expectancy=df.Life_expectancy.fillna(life_mean)


# In[32]:


adult_mean=df['Adult_Mortality'].mean()
df.Adult_Mortality=df.Adult_Mortality.fillna(adult_mean)


# In[33]:


Alcohol_mean=df['Alcohol'].mean()
df.Alcohol=df.Alcohol.fillna(Alcohol_mean)


# In[36]:


hb_mean=df['Hepatitis_B'].mean()
df.Hepatitis_B=df.Hepatitis_B.fillna(hb_mean)


# In[37]:


BMI_mean=df['BMI'].mean()
df.BMI=df.BMI.fillna(BMI_mean)


# In[38]:


Polio_mean=df['Polio'].mean()
df.Polio=df.Polio.fillna(Polio_mean)


# In[39]:


T_exp_mean=df['Total_expenditure'].mean()
df.Total_expenditure=df.Total_expenditure.fillna(T_exp_mean)


# In[40]:


Diphtheria_mean=df['Diphtheria'].mean()
df.Diphtheria=df.Diphtheria.fillna(Diphtheria_mean)


# In[41]:


GDP_mean=df['GDP'].mean()
df.GDP=df.GDP.fillna(GDP_mean)


# In[42]:


Population_mean=df['Population'].mean()
df.Population=df.Population.fillna(Population_mean)


# In[43]:


thinness_mean=df['thinness'].mean()
df.thinness=df.thinness.fillna(thinness_mean)


# In[44]:


thinness_yr_mean=df['thinness_yr'].mean()
df.thinness_yr=df.thinness_yr.fillna(thinness_yr_mean)


# In[45]:


Income_composition_mean=df['Income_composition'].mean()
df.Income_composition=df.Income_composition.fillna(Income_composition_mean)


# In[46]:


Schooling_mean=df['Schooling'].mean()
df.Schooling=df.Schooling.fillna(Schooling_mean)


# In[47]:


df.isna().sum()


# In[51]:


sns.countplot(df.Status)


# In[53]:


df["Life_expectancy"].plot.hist(grid=True, bins=30, rwidth=0.8) 
plt.title('Life Expectancy') 
plt.ylabel('Count') 
plt.xlabel('Age')


# In[55]:


df.describe(include= 'O')


# In[57]:


le_country = df.groupby('Country')['Life_expectancy'].mean().sort_values(ascending=True)
le_country.plot(kind='bar', figsize=(50,15), fontsize=25)
plt.title("Life_expectancy w.r.t Country",fontsize=40)
plt.xlabel("Country",fontsize=35)
plt.ylabel("Avg Life_Expectancy",fontsize=35)
plt.show()


# In[59]:


le_country = df.groupby('Country')['Adult_Mortality'].mean().sort_values(ascending=True)
le_country.plot(kind='bar', figsize=(50,15), fontsize=25)
plt.title("Adult_Mortality Country",fontsize=40)
plt.xlabel("Country",fontsize=35)
plt.ylabel("Avg Adult Mortality",fontsize=35)
plt.show()


# In[60]:


le_country = df.groupby('Country')['Alcohol'].mean().sort_values(ascending=True)
le_country.plot(kind='bar', figsize=(50,15), fontsize=25)
plt.title("Alcohol Country",fontsize=40)
plt.xlabel("Country",fontsize=35)
plt.ylabel("Avg Alcohol Comsumption",fontsize=35)
plt.show()


# In[62]:


le_country = df.groupby('Country')['HIV_AIDS'].mean().sort_values(ascending=True)
le_country.plot(kind='bar', figsize=(50,15), fontsize=25)
plt.title("HIV Country",fontsize=40)
plt.xlabel("Country",fontsize=35)
plt.ylabel("Avg HIV cases",fontsize=35)
plt.show()


# In[64]:


le_country = df.groupby('Country')['Income_composition'].mean().sort_values(ascending=True)
le_country.plot(kind='bar', figsize=(50,15), fontsize=25)
plt.title("Income Composition of Resources Country",fontsize=40)
plt.xlabel("Country",fontsize=35)
plt.ylabel("Avg income composition of resourses",fontsize=35)
plt.show()


# In[ ]:





# In[ ]:





# In[65]:



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[66]:


df.Country=le.fit_transform(df.Country)


# In[68]:


df.Status=le.fit_transform(df.Status)


# In[9]:





# In[69]:


df.shape


# In[70]:


df.isna().sum()


# In[71]:


df.isnull().sum()


# In[82]:


plt.boxplot(df.thinness)


# In[74]:


iqr=df.thinness.quantile(0.75)-df.thinness.quantile(0.25)
lowerlimit=df.thinness.quantile(0.25)-(iqr*1.5)
upperlimit=df.thinness.quantile(0.75)-(iqr*1.5)


# In[75]:


from feature_engine.outliers import Winsorizer


# In[76]:


winsorizer=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=["thinness"])


# In[77]:


df_t=winsorizer.fit_transform(df[['thinness']])


# In[80]:


df.thinness=df_t


# In[81]:


plt.boxplot(df.thinness)


# In[20]:





# In[21]:





# In[83]:


plt.boxplot(df.thinness_yr)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[84]:



IQR=df.thinness_yr.quantile(0.75)-df.thinness_yr.quantile(0.25)
lower_limit=df.thinness_yr.quantile(0.25)-(IQR*1.5)
upper_limit=df.thinness_yr.quantile(0.75)-(IQR*1.5)

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                           variables=['thinness_yr'])


# In[85]:


df_t=winsor.fit_transform(df[["thinness_yr"]])
df.thinness_yr=df_t


# In[86]:


plt.boxplot(df.thinness_yr)


# In[88]:


df.corr()


# In[89]:


import statsmodels.formula.api as smf


# In[90]:


model1=smf.ols("Life_expectancy ~ Country+Year+Status+Adult_Mortality+infant_deaths+Alcohol+percentage_expenditure+Hepatitis_B+Measles+BMI+under_five_deaths+Polio+Total_expenditure+Diphtheria+HIV_AIDS+GDP+Population+thinness_yr+Income_composition+Schooling",data=df).fit()


# In[91]:


model1.summary()


# In[93]:



pred1 = model1.predict(pd.DataFrame(df))


# In[94]:


pred1


# In[95]:


import statsmodels.api as sm


# In[ ]:





# In[100]:


sm.graphics.influence_plot(model1)


# In[97]:


df_new = df.drop(df.index[[1187]])


# In[111]:


model2=smf.ols("Life_expectancy ~ Country+Year+Status+Adult_Mortality+infant_deaths+Alcohol+percentage_expenditure+Hepatitis_B+Measles+BMI+under_five_deaths+Polio+Total_expenditure+Diphtheria+HIV_AIDS+GDP+Population+thinness_yr+Income_composition+Schooling",data=df_new).fit()


# In[112]:


model2.summary()


# In[103]:


pred2 = model2.predict(pd.DataFrame(df_new))


# In[105]:



# Error calculation
res1 = df.Life_expectancy - pred2
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


# In[106]:


sm.graphics.influence_plot(model2)


# In[115]:


model3=smf.ols("Life_expectancy ~ Country+Status+Adult_Mortality+infant_deaths+Alcohol+Hepatitis_B+Measles+BMI+under_five_deaths+Polio+Diphtheria+HIV_AIDS+GDP+thinness_yr+Income_composition+Schooling",data=df_new).fit()


# In[116]:


model3.summary()


# In[127]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[123]:


final_model=smf.ols("Life_expectancy ~ Country+Year+Status+Adult_Mortality+infant_deaths+Alcohol+percentage_expenditure+Hepatitis_B+Measles+BMI+under_five_deaths+Polio+Total_expenditure+Diphtheria+HIV_AIDS+GDP+Population+thinness_yr+Income_composition+Schooling",data=df_new).fit()


# In[124]:


final_model.summary()


# In[129]:



res = final_model.resid
sm.qqplot(res);plt.title("final_model")
plt.show()


# In[134]:


from scipy import stats
import pylab
# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()


# In[136]:


pred = final_model.predict(df_new)


# In[137]:



sns.residplot(x = pred, y = df_new.Life_expectancy, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()


# In[138]:


from sklearn.model_selection import train_test_split


# In[139]:


df_train,df_test=train_test_split(df_new,test_size=0.2,random_state=10)


# In[140]:


model_train=smf.ols("Life_expectancy ~ Country+Year+Status+Adult_Mortality+infant_deaths+Alcohol+percentage_expenditure+Hepatitis_B+Measles+BMI+under_five_deaths+Polio+Total_expenditure+Diphtheria+HIV_AIDS+GDP+Population+thinness_yr+Income_composition+Schooling",data=df_train).fit()


# In[141]:


test_predict=final_model.predict(df_test)


# In[ ]:





# In[143]:


test_residual = test_predict - df_test.Life_expectancy


# In[144]:


test_rmse=np.sqrt(np.mean(test_residual*test_residual))


# In[145]:


test_rmse


# In[146]:


train_predict=model_train.predict(df_train)


# In[148]:


train_residual=train_predict - df_train.Life_expectancy


# In[149]:


train_rmse=np.sqrt(np.mean(train_residual*train_residual))


# In[150]:


train_rmse


# In[151]:




### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
rm = Ridge(alpha = 5, normalize = True)


# In[159]:



rm.fit(df_new.iloc[:,df_new.columns!='Life_expectancy'], df_new.Life_expectancy)


# In[161]:


df_new=df_new.iloc[:,[3,0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]


# In[156]:



# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_


# In[164]:



plt.bar(height = pd.Series(rm.coef_), x = pd.Series(df_new.columns[1:]));plt.xticks(rotation=90,ha='right')


# In[165]:



rm.alpha


# In[166]:



pred_rm = rm.predict(df_new.iloc[:, 1:])


# In[168]:



# Adjusted r-square
rm.score(df_new.iloc[:, 1:], df_new.Life_expectancy)


# In[169]:



# RMSE
np.sqrt(np.mean((pred_rm - df_new.Life_expectancy)**2))


# In[170]:




from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.13, normalize = True)


# In[171]:



lasso.fit(df_new.iloc[:, 1:], df_new.Life_expectancy)


# In[172]:



# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_


# In[175]:



plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(df_new.columns[1:]));plt.xticks(rotation=90,ha='right')


# In[176]:



lasso.alpha


# In[178]:



pred_lasso = lasso.predict(df_new.iloc[:, 1:])


# In[180]:



# Adjusted r-square
lasso.score(df_new.iloc[:, 1:], df_new.Life_expectancy)


# In[181]:



# RMSE
np.sqrt(np.mean((pred_lasso - df_new.Life_expectancy)**2))


# In[ ]:




