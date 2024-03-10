#!/usr/bin/env python
# coding: utf-8

# #  Importing the libraries and data

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Provide the file path as a string
file_path = r'D:\practices\csv file\fraud_oracle.csv'

# Read the CSV file using pandas
data = pd.read_csv(file_path)


# In[3]:


data


# # Exploratory data analysis -  Initial Investigation of the data set
# Look at unique values within the data as a fast way to identify any missing or interesting values.

# In[4]:


data.isnull().sum()


# In[5]:


print(data.describe().T)


# Analysis of Categorical Variables;

# # Further investigation for insights - possible features
# The goal here is to gather some insight into the relationship between our observations and the desired predicted feature, 'FraudFound_P'.

# In[6]:


# Analysis the accident based on the month
df_month = data.groupby("Month").agg({"Month":"count"})
df_month.columns = ["Counts"]
df_month.reset_index(inplace=True)
print(df_month)


# In[7]:


plt.bar(df_month['Month'], df_month['Counts'])
plt.title("Number of accidents per month")
plt.xlabel("Month")
plt.ylabel("Counts")
plt.show()


# In[8]:


# Assuming df_month is your DataFrame containing the months and their counts
max_month = df_month['Month'][df_month['Counts'].idxmax()]
print("Month with maximum count:", max_month)


# In[9]:


data_month_weekday=data.groupby(["Month","DayOfWeek"]).agg({"Month":"count"})


# In[10]:


data_month_weekday.columns=["count"]
data_month_weekday.reset_index(inplace=True)
print(data_month_weekday.head(10))


# In[11]:


pip install plotly


# In[12]:


import plotly.express as px
fig = px.bar(data_month_weekday, x="Month", y="count", color="DayOfWeek",
             pattern_shape="DayOfWeek", pattern_shape_sequence=[".", "x", "+"],
            title = "How many accidents happened on which days of the month?")
fig.show()


# In[ ]:





# In[13]:


# Gender and marital status of the accident victims (consider with all years)
df_sex_maritalstatus = data.groupby(["Sex", "MaritalStatus"]).agg({"Sex":"count"})
df_sex_maritalstatus.columns = ["Counts"]
df_sex_maritalstatus.reset_index(inplace=True)
print(df_sex_maritalstatus.head(10))


# In[14]:


fig = px.bar(df_sex_maritalstatus, x="Sex", y="Counts",
             color='MaritalStatus', barmode='group',
             height=400,
            title = "Gender and marital status of the accident victims")
fig.show()


# In[15]:


for column in data:
    if column == 'PolicyNumber':
        pass
    else:
        print(column)
        print(sorted(data[column].unique()),"\n")


# From the analysis we could see Age, MonthClaimed, DayOfWeekClaimed has a contain 0

# In[16]:


data[data['Age']==0].shape


# In[17]:


# Analysis row with Age == 0
data[data['Age']==0]['AgeOfPolicyHolder'].unique() #Result = array(['16 to 17'], dtype=object)

# Because the row with Age == 0, only appear in row with AgeOfPolicyHolder == '16 to 17', i will impute with 16.5
data['Age'] =data['Age'].replace({0:16.5})


# In[18]:


data[data['MonthClaimed']=='0']


# In[19]:


data[data['DayOfWeekClaimed']=='0']


# We can remove the the data MonthClaimed and DayOfWeekClaimed. Since there is just one entries in the data set. Further we can policy number column since it is unique value doesn't going affect the model

# In[20]:


print(data.columns)

# Drop the 'PolicyNumber' column if it exists
if 'PolicyNumber' in data.columns:
    data.drop(columns='PolicyNumber', inplace=True)


# In[21]:


data = data[~(data['MonthClaimed']=='0')]
data = data[~(data['DayOfWeekClaimed']=='0')]


# In[22]:


data


# In[23]:


# Analysis the accident based on the month
df_month = data.groupby("Month").agg({"FraudFound_P":"sum"})
df_month.columns = ["Counts"]
df_month.reset_index(inplace=True)
print(df_month)


# In[24]:


plt.bar(df_month['Month'], df_month['Counts'])
plt.title("Number of accidents per month")
plt.xlabel("Month")
plt.ylabel("Counts")
plt.show()


# In[25]:


fig, ax = plt.subplots(3,3, figsize=(15,10))
sns.countplot(data=data, x='Month', hue='FraudFound_P', order=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ax=ax[0][0])
ax[0][0].set_title('Month')

sns.countplot(data=data, x='DayOfWeek', hue='FraudFound_P', order=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'], ax=ax[0][1])
ax[0][1].set_title('Day')

sns.countplot(data=data, x='Sex', hue='FraudFound_P', ax=ax[0][2])
ax[0][2].set_title('Sex')

sns.countplot(data=data, x='MaritalStatus', hue='FraudFound_P', ax=ax[1][0])
ax[1][0].set_title('Marital Status')

sns.countplot(data=data, x='NumberOfCars', hue='FraudFound_P', ax=ax[1][1])
ax[1][1].set_title('Number Of Cars')

sns.countplot(data=data, x='AccidentArea', hue='FraudFound_P', ax=ax[1][2])
ax[1][2].set_title('Accident Area')

sns.countplot(data=data, x='DriverRating', hue='FraudFound_P', ax=ax[2][0])
ax[2][0].set_title('Driver Rating')

sns.countplot(data=data, x='AgentType', hue='FraudFound_P', ax=ax[2][1])
ax[2][1].set_title('Agent Type')

sns.countplot(data=data, x='BasePolicy', hue='FraudFound_P', ax=ax[2][2])
ax[2][2].set_title('Base Policy')


plt.tight_layout()


# In[26]:


plt.figure(figsize=(18,15))
corr = data.corr()
sns.heatmap(data=corr, annot = True)
plt.show()


# In[27]:


# get the taret and independent Separated
x = data.drop('FraudFound_P', axis=1)
y = data['FraudFound_P']


# In[28]:


x


# In[29]:


#Converting label columns into Numerical by doing one-encoding
categorical_cols=x.select_dtypes(include=['object'])
categorical_cols=pd.get_dummies(categorical_cols, drop_first = True)
categorical_cols.head()


# In[30]:


numerical_col = x.select_dtypes(include = ['int64'])
x = pd.concat([numerical_col,categorical_cols], axis =1)


# In[31]:


# setting the figure size
width = 20
height = 8
sns.set(rc = {'figure.figsize':(width,height)})
sns.barplot(data = data, x='PolicyType', y='FraudFound_P')


# In[32]:


#plotting by FraudFound, looking to see if there are anything obvious that correlates to fraud
val1=data.groupby('PolicyType').agg({'FraudFound_P':'sum'}).reset_index()
val2=data.groupby('PolicyType').agg('count').reset_index()

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(22, 6))
sns.barplot(x='PolicyType', y='FraudFound_P', data =val1, ax=ax1)
sns.barplot(x='PolicyType', y='FraudFound_P', data=val2, ax=ax2)

ax2.set(ylabel='Total counts')


# In[33]:


#plotting by FraudFound, looking to see if there are anything obvious that correlates to fraud
val3=data.groupby('VehicleCategory').agg({'FraudFound_P':'sum'}).reset_index()
val4=data.groupby('VehicleCategory').agg('count').reset_index()
val5=data.groupby('BasePolicy').agg({'FraudFound_P':'sum'}).reset_index()
val6=data.groupby('BasePolicy').agg('count').reset_index()

fig, (ax1, ax3) = plt.subplots(1,2,figsize=(15, 5))
sns.barplot(x='VehicleCategory', y='FraudFound_P', data = val3, ax=ax1)
#sns.barplot(x='VehicleCategory', y='FraudFound_P', data = gpd_val2, ax=ax2)
sns.barplot(x='BasePolicy', y='FraudFound_P', data = val5, ax=ax3)


# In[34]:


# Outlier Check
plt.figure(figsize = (20,15))
plotnumber = 1
for col in x.columns:
    if plotnumber<=24:
        ax = plt.subplot(5,5,plotnumber)
        sns.boxplot(x[col])
        plt.xlabel(col,fontsize = 15)
    plotnumber += 1
plt.tight_layout()
plt.show()
    


# Outlier are in there, so we need to standaries those columns uning standard scaler

# # Preprocessing and Splitting data into dependent and independent variables

# In[35]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25)


# In[36]:


x_train.head()


# In[37]:


for col in x_train.columns:
    print(f"'{col}',", end=" ")


# In[38]:


numerical_data = x_train[['WeekOfMonth', 'WeekOfMonthClaimed', 'RepNumber', 'Deductible', 'DriverRating', 'Year', 'Month_Aug', 'Month_Dec', 'Month_Feb', 'Month_Jan', 'Month_Jul', 'Month_Jun', 'Month_Mar', 'Month_May', 'Month_Nov', 'Month_Oct', 'Month_Sep', 'DayOfWeek_Monday', 'DayOfWeek_Saturday', 'DayOfWeek_Sunday', 'DayOfWeek_Thursday', 'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday', 'Make_BMW', 'Make_Chevrolet', 'Make_Dodge', 'Make_Ferrari', 'Make_Ford', 'Make_Honda', 'Make_Jaguar', 'Make_Lexus', 'Make_Mazda', 'Make_Mecedes', 'Make_Mercury', 'Make_Nisson', 'Make_Pontiac', 'Make_Porche', 'Make_Saab', 'Make_Saturn', 'Make_Toyota', 'Make_VW', 'AccidentArea_Urban', 'DayOfWeekClaimed_Monday', 'DayOfWeekClaimed_Saturday', 'DayOfWeekClaimed_Sunday', 'DayOfWeekClaimed_Thursday', 'DayOfWeekClaimed_Tuesday', 'DayOfWeekClaimed_Wednesday', 'MonthClaimed_Aug', 'MonthClaimed_Dec', 'MonthClaimed_Feb', 'MonthClaimed_Jan', 'MonthClaimed_Jul', 'MonthClaimed_Jun', 'MonthClaimed_Mar', 'MonthClaimed_May', 'MonthClaimed_Nov', 'MonthClaimed_Oct', 'MonthClaimed_Sep', 'Sex_Male', 'MaritalStatus_Married', 'MaritalStatus_Single', 'MaritalStatus_Widow', 'Fault_Third Party', 'PolicyType_Sedan - Collision', 'PolicyType_Sedan - Liability', 'PolicyType_Sport - All Perils', 'PolicyType_Sport - Collision', 'PolicyType_Sport - Liability', 'PolicyType_Utility - All Perils', 'PolicyType_Utility - Collision', 'PolicyType_Utility - Liability', 'VehicleCategory_Sport', 'VehicleCategory_Utility', 'VehiclePrice_30000 to 39000', 'VehiclePrice_40000 to 59000', 'VehiclePrice_60000 to 69000', 'VehiclePrice_less than 20000', 'VehiclePrice_more than 69000', 'Days_Policy_Accident_15 to 30', 'Days_Policy_Accident_8 to 15', 'Days_Policy_Accident_more than 30', 'Days_Policy_Accident_none', 'Days_Policy_Claim_8 to 15', 'Days_Policy_Claim_more than 30', 'PastNumberOfClaims_2 to 4', 'PastNumberOfClaims_more than 4', 'PastNumberOfClaims_none', 'AgeOfVehicle_3 years', 'AgeOfVehicle_4 years', 'AgeOfVehicle_5 years', 'AgeOfVehicle_6 years', 'AgeOfVehicle_7 years', 'AgeOfVehicle_more than 7', 'AgeOfVehicle_new', 'AgeOfPolicyHolder_18 to 20', 'AgeOfPolicyHolder_21 to 25', 'AgeOfPolicyHolder_26 to 30', 'AgeOfPolicyHolder_31 to 35', 'AgeOfPolicyHolder_36 to 40', 'AgeOfPolicyHolder_41 to 50', 'AgeOfPolicyHolder_51 to 65', 'AgeOfPolicyHolder_over 65', 'PoliceReportFiled_Yes', 'WitnessPresent_Yes', 'AgentType_Internal', 'NumberOfSuppliments_3 to 5', 'NumberOfSuppliments_more than 5', 'NumberOfSuppliments_none', 'AddressChange_Claim_2 to 3 years', 'AddressChange_Claim_4 to 8 years', 'AddressChange_Claim_no change', 'AddressChange_Claim_under 6 months', 'NumberOfCars_2 vehicles', 'NumberOfCars_3 to 4', 'NumberOfCars_5 to 8', 'NumberOfCars_more than 8', 'BasePolicy_Collision', 'BasePolicy_Liability']]


# # Encoding the data.

# In[39]:


# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)


# In[40]:


scaled_num_df = pd.DataFrame(data=scaled_data, columns=numerical_data.columns,index=x_train.index)
scaled_num_df.head()


# # Modelling

# Support Vector Classifier

# In[41]:


from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(x_train,y_train)

y_pred = svc_model.predict(x_test)


# In[42]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

svc_model_train_acc = accuracy_score(y_train, svc_model.predict(x_train))
svc_model_test_acc = accuracy_score(y_test,y_pred)

print("Training Accuracy:",svc_model_train_acc)
print("testing Accuracy:",svc_model_test_acc)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# K-Nearest Neighborhood

# In[45]:


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 30)
knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_test)


# In[46]:


knn_model_train_acc = accuracy_score(y_train, knn_model.predict(x_train))
knn_model_test_acc = accuracy_score(y_test,y_pred)

print("Training Accuracy:",knn_model_train_acc)
print("testing Accuracy:",knn_model_test_acc)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# Decision Tree Classifier

# In[48]:


from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x_train, y_train)

y_pred = decision_tree_model.predict(x_test)


# In[49]:


decision_tree_model_train_acc = accuracy_score(y_train, decision_tree_model.predict(x_train))
decision_tree_model_test_acc = accuracy_score(y_test,y_pred)

print("Training Accuracy:",decision_tree_model_train_acc)
print("testing Accuracy:",decision_tree_model_test_acc)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# Random Forest Classifer

# In[51]:


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(criterion = 'entropy', max_depth = 15, max_features='sqrt', min_samples_leaf = 1, min_samples_split = 3, n_estimators = 14)
rf_model.fit(x_train,y_train)

y_pred = rf_model.predict(x_test)


# In[52]:


rf_model_train_acc = accuracy_score(y_train, rf_model.predict(x_train))
rf_model_test_acc = accuracy_score(y_test,y_pred)

print("Training Accuracy:",rf_model_train_acc)
print("testing Accuracy:",rf_model_test_acc)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[ ]:




