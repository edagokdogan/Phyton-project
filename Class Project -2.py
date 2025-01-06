#!/usr/bin/env python
# coding: utf-8

# CLASS PROJECT 
# 
# 
# EDA GOKDOGAN

# Part 1 

# This dataset demonstrates whether an individual earn over $50K/yr based 
# on census data, which is known as "Census Income" dataset. 
# I am interested in this dataset because of its relevance to its socio-economic analysis and its connection to employment status, income levels, demographic factors. 
# The data supports all the requirement with a great balance of categorical and numerical variables.
# The link to the data and documentation: 
# https://doi.org/10.24432/C5XW20
# 
# https://archive.ics.uci.edu/dataset/2/adult

# In[151]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')


# In[152]:


columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]
adult_data = pd.read_csv('/Users/edagokdogan/Desktop/adult/adult.data', names=columns, na_values=[" ?"])


# In[153]:


# Part 2 


# In[154]:


adult_data.head()


# In[155]:


adult_data.index.astype(str)


# In[156]:


adult_data.tail()


# In[157]:


adult_data.info()


# In[158]:


num_rows, num_columns = adult_data.shape
print(adult_data.shape)


# In[159]:


adult_data.columns


# In[160]:


adult_data.dtypes


# In[161]:


adult_data.index


# In[162]:


adult_data.shape[0]


# In[163]:


adult_data.describe(include='all')


# In[164]:


missing_values = adult_data.isnull().sum()
print(missing_values)


# In[165]:


num_int64 = sum(adult_data.dtypes == 'int64')
print(num_int64)


# In[166]:


num_object = sum(adult_data.dtypes == 'object')
print(num_object)


# In[167]:


# The findings that I obtained makes sense. The number of records that I obtained is 
# 32561. The number of features are 32561 and number of variables are 15. 6 of them is numerical 
# 9 of them is categorical. The categorical values are discrete. There are missing values with 1836 working class, 1843 occupation and 583 native country. 


# In[168]:


#Abnormal values 
infinite = adult_data.isin([np.inf, -np.inf]).sum()
null_counts = adult_data.isnull().sum()
print(infinite)
print(null_counts)


# In[169]:


#Identify the categorical values 
column_type = adult_data.columns[adult_data.dtypes == 'int64']
print(column_type)
def display_categorical_values(adult_data, column_type):
    for col in column_type:
        unique_values = adult_data[col].unique()
        print(unique_values)


# In[170]:


#Quantify relationships between variables 
numerical_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
corr_matrix = adult_data[numerical_cols].corr()
print(corr_matrix)


# The age ranges from 17 to 90, which is considered as normal. Weight is the final 
# weight for each individual, the values are expected to be large in this category. 
# Education levels classified as 1 through 16. Capital gain and loss are starting from 0, which is normal. Capital gain is really high value of 99,999 which 
# could be an outlier. Hours per week is high for 99 hours, which also seems like an outlier.
# There are couple of them that seems to be outliers but seems to be not a big problem for this data. 
# there seems to be some correlation between variables such as education_num and capital gain but the correlations are not that strong between variables.
# 

# In[172]:


Na_values = adult_data.columns[adult_data.isnull().any()]
print(Na_values)


# In[173]:


adult_data_cleaned = adult_data.drop(columns=Na_values)


# In[174]:


adult_data_cleaned['has_capital_gain'] = (adult_data_cleaned['capital_gain'] > 0).astype(int)
capital_gain_binary = adult_data_cleaned['has_capital_gain'].value_counts()
print(capital_gain_binary)


# In[175]:


bins = [0, 35, 40, adult_data_cleaned['hours_per_week'].max()]
labels = ['Part-time', 'Full-time', 'Over-time']
adult_data_cleaned['work_category'] = pd.cut(
    adult_data_cleaned['hours_per_week'],
    bins=bins,
    labels=labels,
    right=True,      
    include_lowest=True  
)
print(adult_data_cleaned[['hours_per_week', 'work_category']].head(15))


# EDA Histograms, Box plots, Line plots, Scatter Plots 

# In[177]:


#Histogram Univariate Plot - using seaborn #1 
sns.histplot(adult_data_cleaned['age'], bins=30, kde=True, color='red')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


#Bar plot  Univariate Plot #2
sns.countplot(x='work_category', data=adult_data_cleaned)
plt.title('Distribution of Work Categories')
plt.xlabel('Work Category')
plt.ylabel('Count')
plt.show()


# In[ ]:


# Histogram Univariate Plot - using matplotlib  #3
plt.hist(
    adult_data_cleaned['hours_per_week'],  
    bins=30, 
    color='yellow', 
    edgecolor='black', 
    alpha=0.7  
)
plt.title('Distribution of Hours Worked Per Week', fontsize=16)
plt.xlabel('Hours Worked Per Week', fontsize=14)
plt.ylabel('Frequency', fontsize=14)


# In[ ]:


#Bivariate Bar Plot #1 
sns.countplot(x='work_category', hue='income', data=adult_data_cleaned)
plt.xlabel('Work Hours')
plt.ylabel('Count')
plt.legend(title='Income', loc='upper right')
plt.show()
#The data shows the relationship between work hours category and income. 
#It could be interpreted that earn over 50k seems to be working over time.


# Bivariate Boxplot #2

# In[ ]:


sns.boxplot(x='income', y='education_num', data=adult_data_cleaned)
plt.title('Education Level by Income Level')
plt.xlabel('Income Level')
plt.ylabel('Education Level')
plt.show()
#This shows how the education level affects the income level. 


# Bivariate Scatterplot #3 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.scatter(
    adult_data_cleaned['education_num'],
    adult_data_cleaned['hours_per_week'],
    alpha=0.3,            
    color='blue',               
    linewidth=0.5,        
    s=25                  
)
plt.title('Education Level vs. Hours Worked Per Week', fontsize=16)
plt.xlabel('Education Level', fontsize=14)
plt.ylabel('Hours Worked Per Week', fontsize=14)
plt.tight_layout()
plt.show()


# In[178]:


#Bivariate Lineplot #4 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure that 'education_num' and 'capital_gain' columns exist
print(adult_data_cleaned.columns)
avg_capital_gain = adult_data_cleaned.groupby('education_num')['capital_gain'].mean().reset_index()
avg_capital_gain = avg_capital_gain.sort_values('education_num')
sns.lineplot(
    x='education_num',
    y='capital_gain',
    data=avg_capital_gain,
    marker='o',
    color='black'
)
plt.title('Average Capital Gain By Education Levels', fontsize=16)
plt.xlabel('Education Level', fontsize=14)
plt.ylabel('Average Capital Gain', fontsize=14)
plt.show()


# Multivariate Plot 1 

# In[179]:


import seaborn as sns
import matplotlib.pyplot as plt
selected_features = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
hue_variable = 'income'
sns.pairplot(
    data=adult_data_cleaned,
    vars=selected_features,
    hue=hue_variable,
    palette='Set2',
    diag_kind='kde',  
    corner=True  
)
plt.suptitle('Pair Plot of Numerical Features Colored by Income', y=1.00, fontsize=15)
plt.show()


# Multivariate Plot 2 

# In[180]:


import seaborn as sns
import matplotlib.pyplot as plt

numerical_vars = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
correlation_matrix = adult_data_cleaned[numerical_vars].corr()
sns.heatmap(
    correlation_matrix,
    annot=True,  
    cmap='coolwarm', 
    linewidths=0.5  )

plt.title('Correlation Matrix of Numerical Variables', fontsize=16)
plt.tight_layout()
plt.show()


# * Based on the visualization that I have depicted, I learned that education level and capital level is highly correlated to each other. Also, hours per week affects the capital gain and income positively.Based on the data, the income is the highest for workers working overtime.  One research question that I would like to examine is: How much of overtime work lead to high income? The data also showed that the full time  working earning less than 50k is higher than the part time working earning less than 50k. How the education number  is determined and education number 14 is the highest and the capital gain depreciates after? I was not able to get the age factor into account due to the disparity of this group. I would like to further examine this group and learn how would age plays role in income and capital gain?  
