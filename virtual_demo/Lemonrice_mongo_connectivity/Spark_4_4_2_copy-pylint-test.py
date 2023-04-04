#!/usr/bin/env python
# coding: utf-8

# Importing libraries for operations

import os
import numpy as np

# import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import findspark
import pyspark.pandas as ps
import pyspark.sql.functions as F
from pprint import pprint
from pymongo import MongoClient
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import when, count
from pyspark.sql.functions import col, sum
from pyspark.sql.functions import mean, desc
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier as dtc
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)

findspark.init()

#
# # <strong><center>Predicting Customer Conversion on Bank Telemarketing Dataset</center></strong>
#
# <div class="alert alert-info"><strong>TODAY:</strong> While bank marketing campaigns have largely remained traditional, standard bank advertising is boring for both marketers and their consumers. Thinking outside the box and implementing creative marketing ideas for banks will help you to develop creative campaigns, benefiting your bank, consumer engagement, and likely the success of the actual bank marketing campaigns as well.
#
# Most consumers see banking as a mundane necessity, something they’d rather avoid if they could. Taking a creative approach to bank marketing campaigns might change that, especially if you integrate creative marketing ideas for banks like gamification, automation, chatbots, and rewards so that consumers are motivated to use your services.</div>

# In[1]:


# Setting path for working

try:
    # Getting the current working directory
    cwd = os.getcwd()

    # Printing the current working directory
    print("The current directory is: {0}".format(cwd))

    # Changing the current working directory
    os.chdir(cwd)

    print("The current working directory was set succesfully!")

except:
    print("OOPs!! The current working directory was not set succesfully...")


# ## Using API from Kaggle to download dataset

# # downloading the dataset
# !kaggle datasets download -d princeganer/bank-telemarketing-dataset
#
# # unzipping the downloaded dataset
# !unzip bank-telemarketing-dataset.zip

# In[2]:


# the zipped as well as unzipped files can be seen
get_ipython().system("dir")

print("\n\nSuccessfully downloaded the dataset from Kaggle's API.")


# <div class="alert alert-info">Sucessfully imported data from Kaggle API</div>

# In[3]:





# In[4]:


# create a sparksession
spark = SparkSession.builder.master("local[4]").appName("ml").getOrCreate()


# In[5]:


# display function for spark dataframe
spark.conf.set("spark.sql.repl.eagerEval.enabled", True)

# to display maximum columns in pandas
ps.pandas.set_option("display.max_columns", None)


# # Data Preprocessing

# #### Bank Full (dataset-1) <br>
#
#
# `bank-full.csv` Bank Full Preprocessing
#
# Header | Definition
# ---|---------
# `Age`| Age of the client
# `Job` | Job type of the client
# `Martial` | Martial status of the client
# `Education` | Client's highest education level
# `Default` |  Does the customer have credit in default?
# `Balance` | Client's individual balance
# `Housing` | Does the client have housing loan
# `Loan` | Does the client have personal Loan
# `Contact` | Communication type of contact with customer
# `Day` | Last contact day of the week
# `Month` |  Last contact month of year
# `Duration` | Last contact duration (in seconds)
# `Campaign` | Number of contacts performed during this campaign and for this client
# `Pdays` | Number of days that passed by after the client was last contacted from a previous campaign
# `Previous` | Number of contacts performed before this campaign and for this client
# `Poutcome` | Outcome of the previous marketing campaign
# `y` | Has the client subscribed for a term deposit

# In[6]:


# Read Spark Dataframe
bank_full = spark.read.csv("bank-full.csv", sep=";", header=True, inferSchema=True)


# In[7]:


bank_full.count()


# In[8]:


display(bank_full.limit(5))


# In[9]:


bank_full.printSchema()


# In[10]:


# Adding the index columns

new_cols = [
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor_3m",
    "nr_employed",
]
for column in new_cols:
    bank_full = bank_full.withColumn(column, bank_full["poutcome"] + 1)

display(bank_full.limit(5))


# In[11]:


# map years into a dataframe


def year_mapper(data, start_yr, end_yr):
    """
    This function takes dataframe, start year of data, end year of data as input and
    returns a new dataframe having year column mapped to it.
    """
    month_lst = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]

    # Make a copy of the original dataframe
    new_data = data.copy()

    # Insert a new "year" column filled with zeros
    new_data.insert(loc=0, column="year", value=0)

    # Set the first year to the start year
    current_year = int(start_yr)
    new_data.at[0, "year"] = current_year

    # Loop through the rows of the dataframe, updating the year column when the month changes
    for i in range(1, len(new_data)):
        # If the current month is earlier in the year than the previous month, increment the year
        if month_lst.index(new_data["month"][i]) < month_lst.index(
            new_data["month"][i - 1]
        ):
            current_year += 1

        new_data.at[i, "year"] = current_year

        # If the current year exceeds the end year, break out of the loop
        if current_year > end_yr:
            break

    return new_data


# In[12]:


# Use default index prevent overhead.
ps.set_option("compute.default_index_type", "distributed")

# Ignore warnings coming from Arrow optimizations.
warnings.filterwarnings("ignore")

# To speed up dataset processing
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", True)

bank_full_pdf = bank_full.toPandas()

# Apply the function to the Pandas DataFrame
new_pandas_df = year_mapper(bank_full_pdf, 2008, 2010)

# Convert the updated Pandas DataFrame back to a PySpark DataFrame
bank_full = spark.createDataFrame(new_pandas_df)


# In[13]:


display(bank_full.limit(5))


# In[14]:


bank_full.groupBy("year").count().show()


# ### Index mapper

# In[15]:


# Adding the missing index into the dataframes because it have the same values along the months


def map_index(new_data):
    index_2008 = {
        "may": {
            "emp_var_rate": 1.1,
            "cons_price_idx": 93.994,
            "cons_conf_idx": -36.4,
            "euribor_3m": 4.85,
            "nr_employed": 5191,
        },
        "jun": {
            "emp_var_rate": 1.4,
            "cons_price_idx": 94.465,
            "cons_conf_idx": -41.8,
            "euribor_3m": 4.86,
            "nr_employed": 5228.1,
        },
        "jul": {
            "emp_var_rate": 1.4,
            "cons_price_idx": 93.918,
            "cons_conf_idx": -42.7,
            "euribor_3m": 4.96,
            "nr_employed": 5228.1,
        },
        "aug": {
            "emp_var_rate": 1.4,
            "cons_price_idx": 93.444,
            "cons_conf_idx": -36.1,
            "euribor_3m": 4.965,
            "nr_employed": 5228.1,
        },
        "oct": {
            "emp_var_rate": -0.1,
            "cons_price_idx": 93.798,
            "cons_conf_idx": -40.4,
            "euribor_3m": 5,
            "nr_employed": 5195.8,
        },
        "nov": {
            "emp_var_rate": -0.1,
            "cons_price_idx": 93.2,
            "cons_conf_idx": -42,
            "euribor_3m": 4.406,
            "nr_employed": 5195.8,
        },
        "dec": {
            "emp_var_rate": -0.2,
            "cons_price_idx": 92.75,
            "cons_conf_idx": -45.9,
            "euribor_3m": 3.563,
            "nr_employed": 5176.3,
        },
    }

    index_2009 = {
        "jan": {"emp_var_rate": -0.2, "nr_employed": 5176.3},
        "feb": {"emp_var_rate": -0.2, "nr_employed": 5176.3},
        "mar": {
            "emp_var_rate": -1.8,
            "cons_price_idx": 92.84,
            "cons_conf_idx": -50,
            "euribor_3m": 1.811,
            "nr_employed": 5099.1,
        },
        "apr": {
            "emp_var_rate": -1.8,
            "cons_price_idx": 93.075,
            "cons_conf_idx": -47.1,
            "euribor_3m": 1.498,
            "nr_employed": 5099.1,
        },
        "may": {
            "emp_var_rate": -1.8,
            "cons_price_idx": 92.89,
            "cons_conf_idx": -46.2,
            "euribor_3m": 1.334,
            "nr_employed": 5099.1,
        },
        "jun": {
            "emp_var_rate": -2.9,
            "cons_price_idx": 92.963,
            "cons_conf_idx": -40.8,
            "euribor_3m": 1.26,
            "nr_employed": 5076.2,
        },
        "jul": {
            "emp_var_rate": -2.9,
            "cons_price_idx": 93.469,
            "cons_conf_idx": -33.6,
            "euribor_3m": 1.072,
            "nr_employed": 5076.2,
        },
        "aug": {
            "emp_var_rate": -2.9,
            "cons_price_idx": 92.201,
            "cons_conf_idx": -31.4,
            "euribor_3m": 0.884,
            "nr_employed": 5076.2,
        },
        "sep": {
            "emp_var_rate": -3.4,
            "cons_price_idx": 92.379,
            "cons_conf_idx": -29.8,
            "euribor_3m": 0.813,
            "nr_employed": 5017.5,
        },
        "oct": {
            "emp_var_rate": -3.4,
            "cons_price_idx": 92.431,
            "cons_conf_idx": -26.9,
            "euribor_3m": 0.754,
            "nr_employed": 5017.5,
        },
        "nov": {
            "emp_var_rate": -3.4,
            "cons_price_idx": 92.649,
            "cons_conf_idx": -30.1,
            "euribor_3m": 0.722,
            "nr_employed": 5017.5,
        },
        "dec": {
            "emp_var_rate": -3.0,
            "cons_price_idx": 92.713,
            "cons_conf_idx": -33,
            "euribor_3m": 0.718,
            "nr_employed": 5023.5,
        },
    }

    index_2010 = {
        "jan": {"emp_var_rate": -3.0, "nr_employed": 5023.5},
        "feb": {"emp_var_rate": -3.0, "nr_employed": 5023.5},
        "mar": {
            "emp_var_rate": -1.8,
            "cons_price_idx": 92.369,
            "cons_conf_idx": -34.8,
            "euribor_3m": 0.655,
            "nr_employed": 5008.7,
        },
        "apr": {
            "emp_var_rate": -1.8,
            "cons_price_idx": 93.749,
            "cons_conf_idx": -34.6,
            "euribor_3m": 0.64,
            "nr_employed": 5008.7,
        },
        "may": {
            "emp_var_rate": -1.8,
            "cons_price_idx": 93.876,
            "cons_conf_idx": -40,
            "euribor_3m": 0.668,
            "nr_employed": 5008.7,
        },
        "jun": {
            "emp_var_rate": -1.7,
            "cons_price_idx": 94.055,
            "cons_conf_idx": -39.8,
            "euribor_3m": 0.704,
            "nr_employed": 4991.6,
        },
        "jul": {
            "emp_var_rate": -1.7,
            "cons_price_idx": 94.215,
            "cons_conf_idx": -40.3,
            "euribor_3m": 0.79,
            "nr_employed": 4991.6,
        },
        "aug": {
            "emp_var_rate": -1.7,
            "cons_price_idx": 94.027,
            "cons_conf_idx": -38.3,
            "euribor_3m": 0.898,
            "nr_employed": 4991.6,
        },
        "sep": {
            "emp_var_rate": -1.1,
            "cons_price_idx": 94.199,
            "cons_conf_idx": -37.5,
            "euribor_3m": 0.886,
            "nr_employed": 4963.6,
        },
        "oct": {
            "emp_var_rate": -1.1,
            "cons_price_idx": 94.601,
            "cons_conf_idx": -49.5,
            "euribor_3m": 0.959,
            "nr_employed": 4963.6,
        },
        "nov": {
            "emp_var_rate": -1.1,
            "cons_price_idx": 94.767,
            "cons_conf_idx": -50.8,
            "euribor_3m": 1.05,
            "nr_employed": 4963.6,
        },
    }

    indx = [index_2008, index_2009, index_2010]
    years = [2008, 2009, 2010]

    for i in range(len(years)):
        for months, indexes in indx[i].items():
            for index, index_val in indexes.items():
                new_data = new_data.withColumn(
                    index,
                    when(
                        (col("year") == years[i]) & (col("month") == months), index_val
                    ).otherwise(col(index)),
                )
    return new_data


# In[16]:


# Calling the index_mapper function
bank_full = map_index(new_data=bank_full)
display(bank_full.limit(5))


# In[17]:


bank_full.printSchema()


# In[18]:


# dropping the balance, day column
bank_full = bank_full.drop("balance", "day")


# In[19]:


bank_full.head()


# In[20]:


# Converting the dataframe into the pandas dataframe
dataframe_1 = bank_full.toPandas()


# In[21]:


dataframe_1.head()


# In[22]:


temp_cols = dataframe_1.columns.tolist()
print(temp_cols)

index = dataframe_1.columns.get_loc("y")

reordered_cols = (
    temp_cols[:index] + temp_cols[index + 1 :] + temp_cols[index : index + 1]
)

dataframe_1 = dataframe_1[reordered_cols]
dataframe_1.head()


# In[24]:


dataframe_1.info()


# In[ ]:


# #### Bank Additional Full (dataset-2) <br>
#
# `bank-additional-full.csv` Bank Additional Full Preprocessing
#
# Header | Definition
# ---|---------
# `Age`| Age of the client
# `Job` | Job type of the client
# `Martial` | Martial status of the client
# `Education` | Client's highest education level
# `Default` |  Does the customer have credit in default?
# `Housing` | Does the client have housing loan
# `Loan` | Does the client have personal Loan
# `Contact` | Communication type of contact with customer
# `Month` |  Last contact month of year
# `Day_of_week` | Last contact day of the week
# `Duration` | Last contact duration (in seconds)
# `Campaign` | Number of contacts performed during this campaign and for this client
# `Pdays` | Number of days that passed by after the client was last contacted from a previous campaign
# `Previous` | Number of contacts performed before this campaign and for this client
# `Poutcome` | Outcome of the previous marketing campaign
# `Emp_var_rate` | Employment variation rate - quarterly indicator
# `Cons_price_idx` | Consumer price index - monthly indicator
# `Cons.conf.idx` | Consumer confidence index - monthly indicator
# `Euribor3m` | Euribor 3 month rate - daily indicator
# `Nr.employed` | Number of employees - quarterly indicator
# `y` | Has the client subscribed for a term deposit?

# In[25]:


# Read Spark Dataframe
bank_add_full = spark.read.csv(
    "bank-additional-full.csv", sep=";", header=True, inferSchema=True
)


# In[26]:


bank_add_full.count()


# In[27]:


display(bank_add_full.limit(5))


# In[28]:


bank_full_pdf = bank_add_full.toPandas()

# Apply the function to the Pandas DataFrame
new_pandas_df = year_mapper(bank_full_pdf, 2008, 2010)

# Convert the updated Pandas DataFrame back to a PySpark DataFrame
bank_add_full = spark.createDataFrame(new_pandas_df)


# ### Renaming columns names and values

# In[31]:


# Replacing the columns names
old_col_list = [
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
for i in range(0, len(old_col_list)):
    bank_add_full = bank_add_full.withColumnRenamed(old_col_list[i], new_cols[i])


# In[32]:


display(bank_add_full)


# ### Replace values from 999 to -1 in pdays column

# In[33]:


# changes the values from 999 to -1
bank_add_full = bank_add_full.withColumn(
    "pdays", when(col("pdays") == 999, -1).otherwise(col("pdays"))
)


# In[34]:


# Renaming the category names from education columns
old_edu = [
    "basic.4y",
    "high.school",
    "basic.6y",
    "basic.9y",
    "university.degree",
    "professional.course",
]
new_edu = [
    "basic_4y",
    "high_school",
    "basic_6y",
    "basic_9y",
    "university_degree",
    "professional_course",
]

for i in range(0, 6):
    bank_add_full = bank_add_full.withColumn(
        "education",
        when(col("education") == old_edu[i], new_edu[i]).otherwise(col("education")),
    )


# In[35]:


display(bank_add_full.limit(5))


# In[36]:


bank_full.printSchema()


# In[37]:


# dropping the column 'day_of week'
bank_add_full = bank_add_full.drop("day_of_week")


# In[38]:


dataframe_2 = bank_add_full.toPandas()


# In[39]:


dataframe_1.head()


# In[40]:


dataframe_2.head()


# In[42]:


dataframe_2.info()


# # MongoDB connectivity

# In[43]:


# create a mongo client for connection to host and port
try:
    client = MongoClient("localhost", 27017)
    print("Connection Successful!!")
except:
    print("Could not connect to MongoDB")


# In[44]:


# create a database or switch to an existing database
db_name = "telemarketing"
my_database = client[db_name]
print(my_database, end="\n\n")

# create a collection
collection_name = "bankmkt"
my_collection = my_database[collection_name]
print(my_collection)


# In[42]:


get_ipython().run_cell_magic(
    "time",
    "",
    "# to prevent unnecessary insertion into the MongoDB, using flag\nrun_MongoDB_insertion = False\nif run_MongoDB_insertion:\n    # Insert bank-full into the collection\n    try:\n        print(\"Attempting data insertion for bank-full\")\n        dataframe_1.reset_index(inplace = True)\n        df1_dicti = dataframe_1.to_dict(\"records\")\n        my_collection.insert_many(df1_dicti)\n\n        print(\"Dumping into MongoDB database '{0}' in collection '{1}' succesfull!\" \\\n              .format(db_name, collection_name))\n\n    except:\n        print(\"OOPs!! Attempt for insertion of bank-full data failed.\")\n        print(\"Dumping into MongoDB database '{0}' in collection '{1}' unsuccesfull...\" \\\n              .format(db_name, collection_name))\n    \n    print(100*'-')\n\n    # Insert bank-additional full into the collection   \n    try:\n        print(\"Attempting data insertion for bank-additional-full\")\n        dataframe_2.reset_index(inplace = True)\n        df2_dicti = dataframe_2.to_dict(\"records\")\n        my_collection.insert_many(df2_dicti)\n\n        print(\"Dumping into MongoDB database '{0}' in collection '{1}' succesfull!\"\\\n              .format(db_name, collection_name))\n\n    except:\n        print(\"OOPs!! Attempt for insertion of bank-additional-full data failed.\")\n        print(\"Dumping into MongoDB database '{0}' in collection '{1}' unsuccesfull....\"\\\n              .format(db_name,collection_name))\n    \n    print(100*'-')\n",
)


# In[45]:


my_collection.count_documents({})


# In[47]:


cursor = my_collection.find({}, {"_id": 0}).limit(5)
for document in cursor:
    pprint(document)


# In[48]:


get_ipython().run_cell_magic(
    "time",
    "",
    "bank = ps.DataFrame( list( my_collection.find( {}, { '_id' : 0 }  ) ) )\nbank.set_index( ['index'], inplace = True )\nbank.head()\n",
)


# In[49]:


bank.info()


# In[58]:


bank_data = bank.to_spark()


# In[60]:


display(bank_data)


# Concatenated two dataframes into a single MongoDB collection and <br>
# now reading directly from MongoDB's database

# In[61]:


bank_data.printSchema()


# In[62]:


bank_data.count()


# In[64]:


# prints the summary of dataframes with std, means and quartiles
bank_data.summary()


# In[65]:


# seperating the continuous and categorical variables
cat_col = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "year",
    "y",
]
cont_col = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor_3m",
    "nr_employed",
]

categories = bank_data.select(cat_col)
continuous = bank_data.select(cont_col)


# ### Unique value counts

# In[66]:


# prints the value counts for categorical columns
for columns in categories:
    print("Column Name", columns)
    print("-----------------------")
    counts = bank_data.groupBy(columns).count()
    counts.show()
    print("     ")
    print("******************************************************")
    print("     ")


# ## Data Preparation

# In[67]:


# Rename .admin category to admin
bank_data = bank_data.withColumn(
    "job", when(col("job") == "admin.", "admin").otherwise(col("job"))
)


# In[68]:


# Replacing "unknown" and "nonexistent" with the null values
for column in bank_data.columns:
    bank_data = bank_data.withColumn(
        column,
        when(col(column).isin("unknown", "nonexistent"), None).otherwise(col(column)),
    )


# In[69]:


display(bank_data.limit(5))


# ### Checking for null values

# In[70]:


# Checks the null values for categorical values
bank_data.agg(
    *[count(when(col(c).isNull(), c)).alias(c) for c in categories.columns]
).show()


# In[71]:


# Checks the null values for continuous values
bank_data.agg(
    *[count(when(col(c).isNull(), c)).alias(c) for c in continuous.columns]
).show()


# ### Replacing continue variables

# In[72]:


# calculate the mean of non-null values in columns
mean_dict = bank_data.select(*(mean(c).alias(c) for c in cont_col)).first().asDict()

# replace null values with the mean
bank_data = bank_data.fillna(mean_dict)


# In[73]:


# checking for null values
bank_data.agg(
    *[count(when(col(c).isNull(), c)).alias(c) for c in continuous.columns]
).show()


# ### Replacing categorical variables

# In[74]:


bank_data = bank_data.drop("poutcome")


# In[75]:


# calculate the mode of non-null values and replaced in columns

for column in cat_col:
    mode = (
        bank_data.groupBy(column)
        .agg(count("*").alias("count"))
        .orderBy(desc("count"))
        .select(column)
        .first()[0]
    )
    bank_data = bank_data.fillna({column: mode})


# In[76]:


# checking for null values
bank_data.agg(
    *[count(when(col(c).isNull(), c)).alias(c) for c in categories.columns]
).show()


# In[77]:


pdf = bank_data.toPandas()


# ### Heatmap

# In[78]:


correlation = pdf.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation)


# In[79]:


# function to plot the normal distribution
def plot_dist():
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 2, 1)
    plt.hist(pdf["age"], bins=40)
    plt.title("age")

    plt.subplot(3, 2, 2)
    plt.hist(pdf["duration"], bins=40)
    plt.title("duration")

    plt.subplot(3, 2, 3)
    plt.hist(pdf["campaign"], bins=40)
    plt.title("campaign")

    plt.subplot(3, 2, 4)
    plt.hist(pdf["pdays"], bins=40)
    plt.title("pdays")


plot_dist()


# ## ordinal data encoding

# In[80]:


# Creating a dictionary for converting categorical textual data entries
# conversion into categorical numeric values on basis on job profile
job_dict = {
    "entrepreneur": 11,
    "self-employed": 10,
    "admin": 9,
    "management": 8,
    "services": 7,
    "technician": 6,
    "blue-collar": 5,
    "housemaid": 4,
    "retired": 3,
    "student": 2,
    "unemployed": 1,
}

for key, value in job_dict.items():
    bank_data = bank_data.withColumn(
        "job", when(bank_data["job"] == key, int(value)).otherwise(bank_data["job"])
    )


# In[81]:


# conversion into categorical numeric values on basis on marital status
marital_dict = {"married": 3, "single": 2, "divorced": 1}

for key, value in marital_dict.items():
    bank_data = bank_data.withColumn(
        "marital",
        when(bank_data["marital"] == key, value).otherwise(bank_data["marital"]),
    )


# In[82]:


# conversion into categorical numeric values on basis on education
edu_dict = {
    "professional_course": 10,
    "university_degree": 9,
    "tertiary": 8,
    "secondary": 7,
    "high_school": 6,
    "basic_9y": 5,
    "basic_6y": 4,
    "primary": 3,
    "basic_4y": 2,
    "illiterate": 1,
}

for key, value in edu_dict.items():
    bank_data = bank_data.withColumn(
        "education",
        when(bank_data["education"] == key, value).otherwise(bank_data["education"]),
    )


# In[83]:


y_dict = {"yes": 1, "no": 0}

for key, value in y_dict.items():
    bank_data = bank_data.withColumn(
        "y", when(bank_data["y"] == key, value).otherwise(bank_data["y"])
    )


# In[84]:


display(bank_data.limit(5))


# In[85]:


# Conversion of months into the quarters
quarter_dict = {
    "jan": "Q1",
    "feb": "Q1",
    "mar": "Q1",
    "apr": "Q2",
    "may": "Q2",
    "jun": "Q2",
    "jul": "Q3",
    "aug": "Q3",
    "sep": "Q3",
    "oct": "Q4",
    "nov": "Q4",
    "dec": "Q4",
}

for key, value in quarter_dict.items():
    bank_data = bank_data.withColumn(
        "month", when(bank_data["month"] == key, value).otherwise(bank_data["month"])
    )


# In[86]:


bank_data.printSchema()


# In[87]:


bank = bank_data


# ### one hot encoding

# In[88]:


# One hot encoding on the nominal data
one_hot_cols = ["default", "housing", "loan"]

for i in one_hot_cols:
    bank = bank.withColumn(
        i, when(col(i) == "yes", 1).when(col(i) == "no", 0).otherwise(col(i))
    )

bank = bank.withColumn(
    "contact",
    when(col("contact") == "telephone", 1)
    .when(col("contact") == "cellular", 0)
    .otherwise(col("contact")),
)


# In[89]:


indexer = StringIndexer(inputCol="month", outputCol="class_numeric")
indexer_fitted = indexer.fit(bank)
df_indexed = indexer_fitted.transform(bank)


encoder = OneHotEncoder(inputCols=["class_numeric"], outputCols=["class_onehot"])
df_onehot = encoder.fit(df_indexed).transform(df_indexed)


df_col_onehot = df_onehot.select("*", vector_to_array("class_onehot").alias("Quarter"))


num_categories = len(df_col_onehot.first()["Quarter"])
cols_expanded = [
    (F.col("Quarter")[i].alias(f"{indexer_fitted.labels[i]}"))
    for i in range(num_categories)
]
bank_df = df_col_onehot.select(
    "year",
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor_3m",
    "nr_employed",
    "y",
    *cols_expanded,
)


# In[90]:


display(bank_df.limit(5))


# In[91]:


bank_df.printSchema()


# In[92]:


bank_data = bank_df


# In[93]:


bank_data = bank_data.drop("month")


# ### Converting string datatype to double

# In[94]:


column_types = bank_data.dtypes
# Filter the list to only include the string datatype columns

string_columns = [column[0] for column in column_types if column[1] == "string"]
print(string_columns)


# In[95]:


for cols in string_columns:
    # Change the datatype of the columns to double
    bank_data = bank_data.withColumn(cols, bank_data[cols].cast("double"))


# In[96]:


display(bank_data.limit(5))


# In[97]:


bank_data.printSchema()


# ### Outliers visualization and removal

# In[98]:


out = bank_data.toPandas()


# In[99]:


outliers_columns = ["age", "duration", "campaign", "pdays", "previous"]


# In[100]:


# Function defination to plot the outliers
def plot_box():
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 2, 1)
    out.boxplot(column=["age"])

    plt.subplot(3, 2, 2)
    out.boxplot(column=["duration"])

    plt.subplot(3, 2, 3)
    out.boxplot(column=["campaign"])

    plt.subplot(3, 2, 4)
    out.boxplot(column=["pdays"])

    plt.subplot(3, 2, 5)
    out.boxplot(column=["previous"])


plot_box()


# In[101]:


# Removing the outliers with the help of maximum binding limit
max_out_limit = []
for cols in outliers_columns:
    quantiles = bank_data.approxQuantile(cols, [0.25, 0.5, 0.75], 0.01)

    q3 = quantiles[2]
    q1 = quantiles[0]
    iqr = q3 - q1
    iqr = iqr * 1.5
    max_limit = q3 + iqr
    min_limit = q1 - iqr
    max_out_limit.append(max_limit)

    print(cols, "max_limit: ", max_limit, "      min_limit: ", min_limit)
else:
    print("------------------------------------------")
    print(max_out_limit)


# In[102]:


for i, j in zip(outliers_columns, max_out_limit):
    bank_data = bank_data.withColumn(i, when((col(i) >= j), j).otherwise(col(i)))


# In[103]:


out = bank_data.toPandas()


# In[104]:


plot_box()


# In[105]:


bank_data.printSchema()


# ## PySpark Model Building

# In[106]:


bank = bank_data


# In[107]:


display(bank.limit(5))


# In[108]:


bank.groupBy("y").count().show()


# *OBSERVATION :* <br><br>
# We have excess count of 0 i.e. (no) in 'y' so the dataset is imblanced. <br>
# Hence, we are going for oversampling of yes

# ## Data Oversampling of "yes" label in y

# In[109]:


# Data oversampling function of the "yes" lebel in the dependent column
def over_sample(data, oversampling_ratio):
    import math

    # avoid changing the original object accidentally
    new_data = data.toPandas()

    nums = new_data["y"].value_counts()
    num_n = nums[0]
    num_y = nums[1]
    nums = len(new_data["y"])

    new_y = ((1 - oversampling_ratio) / oversampling_ratio) * num_n

    loop_num = int(math.ceil(new_y / num_y))

    new_df = new_data[new_data["y"] == 1.0]

    for i in range(0, loop_num - 1):
        # randomly select all rows from new_df
        random_rows = new_df.sample(n=num_y, replace=True, random_state=14)

        # append the selected row to bank_df
        new_data = new_data.append(random_rows, ignore_index=True)

    new_data = spark.createDataFrame(new_data)
    return new_data


# In[110]:


bank_oversampled_df = over_sample(data=bank, oversampling_ratio=0.6)


# In[111]:


# Value counts for y column
bank_oversampled_df.groupBy("y").count().show()


# In[112]:


bank = bank_oversampled_df


# ## Data Scaling

# In[113]:


# Min_max scaling to the selected columns

numerical_cols = [
    "year",
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "duration",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor_3m",
    "nr_employed",
    "Q2",
    "Q3",
    "Q4",
]

# numerical_cols = ['year', 'age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'duration',
#'cons_price_idx', 'Q4']


# Create a vector assembler to combine the numerical columns into a single vector
assembler = VectorAssembler(inputCols=numerical_cols, outputCol="numerical_features")

# Transform the DataFrame to create the numerical features vector
bank = assembler.transform(bank)

# Apply MinMaxScaler to the numerical features vector
scaler = MinMaxScaler(
    inputCol="numerical_features", outputCol="scaled_numerical_features"
)
scaler_model = scaler.fit(bank)
df = scaler_model.transform(bank)

df = df.drop("numerical_features")

# Drop the original numerical columns and keep only the scaled numerical features
bank = df.drop(*numerical_cols).withColumnRenamed(
    "scaled_numerical_features", "sc_features"
)


# In[114]:


display(bank.limit(5))


# In[115]:


banks = bank


# In[116]:


display(banks.limit(5))


# In[117]:


# getting the names of columns of the dataframe
feature = []
for columns in banks.columns:
    feature.append(columns)
else:
    print(feature)


# In[118]:


banks.count()


# In[119]:


# train test split: train data percentage: 70% and test data percentage: 30%
train_data, test_data = banks.randomSplit([0.70, 0.30], seed=14)
display(train_data.limit(5))


# In[120]:


# train data counts
train_data.count()


# In[121]:


# test data counts
test_data.count()


# ## Decision Tree Classifier

# In[122]:


# Decision tree Algorithm


# In[123]:


dt = dtc(labelCol="y", featuresCol="sc_features")
dt_model = dt.fit(train_data)

# Predict the values on test data
dt_predictions = dt_model.transform(test_data)


# In[124]:


dt_pred = dt_predictions.select("prediction").toPandas()
actual = dt_predictions.select("y").toPandas()

# Shows confusion Report
ConfusionMatrixDisplay.from_predictions(actual, dt_pred)


# In[125]:


# Prints Classification Report
print(classification_report(actual, dt_pred))


# ## Logistic Regression

# In[126]:


# In[127]:


lr = LogisticRegression(featuresCol="sc_features", labelCol="y", maxIter=1000)
lr_model = lr.fit(train_data)
lr_predictions = lr_model.transform(test_data)


# In[128]:


lr_pred = lr_predictions.select("prediction").toPandas()
actual = lr_predictions.select("y").toPandas()

ConfusionMatrixDisplay.from_predictions(actual, lr_pred)


# In[129]:


print(classification_report(actual, lr_pred))


# ## Support Vector Machines

# In[130]:


# Load training data
lsvc = LinearSVC(featuresCol="sc_features", labelCol="y", maxIter=10, regParam=0.1)

# Fit the model
lsvcModel = lsvc.fit(train_data)

svc_predictions = lsvcModel.transform(test_data)

svc_pred = svc_predictions.select("prediction").toPandas()
actual = svc_predictions.select("y").toPandas()

ConfusionMatrixDisplay.from_predictions(actual, svc_pred)

print(classification_report(actual, svc_pred))


# ## Random Forest

# In[131]:


rf = RandomForestClassifier(labelCol="y", featuresCol="sc_features")
rf_model = rf.fit(train_data)
predictions = rf_model.transform(test_data)


# In[132]:


rf_prediction = rf_model.transform(test_data)
rf_preds = rf_prediction.select("prediction").toPandas()
actual = rf_prediction.select("y").toPandas()

ConfusionMatrixDisplay.from_predictions(actual, rf_preds)


# In[133]:


print(classification_report(actual, rf_preds))


# ## Grid Search CV

# ### Grid search_CV for Random forest

# In[134]:


# Define parameter grid
param_grid = (
    ParamGridBuilder()
    .addGrid(rf.maxDepth, [2, 5, 10])
    .addGrid(rf.numTrees, [20, 50, 100])
    .build()
)

# Define evaluator
evaluator = BinaryClassificationEvaluator(
    metricName="areaUnderROC", labelCol=lr.getLabelCol()
)

# Define cross-validator
cv = CrossValidator(
    estimator=rf, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5
)

# Fit cross-validator to training data
cv_model = cv.fit(train_data)

# Evaluate model on test data
gcv_pred = cv_model.transform(test_data)
evaluator.evaluate(predictions)


# In[135]:


gcv_preds = gcv_pred.select("prediction").toPandas()
gcv_actual = gcv_pred.select("y").toPandas()

ConfusionMatrixDisplay.from_predictions(gcv_actual, gcv_preds)


# In[136]:


print(classification_report(gcv_actual, gcv_preds))


# ### Grid search_CV for Decision Tree

# In[137]:


param_grid = (
    ParamGridBuilder()
    .addGrid(dt.maxDepth, [2, 5, 10])
    .addGrid(dt.maxBins, [16, 32, 64])
    .build()
)

# Define evaluator
evaluator = BinaryClassificationEvaluator(
    metricName="areaUnderROC", labelCol=dt.getLabelCol()
)

# Define cross-validator
cv = CrossValidator(
    estimator=dt, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5
)

# Fit cross-validator to training data
cv_model = cv.fit(train_data)

# Evaluate model on test data
gcv_dt_pred = cv_model.transform(test_data)
evaluator.evaluate(predictions)


# In[138]:


gcv_preds = gcv_dt_pred.select("prediction").toPandas()
gcv_actual = gcv_dt_pred.select("y").toPandas()

ConfusionMatrixDisplay.from_predictions(gcv_actual, gcv_preds)


# In[139]:


print(classification_report(gcv_actual, gcv_preds))


# In[ ]:
