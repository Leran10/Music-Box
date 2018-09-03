
# coding: utf-8

# # Load data into Spark DataFrame

# In[85]:


from pyspark import SparkContext


# In[87]:


from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)


# In[3]:


import pyspark.sql.functions as F


# In[4]:


# We use matplotlib for plotting
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# In[6]:


df = spark.read.csv('data/event_ds.csv',header=True).cache()
df


# In[8]:


df.show()


# In[9]:


# create new or overwrite original field with withColumn
df = df.withColumn('date',F.col('date').cast('date'))
df


# In[10]:


df.show()


# # Some exploration

# In[11]:


# simple count rows
df.count()


# In[12]:


# select operation, count distinct rows
df.select('uid').distinct().count()


# In[13]:


# group by aggregation
df.groupBy('event').count().show()


# In[14]:


# group by aggregation, more general (count, min, max, mean), multiple at once
df.groupBy('event').agg(
    F.count(F.col('uid')).alias('count'),
    F.max(F.col('uid')).alias('max_uid')
).show()


# In[15]:


# filter operation
# group by aggregation
# order by operation
df.filter((F.col('date')>='2017-04-01') & (F.col('date')<='2017-04-05'))                     .groupBy('date','event').count()                     .orderBy('date','event').show()


# In[16]:


date_count = df.groupBy('date').count().toPandas()


# In[17]:


plt.bar(date_count['date'],date_count['count'])
plt.xticks(rotation='vertical')


# # Label definition

# In[18]:


import datetime
from dateutil import parser

label_window_size = 14
label_window_end_date = parser.parse('2017-05-12').date()
label_window_start_date = label_window_end_date - datetime.timedelta(label_window_size - 1)
print('label window:',label_window_start_date,'~',label_window_end_date,'days:',label_window_size)

feature_window_size = 30
feature_window_end_date = label_window_start_date - datetime.timedelta(1)
feature_window_start_date = feature_window_end_date  - datetime.timedelta(feature_window_size - 1)
print('feature window:',feature_window_start_date,'~',feature_window_end_date,'days:',feature_window_size)


# In[19]:


# all the uid we will model
df_model_uid = df.filter((F.col('date')>=feature_window_start_date) & (F.col('date')<=feature_window_end_date))                    .select('uid').distinct()
# active in label window (active label=0)
df_active_uid_in_label_window = df.filter((F.col('date')>=label_window_start_date) & (F.col('date')<=label_window_end_date))                            .select('uid').distinct().withColumn('label',F.lit(0))


# In[20]:


# prepare label data (churn label=1; active label=0)
df_label = df_model_uid.join(df_active_uid_in_label_window,on=['uid'],how='left')
df_label = df_label.fillna(1)


# In[21]:


df_label.groupBy('label').count().show()


# # Feature generation

# In[22]:


# event_data in feature_window
df_feature_window = df.filter((F.col('date')>=feature_window_start_date) & (F.col('date')<=feature_window_end_date))


# ### Frequency features

# ##### method 1

# In[23]:


# define a function to generate frequency features
def frequency_feature_generation(df,event,time_window,snapshot_date):
    """
    generate frequency features for one event type and one time window
    """
    df_feature = df.filter(F.col('event')==event)            .filter((F.col('date')>=snapshot_date-datetime.timedelta(time_window-1)) & (F.col('date')<=snapshot_date))            .groupBy('uid').agg(F.count(F.col('uid')).alias('freq_'+event+'_last_'+str(time_window)))
    return df_feature


# In[24]:


# generate one feature
event = 'S'
time_window = 3
snapshot_date = feature_window_end_date
df_feature = frequency_feature_generation(df_feature_window,event,time_window,snapshot_date)


# In[25]:


df_feature.show(5)


# In[26]:


# generate frequency features for all event_list, time_window_list
event_list = ['P','D','S']
time_window_list = [1,3,7,14,30]
df_feature_list = []
for event in event_list:
    for time_window in time_window_list:
        df_feature_list.append(frequency_feature_generation(df_feature_window,event,time_window,snapshot_date))


# In[27]:


df_feature_list


# ##### method 2: too many dfs to join? do it another way

# In[28]:


# define a function to generate frequency features for a list of time windows
# using when().otherwise(), and list comprehension trick!
def frequency_feature_generation_time_windows(df,event,time_window_list,snapshot_date):
    """
    generate frequency features for one event type and a list of time windows
    """
    df_feature = df         .filter(F.col('event')==event)         .groupBy('uid')         .agg(*[F.sum(F.when((F.col('date')>=snapshot_date-datetime.timedelta(time_window-1)) & (F.col('date')<=snapshot_date),1).otherwise(0)).alias('freq_'+event+'_last_'+str(time_window))                 for time_window in time_window_list]
            )# *[] opens list and make them comma separated
    return df_feature


# In[29]:


# generate one event type, all time windows 
event = 'S'
time_window_list = [1,3,7,14,30]
snapshot_date = feature_window_end_date
df_feature = frequency_feature_generation_time_windows(df_feature_window,event,time_window_list,snapshot_date)
df_feature.show(5)


# In[30]:


# generate frequency features for all event_list, time_window_list
event_list = ['P','D','S']
time_window_list = [1,3,7,14,30]
df_feature_list = []
for event in event_list:
    df_feature_list.append(frequency_feature_generation_time_windows(df_feature_window,event,time_window_list,snapshot_date))


# In[31]:


df_feature_list


# ### Recency features

# In[32]:


# defined as days from last event
# can generate one feature for each type of event

from pyspark.sql.functions import collect_list, sort_array
from pyspark.sql.functions import datediff, to_date, lit


# In[33]:


#df.groupBy("uid","event").count().show()
def genarate_recency_feature(event,snapshot_date):
    
   
    df_grouped = df.filter(F.col("event") == "P").groupBy("uid").agg(F.collect_set("event").alias("event"),F.collect_list("date").alias("recent"))
    df_sorted = df_grouped.withColumn("recent",sort_array("recent",asc = False))
    #spark.sql('set spark.sql.caseSensitive=true')
    df_last_time = df_sorted.selectExpr("uid","recent[0]")

    df_recency_feature = df_last_time.withColumn('rec_'+event,datediff(to_date(lit(snapshot_date)),"recent[0]"))
    df_recency_feature.drop(F.col('recent[0]'))

    return  df_recency_feature


# In[34]:


events = ['S','D','P']
snapshot_date = "2017-05-12"
df_feature_list2 = []

for event in events:
      df_feature_list2.append(genarate_recency_feature(event,snapshot_date))


# In[35]:


df_feature_list2


# ### Profile features

# In[36]:


df_play = spark.read.csv('data/play_ds.csv',header=True)
df_play.show(5)


# In[37]:


df_play_feature_window = df_play.filter((F.col('date')>=feature_window_start_date) & (F.col('date')<=feature_window_end_date))
df_profile_tmp = df_play_feature_window.select('uid','device').distinct()


# In[43]:


df_profile_tmp.groupBy('device').count().show()


# In[38]:


# check if one user has two devices
df_profile_tmp.count()


# In[39]:


df_profile_tmp.distinct().count()


# In[40]:


df_profile_tmp = df_profile_tmp.withColumn('device_type',F.when(F.col('device')=='ip',1).otherwise(2))
df_profile_tmp.groupBy('device_type').count().show()


# In[41]:


df_profile = df_label.select('uid').join(df_profile_tmp.select('uid','device_type'),on='uid',how='left')
df_profile.groupBy('device_type').count().show()


# ### Total play time features

# In[42]:


# generate total song play time features

df_play = spark.read.csv('data/play_ds.csv',header=True).cache()

df_play_new = df_play.selectExpr("uid","play_time","song_length","date")


# In[43]:


#check missing values
df_play_new.where(F.col("play_time").isNull()).count()

df_play_new.where(F.col("song_length").isNull()).count()

# fill na with 0
df_play_nonull = df_play_new.na.fill('0')

df_play_nonull.where(F.col("uid").isNull()).count()


# In[44]:


df_play_time = df_play_nonull.selectExpr("uid","play_time","date")


# In[45]:


def generate_total_play_feature(time_window_list,snapshot_date):
        df_total_play_feature = df_play_time.filter((F.col('date')>=snapshot_date-datetime.timedelta(time_window-1)) & (F.col('date')<=snapshot_date)).                                groupBy("uid").agg(F.sum("play_time").alias("total_time_last_" + str(time_window)))
                                                
        return  df_total_play_feature


# In[46]:


time_window_list = [1,3,7,14,30]
df_feature_list3 = []
snapshot_date = feature_window_end_date

for time_window in time_window_list:
    df_feature_list3.append(generate_total_play_feature(time_window_list,snapshot_date))


# In[47]:


df_feature_list3


# ### Fancier frequency features

# In[48]:


# generate counts of songs play 80% of their song length 

df_play = spark.read.csv('data/play_ds.csv',header=True).cache()


# In[49]:


df_play_date = df_play.selectExpr("uid","play_time","song_length","date").show()


# In[50]:


df_play_new = df_play.selectExpr("uid","play_time","song_length","date")
df_play_new.dtypes


# In[51]:


#check missing values
df_play_new.where(F.col("play_time").isNull()).count()

df_play_new.where(F.col("song_length").isNull()).count()

# fill na with 0
df_play_nonull = df_play_new.na.fill('0')

df_play_nonull.where(F.col("uid").isNull()).count()


# In[52]:


def generate_play_feature(time_window_list,snapshot_date):        
        
        # fill na in columns

        df_play_date = df_play_nonull.filter((F.col('date')>=snapshot_date-datetime.timedelta(time_window-1)) & (F.col('date')<=snapshot_date)).                       selectExpr("uid","play_time","song_length","date").                       withColumn("play_percentage", F.col("play_time") / F.col("song_length"))
        df_play_feature = df_play_date.groupBy("uid").agg(F.count(F.col("play_percentage") > 0.8).alias("play_perc_in_last" + str(time_window)))
        
        return df_play_feature


# In[53]:


time_window_list = [1,3,7,14,30]
df_feature_list4 = []
snapshot_date = feature_window_end_date

for time_window in time_window_list:
    df_feature_list4.append(generate_play_feature(time_window_list,snapshot_date))


# In[54]:


df_feature_list4


# # Form training data

# In[55]:


def join_feature_data(df_master,df_feature_list,df_feature_list2,df_feature_list3,df_feature_list4):
    for df_feature in df_feature_list:
        df_master = df_master.join(df_feature,on='uid',how='left')
            #df_master.persist() # uncomment if number of joins is too many
    for df_feature in df_feature_list2:
        df_master = df_master.join(df_feature,on='uid',how='left')
        
    for df_feature in df_feature_list3:
        df_master = df_master.join(df_feature,on='uid',how='left') 
    
    for df_feature in df_feature_list4:
        df_master = df_master.join(df_feature,on='uid',how='left')
        
    return df_master
    


# In[56]:


# join all behavior features
df_model_final = join_feature_data(df_label,df_feature_list,df_feature_list2,df_feature_list3,df_feature_list4)


# In[57]:


# join all profile features
df_model_final = df_model_final.join(df_profile,on='uid',how='left')


# In[88]:


df_model_final.where(F.col("uid").isNull()).sum().sum()


# In[75]:


df_model_final = df_model_final.drop('recent[0]','recent[0]','recent[0]')


# In[89]:


df_model_final.fillna(0)


# In[74]:


df_model_final.toPandas().to_csv('data/df_model_final1.csv',index=False)

