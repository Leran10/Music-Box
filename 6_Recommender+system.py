
# coding: utf-8

# # Recommandation dataframe preparation

# from pyspark import SparkContext
# from pyspark.sql.session import SparkSession
# sc = SparkContext.getOrCreate("local")
# spark = SparkSession(sc)
# 
# import pyspark.sql.functions as F
# from matplotlib import pyplot as plt

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use("ggplot")


# In[5]:


df = pd.read_csv('data/play_ds.csv',dtype = str)


# In[6]:


df.head(20)


# In[7]:


df_rec = df[['uid','song_id','song_name','date']]


# In[8]:


df_rec = df_rec.dropna()


# get target data

# In[9]:


# extract the record for past 7 days
mask = (df_rec['date'] > '2017-05-06') & (df_rec['date'] <= '2017-05-12')

df_play_times = df_rec.loc[mask]


# In[10]:


# some of the user is not active, listened only less than 5 times in the past 7 days
df_active_counts = df_play_times['uid'].value_counts()


# In[11]:


active_users = df_active_counts[df_active_counts > 5]


# In[12]:


active_users.index


# In[13]:


df_cleaned = df_play_times.set_index('uid').ix[active_users.index].reset_index()


# In[14]:


df_cleaned.shape[0]


# In[15]:


df_more_play = df_cleaned['song_id'].value_counts()


# In[16]:


# the data is too large for mamory, so 
#keep the part that has been played more than 100 times
song = df_more_play[df_more_play > 30]


# In[17]:


song = song.astype(str)


# In[18]:


#df_cleaned = df_cleaned.set_index('song_id').ix[song.index].reset_index()

df_cleaned = df_cleaned.loc[df_cleaned['song_id'].isin(song.index)]

#song.index.dtype


# In[19]:


df_cleaned.info()


# In[20]:


df_cleaned.shape[0]


# In[21]:


df_cleaned = df_cleaned.groupby(['uid','song_id','song_name']).size().reset_index(name = 'counts')


# In[22]:


df_cleaned['song_id'].value_counts()


# In[23]:


# transfer "counts" to "like" by categorize them into several range
# if a user have listened to one sone less than 5 times in the past 7 days, I'll rate this song as '1'
# if from 5 times to 10 times, '2'
# if from 10 times to 20 times, '3'
# if from 20 times to 30 times, '4'
# if more than 30 times, '5'

df_cleaned['like'] = pd.cut(df_cleaned['counts'], [0,10,20,30,40,34285], labels = [1,2,3,4,5])


# In[24]:


df_cleaned


# In[25]:


# drop counts column
#df_cleaned.drop(['counts'],axis = 1, inplace = True)
df_cleaned.uid.nunique()


# # Create Utility Matrix from records

# In[26]:


df_cleaned['like'] = df_cleaned['like'].astype(int)
df_cleaned['uid'] = df_cleaned['uid'].astype(str)
df_cleaned['song_id'] = df_cleaned['song_id'].astype(str)


# In[27]:


df_utility = pd.pivot_table(data=df_cleaned,
                            values='like', # fill with stars
                            index='uid', # rows
                            columns='song_id', # columns
                            fill_value=0)


# In[28]:


# get the list of user id by checking out the index of the utility matrix
uid_list = df_utility.index
uid_list.shape


# In[29]:


# get the list of item id by checking out the columns of the utility matrix
song_id_list = df_utility.columns
#df_utility.columns[index]


# # recommendation system with Item-Item Collaborative filtering

# In[30]:


utility_mat = df_utility.as_matrix()


# In[31]:


from sklearn.metrics.pairwise import cosine_similarity
# Item-Item Similarity Matrix
item_sim_mat = cosine_similarity(utility_mat.T)


# calculate neighberhood

# In[32]:


least_to_most_sim_indexes = np.argsort(item_sim_mat, axis=1)

# Neighborhoods
neighborhood_size = 75
neighborhoods = least_to_most_sim_indexes[:, -neighborhood_size:]

neighborhoods


# #Make rating prediction on a user

# In[33]:


# Let's pick a lucky user
user_id = 5


# In[34]:


n_users = utility_mat.shape[0]
n_items = utility_mat.shape[1]


items_rated_by_this_user = utility_mat[user_id].nonzero()[0]
# Just initializing so we have somewhere to put rating preds
out = np.zeros(n_items)
for item_to_rate in range(n_items):
    relevant_items = np.intersect1d(neighborhoods[item_to_rate],
                                    items_rated_by_this_user,
                                    assume_unique=True)  # assume_unique speeds up intersection op
    out[item_to_rate] = np.dot(utility_mat[user_id, relevant_items],        item_sim_mat[item_to_rate, relevant_items]) /         item_sim_mat[item_to_rate, relevant_items].sum()

pred_ratings = np.nan_to_num(out)
print(pred_ratings)


# In[35]:


# get final reccomendation for a user


# In[36]:


# Recommend n movies
n = 10

# Get item indexes sorted by predicted rating
item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))[::-1]

# Find items that have been rated by user
items_rated_by_this_user = utility_mat[user_id].nonzero()[0]

# We want to exclude the items that have been rated by user
unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                if item not in items_rated_by_this_user]

unrated_index = unrated_items_by_pred_rating[:n]


# In[37]:


unrated_index


# # print the first 10 songs for user

# In[38]:


df_utility.columns


# In[39]:


print ('Top 10 songs recommended for the user:\n')
for index in unrated_index:
    for name in df_cleaned[df_cleaned['song_id']==df_utility.columns[index]]['song_name']:
        print (name)
        break


# # recommendation system with NMF

# In[40]:


from scipy import sparse
df_cleaned['uid'] = df_cleaned['uid'].astype(int)
df_cleaned['song_id'] = df_cleaned['song_id'].astype(int)

highest_user_id = df_cleaned.uid.max()
highest_song_id = df_cleaned.song_id.max()

ratings_mat = sparse.lil_matrix((highest_user_id, highest_song_id))

df_cleaned.info()
#ratings_mat.__dict__
for _, row in df_cleaned.iterrows():
    # subtract 1 from id's due to match 0 indexing
    ratings_mat[row.uid-1, row.song_id-1] = row.like



# In[41]:


from sklearn.decomposition import NMF

def fit_nmf(M,k):
    nmf = NMF(n_components=k)
    nmf.fit(M)
    W = nmf.transform(M);
    H = nmf.components_;
    err = nmf.reconstruction_err_
    return W,H,err

# decompose
W,H,err = fit_nmf(ratings_mat,200)
print(err)
print(W.shape,H.shape)


# In[88]:


type(utility_mat)


# In[93]:


# reconstruct# recon 
ratings_mat_fitted = W.dot(H)
errs = np.array((ratings_mat-utility_mat_fitted).flatten()).squeeze()
mask = np.array((ratings_mat.todense()).flatten()).squeeze()>0

mse = np.mean(errs[mask]**2)
average_abs_err = abs(errs[mask]).mean()
print(mse)
print(average_abs_err)


# In[ ]:


# get recommendations for one user
user_id = 100
n = 10

pred_ratings = ratings_mat_fitted[user_id,:]
item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))[::-1]

items_rated_by_this_user = utility_mat[user_id].nonzero()[1]

unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                if item not in items_rated_by_this_user]

unrated_items_by_pred_rating[:n]


# In[ ]:


ratings_true = ratings_mat[user_id, items_rated_by_this_user].todense()
# prediction
ratings_pred = pred_ratings[items_rated_by_this_user]
print(list(zip(np.array(ratings_true).squeeze(),ratings_pred)))
err_one_user = ratings_true-ratings_pred
print(err_one_user)
print(abs(err_one_user).mean())


# # Recommendation system with SVD/UVD

# In[ ]:


from sklearn.decomposition import TruncatedSVD

def fit_uvd(M,k):
    # use TruncatedSVD to realize UVD
    svd = TruncatedSVD(n_components=k, n_iter=7, random_state=0)
    svd.fit(M)

    V = svd.components_
    U = svd.transform(M) # effectively, it's doing: U = M.dot(V.T)
    # we can ignore svd.singular_values_ for our purpose
    
    # why we can do this?
    # recall: 
    # SVD start from u*s*v=M => u*s=M*v.T, where M*v.T is our transformation above to get U in UVD
    # so the above U is effectively u*s in SVD
    # that's why U*V = u*s*v = M our original matrix
    # there are many ways to understand it!
    # here we by-passed singular values.
    
    return U,V, svd

# decompose
U,V,svd = fit_uvd(ratings_mat,200)


# In[ ]:


# reconstruct
ratings_mat_fitted = U.dot(V) # U*V


# recall: U = M.dot(V.T), then this is M.dot(V.T).dot(V)
# original M is transformed to new space, then transformed back
# this is another way to understand it!

# calculate errs
errs = np.array((ratings_mat-ratings_mat_fitted).flatten()).squeeze()
mask = np.array((ratings_mat.todense()).flatten()).squeeze()>0

mse = np.mean(errs[mask]**2)
average_abs_err = abs(errs[mask]).mean()
print(mse)
print(average_abs_err)


# In[ ]:


# compare with another way to reconstruct matrix
# with the above "tranformed to the new space and back" language
# without the UV language, we can do:

# reconstruct M with inverse_transform
ratings_mat_fitted_2 = svd.inverse_transform(svd.transform(ratings_mat))
ratings_mat_fitted = U.dot(V)
print(sum(sum(ratings_mat_fitted - ratings_mat_fitted_2)))
# they are just equivalent!!


# In[ ]:



# get recommendations for one user# get r 
user_id = 100
n = 10

pred_ratings = ratings_mat_fitted[user_id,:]
item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))[::-1]

items_rated_by_this_user = ratings_mat[user_id].nonzero()[1]

unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                if item not in items_rated_by_this_user]

unrated_items_by_pred_rating[:n]


# In[ ]:


### check errors
# truth
ratings_true = ratings_mat[user_id, items_rated_by_this_user].todense()
# prediction
ratings_pred = pred_ratings[items_rated_by_this_user]
print(list(zip(np.array(ratings_true).squeeze(),ratings_pred)))
err_one_user = ratings_true-ratings_pred
print(err_one_user)
print(abs(err_one_user).mean())

