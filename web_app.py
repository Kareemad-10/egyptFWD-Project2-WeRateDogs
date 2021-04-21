# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 18:11:36 2020

@author: Kareem E.Farouk
"""


import pandas as pd
import streamlit as st
import web_app_modules as md
import os

st.title('Project 2 - WeRateDogs Tweets Analysis \n')

st.write('> ### WeRateDogs is a twitter account that rates peoples\' dogs with humorous comments about dogs.\n',\
         '> ### Our task is to pull some tweets from this account using tweepy package for some analysis. *That\'s it!*\n'
         )
st.image('we.jpg')
            
df_archive, df_image_prediction, df_api = md.gather()
st.write('### Before Cleaning Data:')
st.write('*Source 1: Main tweets data*:   **local .csv file. **')
st.dataframe(df_archive)
st.write('*Source 2: Some data about tweets images using NN algorithm*:  **local .tsv file downloaded programmatically from Udacity servers.**')
st.dataframe(df_image_prediction)
st.write('*Source 3*:  **Additional data about these tweets extracted from twitter api: *retweet_count*, *favorite_count* **')
st.dataframe(df_api)
st.write('## Main Quality & Tidness issues detected: as the following: ')
st.write('''
         
        Quality Issues (Content-Related)
----------------------------------------------------------------------------------
#### df_archive:
----------------------------------------------------------------------------------
1. There exist many tweets in the table that are retweets or replies that we should filter out to satisfy our project requirments.
---------------------------------------------------
2. many missing dog stage data about doggo, floofer, pupper, puppo.
----------------------------------------------------------------------------------
3. expanded_urls have multiple identical (repeated) urls separated with commas(,) .
----------------------------------------------------------------------------------
4. expanded_urls in some records have no value at all.
----------------------------------------------------------------------------------
5. many NaNs at the column: in_reply_to_status_id, in_reply_to_user_id, retweeted_status_id, retweeted_status_timestamp, retweeted_status_user_id (Normal situation because some tweets are replies and/or retweets) so that this field will be empty.
----------------------------------------------------------------------------------
6. name column has strange or weird values like: 'None', 'a', 'an', 'the', 'not' + many original tweets have no name value at all like tweet # 27.
----------------------------------------------------------------------------------
7. rating_denominator in some tweets has a strange values much greater than [ 10 ] which is defined for the unique rating system that is a big part of the popularity of WeRateDogs. this can be detected visually and programatically.
----------------------------------------------------------------------------------
8. Extremely Much Null values in the columns doggo, floofer, pupper, puppo . [can be resolved by changing the structure of the df_archive table .]
----------------------------------------------------------------------------------
9. timestamp column has incorrect datatype (object); it should be of a datetime datatype.
----------------------------------------------------------------------------------
10. there exist non-original tweets (retweets and/or replies) for another tweets that violate the requirments of the project for analysis detected through visually through RT and @ charaters in the tweet text.
----------------------------------------------------------------------------------
11. source column has non descriptive name, it should be like: source_tweet for example.
----------------------------------------------------------------------------------
#### df_image_predictions:
----------------------------------------------------------------------------------
1. non describtive column names for p1, p1_conf, p1_dog and so on for p2,p3.
----------------------------------------------------------------------------------
2. there are missing image predictions for 281 tweets which is the difference in no_rows between df_archive table and the df_image_predictions table . this problem can't be solved because we have no access on the neural network algorithm.
----------------------------------------------------------------------------------
   
Tidness Issues (Structure-Related)   
-----------------------------------------------------------------------------------
#### df_archive:
------------------------------------------------------------------------------------    
1. doggo, floofer, pupper, puppo represent values of a single variable which is: dog_stage, but they appear separated in multiple columns.
----------------------------------------------------------------------------------
#### df_tweets_api:
------------------------------------------------------------------------------------
retweet_count, favorite_count columns should be part of the df_archive table as they belong to the data/features of the tweet.

         ''')
         
#df_archive_clean, df_image_prediction_clean = md.Clean(df_archive, df_image_prediction, df_api) 
#md.store(df_archive_clean, df_image_prediction_clean)        
st.write('### After Cleaning Data:')
st.write('**Cleaned .csv file. **')
df_archive_clean = pd.read_csv('tweets.csv')
df_image_prediction_clean = pd.read_csv('image_predict.csv')

st.dataframe(df_archive_clean)
st.write('**Cleaned .tsv file **')
st.dataframe(df_image_prediction_clean)

if not os.path.isfile('project_2.db'):
       md.Store(df_archive_clean, df_image_prediction_clean)

df_tweets, df_images = md.Retrieve()
df_tweets.timestamp = pd.to_datetime(df_tweets.timestamp)
#print(df_tweets.shape, df_images.shape, df_images.info())
#print(df_image_prediction.info())
md.Analyze(df_tweets, df_images)