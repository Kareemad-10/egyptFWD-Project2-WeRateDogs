# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:35:03 2020

@author: Kareem E.Farouk
"""
import pandas as pd
import numpy as np
import requests
import os
import json
import matplotlib.pyplot as plt
import streamlit as st

def gather():
    # Gather data from multiple sources.
    
    # Source1: already downloaded .csv file.
    df_archive = pd.read_csv('twitter-archive-enhanced.csv')
    
    # Source2:  .tsv file downloaded programmatically.
    url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
    response = requests.get(url)
    response
    file_name = url.split('/')[-1]
    
    if not os.path.isfile(file_name):
           with open(file_name, mode='wb') as file:
                file.write(response.content)
    df_image_prediction = pd.read_csv(file_name, sep='\t')
    
    # Source3: Connect into Twitter api for gathering data about the tweets that we have already using Ids.
    df_tweets = []
    tweets_with_no_entities = []
    with open('tweet_json.txt', mode='r') as file:
         for line in file:
             data = json.loads(line)
             try:
                temp = data['entities']['media'][0]['expanded_url']
                df_tweets.append({
                                 'retweet_count':data['retweet_count'],
                                 'favorite_count':data['favorite_count'],
                                 'expanded_url': temp
                                 }
                                )
             except Exception as e:
                 tweets_with_no_entities.append((data['id'], e))          
        
         df_api = pd.DataFrame(df_tweets)
         return df_archive, df_image_prediction, df_api


def Clean(df_archive, df_image_prediction, df_api):
    df_archive_clean = df_archive.copy()
    df_image_prediction_clean = df_image_prediction.copy() 
    df_api_clean = df_api.copy()
    
    new_col = np.repeat('None', df_archive_clean.shape[0])
    df_archive_clean['other_col'] = new_col
    indices_selected = df_archive_clean[(df_archive_clean.doggo=='None') & (df_archive_clean.floofer=='None') & (df_archive_clean.pupper=='None') & (df_archive_clean.puppo=='None')]
    df_archive_clean.loc[indices_selected.index.tolist(), 'other_col'] = 'other'
    
    list_cols = list(df_archive_clean.columns)[:-5]
    df_archive_clean = pd.melt(df_archive_clean, id_vars=list_cols, var_name='dog_stage', value_name='value_stage')
    df_archive_clean = df_archive_clean[df_archive_clean.value_stage != 'None']
    df_archive_clean.dog_stage.str.replace('other_col', 'None')
    df_archive_clean.dog_stage = df_archive_clean.dog_stage.str.replace('other_col', 'None')
    df_archive_clean.drop('value_stage', axis=1, inplace=True)
   
    df_archive_clean = pd.merge(df_archive_clean, df_api_clean, on=['tweet_id'], how = 'inner')   
    df_archive_clean = df_archive_clean[df_archive_clean.in_reply_to_status_id.isnull() & df_archive_clean.retweeted_status_id.isnull()]
    df_archive_clean.drop(['in_reply_to_status_id', 'in_reply_to_user_id', 'retweeted_status_id', 'retweeted_status_timestamp', 'retweeted_status_user_id'], axis=1, inplace=True)
    df_archive_clean.loc[:,'rating_denominator'] = 10
    list_indices = df_archive_clean[df_archive_clean.rating_numerator >= 14].rating_numerator.index.tolist()
    df_archive_clean.loc[list_indices, 'rating_numerator'] = round(df_archive_clean.rating_numerator.describe()['mean'])
    list_indices = df_archive_clean[df_archive_clean.rating_numerator == 0].rating_numerator.index.tolist()
    df_archive_clean.loc[list_indices, 'rating_numerator'] = 1  # as a minimum rating other than zero.
    df_archive_clean.timestamp = pd.to_datetime(df_archive_clean.timestamp)
    df_archive_clean.rename(columns={'source':'source_tweet'}, inplace=True)
    df_archive_clean.source_tweet = df_archive_clean.source_tweet.apply(lambda x: x.split('>')[1][:-3])
    record = df_archive_clean[df_archive_clean.tweet_id==765395769549590528]
    idx_name = record.text.str.find('Zoey')
    name_extracted = record.text.str[idx_name.values[0]:idx_name.values[0]+4]
    df_archive_clean.loc[record.index, 'name'] = str(name_extracted.values[0])
    df_archive_clean.name = df_archive_clean.name.apply(lambda x: 'None' if x in ['a', 'an', 'the', 'not', 'by', 'old', 'this'] else x)
    df_archive_clean.name = df_archive_clean.name.apply(lambda x: 'None' if x.islower() else x)
    df_archive_clean.drop('expanded_urls', axis=1, inplace=True)

    df_image_prediction_clean.rename(columns={'p1':'first_prediction', 'p1_conf':'predict_confidence_1', 'p1_dog':'predict_dog_1',
                                              'p2':'second_prediction', 'p2_conf':'predict_confidence_2', 'p2_dog':'predict_dog_2',
                                              'p3':'third_prediction', 'p3_conf':'predict_confidence_3', 'p3_dog':'predict_dog_3',
                                              }, inplace=True)
    df_temp = pd.merge(df_image_prediction_clean, df_archive_clean, on='tweet_id', how='right')
    df_temp = df_temp.loc[:,~df_temp.columns.duplicated()]
    df_image_prediction_clean = df_temp.loc[:, 'tweet_id':'predict_dog_3']
    list_index = df_archive_clean[df_archive_clean.tweet_id.duplicated()].index
    df_archive_clean.drop(list_index, inplace=True)
    list_indexes = df_image_prediction_clean[df_image_prediction_clean.tweet_id.duplicated()].index
    df_image_prediction_clean.drop(list_indexes, inplace=True)
    
    
    return df_archive_clean, df_image_prediction_clean


def Store(df_archive_clean, df_image_prediction_clean):
    from sqlalchemy import create_engine
    
    engine = create_engine('sqlite:///project_2.db')
    df_archive_clean.to_sql('tweets', engine, index=False)
    df_image_prediction_clean.to_sql('image_predict', engine, index=False)
    
def Retrieve():
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///project_2.db')
    
    df_tweets = pd.read_sql('SELECT * FROM tweets', engine)
    df_images = pd.read_sql('SELECT * FROM image_predict', engine)
    
    return df_tweets, df_images

# a function to apply complicated selection task from the dataframe using .apply() method.

dog_breed = []
dog_conf = []
def extract_breeds(df):
    if df.predict_dog_1 == 1:
       dog_breed.append(df.first_prediction)
       dog_conf.append(df.predict_confidence_1)
    elif df.predict_dog_2 == 1:
       dog_breed.append(df.second_prediction)
       dog_conf.append(df.predict_confidence_2)
    elif df.predict_dog_3 == 1:
       dog_breed.append(df.third_prediction)
       dog_conf.append(df.predict_confidence_3)
       
def Analyze(df_tweets, df_images): 
    df_images = df_images[(df_images.predict_dog_1 != 0) | (df_images.predict_dog_2 != 0) | (df_images.predict_dog_3 != 0)]
    df_images.apply(extract_breeds, axis=1)

    df_images['dog_breed'] = dog_breed
    df_images['dog_breed_conf'] = dog_conf
    print(df_images.shape, df_images.info())  # we have 113 dog breeds.
    print(df_images.dog_breed.value_counts())
    
    df_breeds = pd.merge(df_images, df_tweets, on='tweet_id', how='inner')
    df_breeds = df_breeds[:][['tweet_id', 'jpg_url', 'dog_breed', 'dog_breed_conf', 'timestamp', 'text', 'rating_numerator', 'retweet_count', 'favorite_count', 'expanded_url']]
    st.write('### The table after merging by tweet id:')
    st.dataframe(df_breeds)
    st.write('### The tweet with the most No. of favorite counts:')
    idx_favorite_img = df_breeds.favorite_count.idxmax()
    st.write(df_breeds.iloc[idx_favorite_img, :])
    st.write('### The tweet with the most No. of retweet counts:')
    idx_retweet_img = df_breeds.retweet_count.idxmax()
    st.write(df_breeds.iloc[idx_retweet_img, :])
    st.write('### Top 5 dog breeds liked most by total of both *favorite count* and *retweet count*:')
    st.write(df_breeds.groupby('dog_breed')[['retweet_count','favorite_count']].sum().sort_values(by='favorite_count', ascending=False).iloc[0:5, :])
    caption_str = ['golden_retriever', 'labrador_retriever', 'pembroke', 'chihuahua', 'samoyed']
    imgs = ['golden.jpg', 'labrador.jpg', 'pembroke.jpg', 'chihuawaa.jpg', 'samoyed.jpg']
    st.image(imgs[0], caption=caption_str[0],
                use_column_width=True)
    st.image(imgs[1], caption=caption_str[1],
                use_column_width=True)
    st.image(imgs[2], caption=caption_str[2],
                use_column_width=True)
    st.image(imgs[3], caption=caption_str[3],
                use_column_width=True)
    st.image(imgs[4], caption=caption_str[4],
                use_column_width=True)
    
    fig,ax = plt.subplots()
    ax.scatter(x=df_breeds.favorite_count, y=df_breeds.retweet_count)
    ax.set_xlabel('favorite count')
    ax.set_ylabel('retweet count')
    fig.suptitle('Relation between retweet count, favorite count of tweets')
    st.pyplot(fig)
    #-------------------------------------
    fig1, ax = plt.subplots()
    ax.scatter(x=df_breeds.retweet_count, y=df_breeds.rating_numerator)
    ax.set_xlabel('retweet count')
    ax.set_ylabel('rating')
    fig1.suptitle('Relation between retweet count, rating of tweets')
    st.pyplot(fig1)
    #-------------------------------------
    fig2, ax = plt.subplots()
    ax.scatter(x=df_breeds.favorite_count, y=df_breeds.rating_numerator)
    ax.set_xlabel('favorite count')
    ax.set_ylabel('rating')
    fig2.suptitle('Relation between favorite count, rating of tweets')
    st.pyplot(fig2)
    #-------------------------------------
    
    df_data_final = df_breeds.groupby('dog_breed')[['retweet_count','favorite_count']].sum().sort_values(by='favorite_count', ascending=False).iloc[0:5, :]
    st.subheader('retweet count, favorite count per dog breed (top 5 only)')
    st.bar_chart(df_data_final[:]['favorite_count'])
    #-------------------------------------
    
    st.bar_chart(df_data_final[:]['retweet_count'])
    
    #-------------------------------------
    
    fig4, ax = plt.subplots()
    se_timestamp = df_tweets.timestamp.dt.month.value_counts()
    locations = np.arange(1, 13)
    labels = ['Dec', 'Nov', 'Jan', 'Feb', 'Mar', 'July', 'June', 'May', 'Apr', 'Oct', 'Sep', '   August']
    heights = se_timestamp.values
    plt.bar(locations, heights, tick_label=labels)
    plt.title('Month of the year with the the most no. of tweets')
    plt.xlabel('Month') 
    plt.ylabel('No. of tweets per each month')
    st.pyplot(fig4)
    #-------------------------------------
    
    