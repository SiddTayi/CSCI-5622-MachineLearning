# Import libraries

import pandas as pd
import pandas as pd
from googleapiclient.discovery import build 
import seaborn as sns
import random
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# API KEY details
# Confidential 
# Denoting the API key value as 'key'
# -----------------------------------------------------------------

# GET youtube service
youtube = build('youtube', 'v3', developerKey = key)

# -----------------------------------------------------------------

# Collected channelId's and made a dataframe
df = pd.read_csv('yt_id.csv')
df[' ID'].head()
channel_ids = df[' ID'].to_list()
channel_ids[:5]

'''
Output:
['UCppHT7SZKKvar4Oc9J4oljQ',
 'UCq-Fj5jknLsUf-MWSy4_brA',
 'UCbCmjCuTUZos6Inko4u57UQ',
 'UCpEhnqL0y41EpW2TvWAHD7Q',
 'UCJ5v_MCY6GNUBTO8-D3XoAg']
'''

# -----------------------------------------------------------------

# API Call - Data gather
def channel_status(youtube, channel_ids):

    request = youtube.channels().list(
        part = 'snippet,contentDetails,statistics',
        id = ','.join(channel_ids))
 
    response = request.execute()

    data = dict(channel_name = response['items'][0]['snippet']['title'],
                Subscribers = response['items'][0]['statistics']['subscriberCount'],
                Viewers = response['items'][0]['statistics']['viewCount'],
                Videos_made = response['items'][0]['statistics']['videoCount'],
                country = response['items'][0]['']['localized']['country'],
                channel_start = response['items'][0]['snippet']['publishedAt'], 
                channel_description = response['items'][0]['snippet']['description']
               )

    return response


# Function to get channel statistics
def channel_status(youtube, channel_id): #vid_id):

    request = youtube.channels().list(
    #request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        #id = channel_id,
        #forUsername="DaftPunkVEVO"
        id = vid_id,
    )

    response = request.execute()
    #print(response)
    return response

channel_status(youtube, channel_id)#vid_id)