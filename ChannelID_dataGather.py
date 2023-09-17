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
# Create an empty list to store channel data
channel_data_list = []

# Define the channel_status function
def channel_status(youtube, channel_id):
    try:
        request = youtube.channels().list(
            part='snippet,contentDetails,statistics',
            id=channel_id)

        response = request.execute()

        # Check if 'items' key exists in the response
        if 'items' in response and len(response['items']) > 0:
            data = {
                'channel_id': channel_id,
                'channel_name': response['items'][0]['snippet']['title'],
                'Subscribers': response['items'][0]['statistics']['subscriberCount'],
                'Viewers': response['items'][0]['statistics']['viewCount'],
                'Videos_made': response['items'][0]['statistics']['videoCount'],
                 #'country' : response['items'][0]['localized']['country'],
                'HiddenSubCount' : response['items'][0]['statistics']['hiddenSubscriberCount'],
                  'channel_start' : response['items'][0]['snippet']['publishedAt'],
            'kind' :   response['items'][0]['kind'],
                'channel_description' : response['items'][0]['snippet']['description']
            }
            return data
        else:
            # If 'items' key is not found or is empty, return None
            return None
    except Exception as e:
        # Handle any exceptions that may occur during the API request
        print(f"Error fetching data for channel {channel_id}: {str(e)}")
        return None



# Function to get channel statistics
# Iterate through the channel IDs and retrieve statistics
for channel_id in df_id[' ID']:
    channel_data = channel_status(youtube, channel_id)
    if channel_data is not None:
        channel_data_list.append(channel_data)

# Create a DataFrame from the list of channel data
channel_data_df = pd.DataFrame(channel_data_list)

channel_data_df.to_csv('Youtube_channel_data.csv')