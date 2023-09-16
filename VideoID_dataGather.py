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

# Importing video_ids
df = pd.read_csv('Video_id.csv', index_col=0)
df.head(1)


# Create an empty list to store channel data
video_data_list = []

# Define the channel_status function
def video_status(youtube, video_id):
    try:
        request = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            id=video_id)

        response = request.execute()

        # Check if 'items' key exists in the response
        if 'items' in response and len(response['items']) > 0:
            response_data = response['items'][0]
            
            data = {
                #'channel_id': response_data['snippet']['channelId'],
                'channel_name': response_data['snippet']['channelTitle'],
                'title': response_data['snippet']['title'],
                #'Subscribers': response_data['statistics']['subscriberCount'],
                'Viewers': response_data['statistics']['viewCount'],
                #'Videos_made': response_data['statistics']['videoCount'],
                'channelTitle': response_data['snippet']['channelTitle'],
                'tags': response_data['snippet']['tags'],
                'categoryId': response_data['snippet']['categoryId'],
                #'liveBroadcastContent': response_data['snippet']['liveBroadcastContent'],
                #'defaultLanguage': response_data['snippet']['defaultLanguage'],
                #'defaultAudioLanguage': response_data['snippet']['defaultAudioLanguage'],
                #'HiddenSubCount': response_data['statistics']['hiddenSubscriberCount'],
                'video_published_date': response_data['snippet']['publishedAt'],
                'kind': response_data['kind'],
                'description': response_data['snippet']['description'],
                'duration': response_data['contentDetails']['duration'],
                'dimension': response_data['contentDetails']['dimension'],
                'definition': response_data['contentDetails']['definition'],
                #'caption': response_data['contentDetails']['caption'],
                'viewCount': response_data['statistics']['viewCount'],
                'likeCount': response_data['statistics']['likeCount'],
                'commentCount': response_data['statistics']['commentCount']
            }
            return data
        else:
            # If 'items' key is not found or is empty, return None
            return None
    except Exception as e:
        # Handle any exceptions that may occur during the API request
        print(f"Error fetching data for channel {video_id}: {str(e)}")
        return None
    

# Iterate through the channel IDs and retrieve statistics
for video_id in df['video_id']:
    video_data = video_status(youtube, video_id)
    if video_data is not None:
        video_data_list.append(video_data)

# Create a DataFrame from the list of channel data
video_data_df = pd.DataFrame(video_data_list)


# Saving the dataframe
video_data_df.to_csv('Youtube_videoData.csv')