import csv
import os

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import requests
from oauth2client.tools import argparser
from io import BytesIO
import time

# Import urlopen() for either Python 2 or 3.
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
import xlsxwriter

# Extract API key from environment variable 
YOUTUBE_API_KEY = "AIzaSyAsicJT2qZbFJzamuH2U_GF6h181uBt104"

# Create Youtube API client 
service = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def get_video_comments(service, **kwargs):
    comments = []
    results = service.commentThreads().list(**kwargs).execute()
 
    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
 
        # Check if another page exists
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = service.commentThreads().list(**kwargs).execute()
        else:
            break
 
    return comments

def write_to_csv(filename, comments):
    with open(f'{filename}.csv', 'w', encoding='UTF8') as comments_file:
        comments_writer = csv.writer(comments_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        comments_writer.writerow(['Video ID', 'Title', 'Comment'])
        for row in comments:
            # convert the tuple to a list and write to the output file
            comments_writer.writerow(list(row))


def get_videos(service, **kwargs):
    final_results = []
    results = service.search().list(**kwargs).execute()
 
    i = 0
    max_pages = 3
    while results and i < max_pages:
        final_results.extend(results['items'])
 
        # Check if another page exists
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = service.search().list(**kwargs).execute()
            i += 1
        else:
            break
 
    return final_results


def search_videos_by_keyword(service, filename, **kwargs):
    results = get_videos(service, **kwargs)
    final_result = []
    for item in results:
        title = item['snippet']['title']
        video_id = item['id']['videoId']
        comments = get_video_comments(service, part='snippet', videoId=video_id, textFormat='plainText')
        # make a tuple consisting of the video id, title, comment and add the result to 
        # the final list
        final_result.extend([(video_id, title, comment) for comment in comments]) 
    
    write_to_csv(filename, final_result)


if __name__ == '__main__':
    keyword_list = ['밴쯔', '한혜연', '송대익', '공혁준', '김민아', '보겸', '철구']

    # When running locally, disable OAuthlib's HTTPs verification. When
    # running in production *do not* leave this option enabled.
    for keyword in keyword_list: 
        search_videos_by_keyword(service, filename = keyword, q=keyword, part='id,snippet', 
                                type='video', order = 'viewCount', maxResults = 2, regionCode = 'KR', publishedAfter = '2020-07-01T00:00:00Z')