
import finnhub
import datetime as dt 
import sys
import pandas as pd


import requests
from pprint import pprint

sys.path.insert(1, '/home/adam/Documents/perso/news-scoring-models/scoring')


from newsSentimentScoring import newsLineToScore

def getNewsFromTicker(companies,from_date,to_date,token):
    
    """
    From company equity ticker, get news headlines between 2 dates, using finnhub Oauth token
    """
    
    from_date = '2024-01-01'
    to_date = '2024-05-30'
    #token = 'c4e1jb2ad3ieqvqh3qbg'

    # Function to fetch and filter news for a given company
    def fetch_and_filter_news(symbol):
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': symbol,
            'from': from_date,
            'to': to_date,
            'token': token
        }

        # Make the request to the API
        response = requests.get(url, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            # Convert the response to a list of dictionaries
            news_data = response.json()
            # Filter the news data
            filtered_news = [news for news in news_data if symbol in news['summary']]
            news_dict = {
                item['headline']: dt.datetime.utcfromtimestamp(item['datetime'])
                for item in filtered_news
            }
            return news_dict

            
        else:
            print(f"Failed to fetch data for {symbol}: {response.status_code}")
            return []

    all_filtered_news = []
    for company in companies:
        all_filtered_news.append(fetch_and_filter_news(company))
    return all_filtered_news
        

