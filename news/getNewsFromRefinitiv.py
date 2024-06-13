import refinitiv.data as rd
from refinitiv.data.content import news
from IPython.display import HTML
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import time
import warnings
warnings.filterwarnings("ignore")
from refinitiv.data.content import symbol_conversion

def tickerToRic(ticker):
    df = symbol_conversion.Definition(
    symbols=[ticker],
    from_symbol_type=symbol_conversion.SymbolTypes.TICKER_SYMBOL,
    to_symbol_types=[
        symbol_conversion.SymbolTypes.RIC,
        symbol_conversion.SymbolTypes.OA_PERM_ID
    ],
).get_data().data.df

    return df["RIC"].iloc[0]

def ricToCompanyName(ric):
    title = symbol_conversion.Definition(symbols=ric).get_data().data.df["DocumentTitle"]
    companyName = title.split(',')[0].split(' ')[0]
    return companyName


def getNewsFromRefintiv(ticker):
    rd.open_session()
    startdate = dNow = datetime.now().date() - timedelta(weeks=24)
    ric = tickerToRic(ticker)
    print(ric)
    companyName = ricToCompanyName(ric)
    print(companyName)
    news = rd.news.get_headlines(f"R:{ric} AND Language:LEN AND Source:RTRS AND Topic:TOPALL", start= startdate, count = 1000).reset_index()    
    realNews = news[news['headline'].str.lower().str.contains(companyName,na=False)]
    return realNews