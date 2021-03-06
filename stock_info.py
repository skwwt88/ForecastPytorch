import requests
import re
import pandas as pd

def stock_kline_day(id: str, enrich = None, max_count = 30000):
    data_uri = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={0}&scale=240&ma=5&datalen={1}".format(id, max_count)
    response = requests.get(data_uri)
    stock_df = pd.read_json(response.content)
    stock_df['day'] = stock_df['day'].apply(pd.to_datetime)
    stock_df = stock_df.set_index('day')
    stock_df.drop(['ma_price5', 'ma_volume5'], inplace=True, axis=1)

    return stock_df if not enrich else enrich(stock_df, id)

def qfq(stock_df, id: str):
    qfq_uri = "https://finance.sina.com.cn/realstock/company/{0}/qfq.js".format(id)
    response = requests.get(qfq_uri)
    qfq_content = re.search('\[.*?\]', response.content.decode(encoding='utf-8')).group()
    qfq_df = pd.read_json(qfq_content)
    
    last_date = '2100-1-1'
    for _, qfq_row in qfq_df.iterrows():
        start_date = qfq_row['d']
        f = qfq_row['f']

        stock_df.loc[start_date:last_date, 'open'] = stock_df.loc[start_date:last_date, 'open'] / f
        stock_df.loc[start_date:last_date, 'high'] = stock_df.loc[start_date:last_date, 'high'] / f
        stock_df.loc[start_date:last_date, 'low'] = stock_df.loc[start_date:last_date, 'low'] / f
        stock_df.loc[start_date:last_date, 'close'] = stock_df.loc[start_date:last_date, 'close'] / f
        stock_df.loc[start_date:last_date, 'volume'] = stock_df.loc[start_date:last_date, 'volume'] * f

        last_date = start_date

    return stock_df


def get_split_price():
    url = "http://market.finance.sina.com.cn/pricehis.php?symbol=sh600900&startdate=2011-08-17&enddate=2011-08-19"
    response = requests.get(url)
    print(response.content)

if __name__ == "__main__":
    print(stock_kline_day('sh600029'))