import requests
import re
import pandas as pd

def day_k_data(id: str, enrich = None):
    data_uri = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={0}&scale=240&ma=5&datalen=5000".format(id)
    response = requests.get(data_uri)
    stock_df = pd.read_json(response.content)
    stock_df['day'] = stock_df['day'].apply(pd.to_datetime)
    stock_df = stock_df.set_index('day')

    return stock_df if not enrich else enrich(stock_df, id)

def qfq(stock_df, id: str):
    qfq_uri = "https://finance.sina.com.cn/realstock/company/{0}/qfq.js".format(id)
    response = requests.get(qfq_uri)
    qfq_content = re.search('\[.*?\]', response.content.decode(encoding='utf-8')).group()
    qfq_df = pd.read_json(qfq_content)
    
    last_date = '2050-1-1'
    for _, qfq_row in qfq_df.iterrows():
        start_date = qfq_row['d']
        f = qfq_row['f']
        stock_df.loc[start_date:last_date, 'open'] = stock_df.loc[start_date:last_date, 'open'] / f
        stock_df.loc[start_date:last_date, 'high'] = stock_df.loc[start_date:last_date, 'high'] / f
        stock_df.loc[start_date:last_date, 'low'] = stock_df.loc[start_date:last_date, 'low'] / f
        stock_df.loc[start_date:last_date, 'close'] = stock_df.loc[start_date:last_date, 'close'] / f
        stock_df.loc[start_date:last_date, 'ma_price5'] = stock_df.loc[start_date:last_date, 'ma_price5'] / f
        stock_df.loc[start_date:last_date, 'volume'] = stock_df.loc[start_date:last_date, 'volume'] * f
        stock_df.loc[start_date:last_date, 'ma_volume5'] = stock_df.loc[start_date:last_date, 'ma_volume5'] * f

        last_date = start_date

    return stock_df


if __name__ == "__main__":
    print(day_k_data('sh600029'))
    print(day_k_data('sh600029', qfq))