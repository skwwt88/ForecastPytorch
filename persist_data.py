import os
import pandas as pd
from tqdm import tqdm

from stock_info import stock_kline_day, qfq
from config import *

def persist_stock_info():
    pbar_stock_ids = tqdm(stock_ids)
    
    for stock_id in pbar_stock_ids:
        stock_df = stock_kline_day(stock_id, qfq)
        stock_df.to_csv(os.path.join(data_folder, "{0}.csv".format(stock_id)))

if __name__ == "__main__":
    persist_stock_info()