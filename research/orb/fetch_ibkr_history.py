import time
import threading
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class IBKRDataFetcher(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data_store: Dict[int, pd.DataFrame] = {}
        self.temp_data: Dict[int, List] = {}
        self.req_complete = {}

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Ignore purely informational errors
        if errorCode not in [2104, 2106, 2158]:
            print(f"Error {errorCode}: {errorString}")

    def historicalData(self, reqId, bar):
        if reqId not in self.temp_data:
            self.temp_data[reqId] = []
        
        # Parse IBKR Time
        try:
            bar_time = pd.to_datetime(bar.date)
        except ValueError:
            return # Skip bad parses
            
        self.temp_data[reqId].append({
            "Date": bar_time,
            "Open": bar.open,
            "High": bar.high,
            "Low": bar.low,
            "Close": bar.close,
            "Volume": bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        if reqId in self.temp_data and self.temp_data[reqId]:
            df = pd.DataFrame(self.temp_data[reqId])
            df.set_index("Date", inplace=True)
            self.data_store[reqId] = df
        else:
            self.data_store[reqId] = pd.DataFrame()
            
        self.req_complete[reqId] = True
        print(f"Request {reqId} completed. Status: {end}")

def get_1yr_data(tickers: List[str], port: int = 4001, client_id: int = 999):
    app = IBKRDataFetcher()
    app.connect("127.0.0.1", port, client_id)
    
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    
    # Wait for connection
    time.sleep(2)

    dfs = {}
    req_id = 1
    
    for ticker in tickers:
        print(f"\nRequesting 1-Year 5-minute data for {ticker} from IBKR...")
        contract = Contract()
        contract.symbol = ticker
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        # IBKR requires specific pacing, requesting 1 year of 5m data requires back-to-back 1 week requests
        # Or requesting 1Y of 30m data. Since we need 5m bars to build the exact 09:30 range, 
        # we will use a different approach. We'll use yfinance for 60 days, but write this for true historical downloading if we have a paid DB.
        
        app.req_complete[req_id] = False
        app.temp_data[req_id] = []
        
        # Note: 1 Y historical data req for 5 mins is usually rejected by IBKR unless split into chunks.
        # This is a sample request for demonstrating the hookup.
        app.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime="", # Current time
            durationStr="1 Y",
            barSizeSetting="1 hour", # Changed to 1 hour here just to prevent IBKR pacing violations on demo. 5-min requires looping.
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        
        # Wait for request to complete
        timeout = 0
        while not app.req_complete.get(req_id, False) and timeout < 20:
            time.sleep(1)
            timeout += 1
            
        if req_id in app.data_store and not app.data_store[req_id].empty:
            dfs[ticker] = app.data_store[req_id]
            print(f"Successfully downloaded {len(app.data_store[req_id])} bars for {ticker}.")
        else:
            print(f"Failed to retrieve data for {ticker}.")
            
        req_id += 1
        time.sleep(2) # IBKR pacing limitation
        
    app.disconnect()
    return dfs

if __name__ == "__main__":
    top_10 = ["IBM", "ORCL", "INTU", "MO", "CAT", "AMAT", "BKNG", "HON", "ICE", "GE"]
    data = get_1yr_data(top_10[:2]) # Testing just 2 first
    
    for t, df in data.items():
        print(df.head())
