#-------------------------------------------------------------------------------
# Name:         dataLoader.py
# Purpose:      Loads the daily data of select stocks from yfinance API.
#
# Author:       Sai Tarun Ganapavarapu
#
# Created:      02-14-2026
# Licence:      MIT License
#-------------------------------------------------------------------------------
import yfinance as yf
import pandas as pd

class DataLoader:
    """
    A class to handle data retrieval and initial processing.
    """
    @staticmethod
    def getData(tickers, startDate, endDate):
        """
        Downloads historical stock data for the given tickers.
        Returns a pandas DataFrame of adjusted close prices (auto_adjust=True will adjust for splits/dividends).
        """
        try:
            # Download Data 
            data = yf.download(tickers, start = startDate, end = endDate, auto_adjust = True)

            if data.empty:
                print("Error: Yfinance download returned empty data set.")
                return None

            if isinstance(data.columns, pd.MultiIndex):
                # Multiple tickers - extract 'Close' slice
                if "Close" in data.columns.get_level_values(0):
                    closeData = data.xs("Close", axis = 1, level = 0)
                    closeData.columns.name = None
                else:
                    print("Error: MultiIndex found but 'Close' level is missing.")
                    return None
            else:
                # Single ticker - extract 'Close' column
                if "Close" in data.columns:
                    closeData = data[["Close"]]
                    if isinstance(tickers, str):
                        closeData.columns = [tickers]
                    elif isinstance(tickers, list) and len(tickers) == 1:
                        closeData.columns = tickers
                else:
                    print("Error: Data downloaded but 'Close' column not found.")
                    return None

            # Drop columns with all NaNs
            closeData = closeData.dropna(axis=1, how="all")

            if closeData.empty:
                print("Error: No valid Close data could be found for the specified tickers.")
                return None

            return closeData.sort_index()

        except Exception as e:
            print(f"Error in DataLoader.getData: {e}")
            return None
