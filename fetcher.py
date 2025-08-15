import ccxt

def fetch_current_price(symbol):
    exchange = ccxt.binance()
    ticker = exchange.fetch_ticker(symbol)
    return ticker['last']
