import yfinance as yf

# ソフトバンクグループの情報を取得（Tは東証を表す）
ticker_info = yf.Ticker("9984.T")

# 会社概要(info)を出力
print(ticker_info.info)

def get_company_name(security_code):
    # 証券コードに.Tを追加してティッカーシンボルを作成
    ticker_symbol = f"{security_code}.T"
    # yfinanceのTickerオブジェクトを作成
    ticker = yf.Ticker(ticker_symbol)
    # Tickerオブジェクトのinfo属性から会社名を取得
    company_name = ticker.info.get('shortName', 'Unknown')
    return company_name

# 例として証券コード7203（トヨタ自動車）の会社名を取得
security_code = "7203"
company_name = get_company_name(security_code)
print(f"証券コード {security_code} の会社名は {company_name} です。")
