import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache

# 同じディレクトリのportfolio.csvの読み込み
df_pf = pd.read_csv('portfolio.csv', encoding='utf-8')
df_pf['Acquisition_Total'] = df_pf['Acquisition_Price'] * df_pf['Quantity']
security_code = ['Security_Code']

# 企業名の取得
@lru_cache(maxsize=100)
def get_company_info(security_code):
    ticker_symbol = f"{security_code}.T"
    ticker = yf.Ticker(ticker_symbol)
    company_name = ticker.info.get('shortName', 'Unknown')
    sector = ticker.info.get('sector', 'Unknown')
    return company_name, sector

# データフレームに企業名の追加
df_pf['Company_Name'], df_pf['Sector'] = zip(*df_pf['Security_Code'].apply(get_company_info))

# 過去1年分のデータ取得
def get_historical_prices(df):
    historical_data = {}
    for i in range(len(df)):
        s_code = str(int(df.iloc[i]['Security_Code'])) + ".T"
        ticker_info = yf.Ticker(s_code)
        history = ticker_info.history(period="1y")
        
        company_name = df.iloc[i]['Company_Name']
        history['Company_Name'] = company_name
        historical_data[company_name] = history[['Close', 'Company_Name']]
    return historical_data

historical_prices = get_historical_prices(df_pf)

df_history = pd.DataFrame()
for name, data in historical_prices.items():
    df_history = pd.concat([df_history, data])

# 価格の予測
def forecast_prices(df, periods=30):
    forecast_data = {}
    for i in range(len(df)):
        s_code = str(int(df.iloc[i]['Security_Code'])) + ".T"
        ticker_info = yf.Ticker(s_code)
        history = ticker_info.history(period="1y")['Close']
        rolling_mean = history.rolling(window=5).mean().dropna()
        last_price = history.iloc[-1]
        predictions = [last_price + (np.mean(rolling_mean[-5:]) - last_price) * (i/periods) for i in range(1, periods+1)]
        future_dates = [history.index[-1] + timedelta(days=i) for i in range(1, periods+1)]
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': predictions})
        forecast_df['Company_Name'] = df_pf['Company_Name']
        forecast_data[df.iloc[i]['Company_Name']] = forecast_df
    return forecast_data

forecast_data = forecast_prices(df_pf)

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='総資産推移', value='tab-1'),
        dcc.Tab(label='個別銘柄推移', value='tab-2'),
    ]),
    html.Div(id='tabs-content-example')
])

@app.callback(Output('tabs-content-example', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        # 総資産の計算
        df_pf['Current_Price'] = [data['Close'].iloc[-1] for data in historical_prices.values()]
        df_pf['Current_Total_Assets'] = df_pf['Current_Price'] * df_pf['Quantity']
        df_pf['Valuation_Gain_Loss'] = (df_pf['Current_Price'] - df_pf['Acquisition_Price']) * df_pf['Quantity']
        
        # 年間配当データの追加
        df_pf['Annual_Dividend_Price'] = [yf.Ticker(str(int(row['Security_Code'])) + ".T").info.get('dividendRate', 0) for index, row in df_pf.iterrows()]
        df_pf['Annual_Dividend'] = df_pf['Annual_Dividend_Price'] * df_pf['Quantity']
        
        # 次回配当データの追加（仮で1/2で計算）
        df_pf['Next_Dividend'] = df_pf['Annual_Dividend'] / 2
        
        # 1か月後の予想金額の取得
        df_pf['Forecast_1_Month'] = [int(forecast_data[row['Company_Name']]['Forecast'].iloc[-1] * row['Quantity']) for index, row in df_pf.iterrows()]
        
        # 配当利回りの追加
        df_pf['Dividend_Yield (%)'] = (df_pf['Annual_Dividend_Price'] / df_pf['Current_Price']) * 100
        
        # 金額を整数に変換
        df_pf['Current_Price'] = df_pf['Current_Price'].apply(lambda x: int(x))
        df_pf['Current_Total_Assets'] = df_pf['Current_Total_Assets'].apply(lambda x: int(x))
        df_pf['Annual_Dividend'] = df_pf['Annual_Dividend'].apply(lambda x: int(x))
        df_pf['Next_Dividend'] = df_pf['Next_Dividend'].apply(lambda x: int(x))
        df_pf['Dividend_Yield (%)'] = df_pf['Dividend_Yield (%)'].apply(lambda x: round(x, 2))
        
        # 数値型に変換
        df_pf['Annual_Dividend_Price'] = pd.to_numeric(df_pf['Annual_Dividend_Price'], errors='coerce')
        df_pf['Current_Price'] = pd.to_numeric(df_pf['Current_Price'], errors='coerce')

        # 配当利回りの再計算
        df_pf['Dividend_Yield (%)'] = (df_pf['Annual_Dividend_Price'] / df_pf['Current_Price']) * 100
      
        # CSSの作成
        cell_style = {
            'border': '1px solid black',
            'padding': '8px',
            'text-align': 'center'            
        }
                
        # テーブルの作成
        table_header = [
            html.Thead(html.Tr([html.Th("企業名", style=cell_style),
                                html.Th("取得単価", style=cell_style),
                                html.Th("取得数", style=cell_style),
                                html.Th("現在単価", style=cell_style),
                                html.Th("現在計", style=cell_style),
                                html.Th("評価損益", style=cell_style),
                                html.Th("1か月後予測", style=cell_style),
                                html.Th("年間配当額", style=cell_style),
                                html.Th("次回配当額（1/2）", style=cell_style),
                                html.Th("配当利回り (%)", style=cell_style)]))
        ]

        # 総計データの計算
        total_current_price = int(df_pf['Current_Total_Assets'].sum())
        total_forecast_price = int(df_pf['Forecast_1_Month'].sum())
        total_annual_dividend = int(df_pf['Annual_Dividend'].sum())
        total_next_dividend = int(df_pf['Next_Dividend'].sum())
        percentage_dividend = round(total_annual_dividend / total_current_price * 100, 2)
        total_valuation_gain_loss = int(df_pf['Valuation_Gain_Loss'].sum())

        # 整数をカンマ区切りにする
        df_pf['Current_Price'] = df_pf['Current_Price'].apply(lambda x: "{:,}".format(int(x)))
        df_pf['Current_Total_Assets'] = df_pf['Current_Total_Assets'].apply(lambda x: "{:,}".format(int(x)))
        df_pf['Valuation_Gain_Loss'] = df_pf['Valuation_Gain_Loss'].apply(lambda x: "{:,}".format(int(x)))
        df_pf['Forecast_1_Month'] = df_pf['Forecast_1_Month'].apply(lambda x: "{:,}".format(int(x)))
        df_pf['Annual_Dividend'] = df_pf['Annual_Dividend'].apply(lambda x: "{:,}".format(int(x)))
        df_pf['Next_Dividend'] = df_pf['Next_Dividend'].apply(lambda x: "{:,}".format(int(x)))
        df_pf['Dividend_Yield (%)'] = df_pf['Dividend_Yield (%)'].apply(lambda x: round(x, 2))
        
        # 総資産行の評価損益のスタイルを設定
        total_valuation_gain_loss_style = cell_style.copy()
        if total_valuation_gain_loss < 0:
            total_valuation_gain_loss_style['color'] = 'red'
        
        # 総資産行を1行目に追加
        rows = []
        rows.insert(0, html.Tr([html.Td("総資産", style=cell_style),
                                html.Td("", style=cell_style),
                                html.Td("", style=cell_style),
                                html.Td("", style=cell_style),
                                html.Td(f"{total_current_price:,} ", style=cell_style),
                                html.Td(f"{total_valuation_gain_loss:,} ",style=total_valuation_gain_loss_style ),
                                html.Td(f"{total_forecast_price:,} ", style=cell_style),
                                html.Td(f"{total_annual_dividend:,} ", style=cell_style),
                                html.Td(f"{total_next_dividend:,} ", style=cell_style),
                                html.Td(f"{percentage_dividend}%", style=cell_style)]))

        # 総資産行の下に保有証券情報を追加
        for index, row in df_pf.iterrows():
            stock_name, sector_name = get_company_info(row['Security_Code'])
            valuation_gain_loss_style = cell_style.copy()
            if int(row['Valuation_Gain_Loss'].replace(',', '')) < 0:
                valuation_gain_loss_style['color'] = 'red'
            rows.append(html.Tr([html.Td([html.Span(f"証券コード: {row['Security_Code']}"), html.Br(), html.Span(stock_name), html.Br(), html.Span(sector_name)], style=cell_style),
                                 html.Td(row['Acquisition_Price'], style=cell_style),
                                 html.Td(row['Quantity'], style=cell_style), 
                                 html.Td(f"{row['Current_Price']} ", style=cell_style),
                                 html.Td(f"{row['Current_Total_Assets']} ", style=cell_style),
                                 html.Td(f"{row['Valuation_Gain_Loss']} ", style=valuation_gain_loss_style),
                                 html.Td(f"{row['Forecast_1_Month']} ", style=cell_style),
                                 html.Td(f"{row['Annual_Dividend']} ", style=cell_style),
                                 html.Td(f"{row['Next_Dividend']} ", style=cell_style),
                                 html.Td(f"{row['Dividend_Yield (%)']}%")], style=cell_style))

        # テーブルの作成
        table_body = [html.Tbody(rows)]
        table = html.Table(table_header + table_body, style={'border': '1px solid black', 'border-collapse': 'collapse', 'width': '100%', 'text-align': 'center'})

        
        def calculate_total_asset_over_time(df, historical_data):
            # 日ごとの総資産を追加するインデックスの作成
            total_asset_per_day = {}

            # 各企業の合計を計算
            for i in range(len(df)):
                company_name = df.iloc[i]['Company_Name']
                acquisition_amount = df.iloc[i]['Quantity']
                close_prices = historical_data[company_name]['Close']  # 過去1年分のデータから計算

                # 各日付の総資産を計算
                for date, price in close_prices.items():
                    if date not in total_asset_per_day:
                        total_asset_per_day[date] = 0
                    total_asset_per_day[date] += price * acquisition_amount  # 株価*取得数でその日の資産を加算

            # 日付と総資産をデータフレームに変換
            total_asset_df = pd.DataFrame(list(total_asset_per_day.items()), columns=['Date', 'TotalAsset'])
            total_asset_df = total_asset_df.sort_values('Date')
            
            return total_asset_df

        # 総資産の時系列データの取得
        total_asset_df = calculate_total_asset_over_time(df_pf, historical_prices)

        # 総資産グラフの作成
        fig_portfolio = px.line(total_asset_df, x='Date', y='TotalAsset', title='Total Asset Trend of Portfolio')

        # グラフとテーブルを返す
        return html.Div([
            dcc.Graph(figure=fig_portfolio),
            table
        ])

    elif tab == 'tab-2':
        sorted_df = df_pf.sort_values(by='Security_Code')
        graphs = []
        for name in sorted_df['Company_Name']:
            individual_df = df_history[df_history['Company_Name'] == name]
            forecast_df = forecast_data[name]
            y_min = min(individual_df['Close'].min(), forecast_df['Forecast'].min()) * 0.95
            y_max = max(individual_df['Close'].max(), forecast_df['Forecast'].max()) * 1.05
            fig = px.line(individual_df, x=individual_df.index, y='Close', title=f'{name} Trend')
            fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast')
            fig.update_layout(yaxis=dict(range=[y_min, y_max]))
            graphs.append(dcc.Graph(figure=fig))
        return html.Div(graphs)

if __name__ == '__main__':
    app.run_server(debug=True)
