import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, MetaData, Table
from sqlalchemy.orm import sessionmaker
from functools import lru_cache

# データ読み込み 
df_pf = pd.read_csv('portfolio.csv', encoding='utf-8') 
df_pf['取得総額(円)'] = df_pf['取得単価(円)'] * df_pf['取得数'] 
security_code = ['証券コード'] 
 
# 企業名の取得 
@lru_cache(maxsize=100) 
def get_company_info(security_code): 
    ticker_symbol = f"{security_code}.T" 
    ticker = yf.Ticker(ticker_symbol) 
    company_name = ticker.info.get('shortName', 'Unknown') 
    sector = ticker.info.get('sector','Unknown') 
    return company_name,sector 
 
#企業名を取得してデータフレームに追加 
df_pf['企業名'], df_pf['セクター'] = zip(*df_pf['証券コード'].apply(get_company_info)) 
 
# 過去1年分のデータ取得 
def get_historical_prices(df): 
    historical_data = {} 
    for i in range(len(df)): 
        s_code = str(int(df.iloc[i]['証券コード'])) + ".T" 
        ticker_info = yf.Ticker(s_code) 
        history = ticker_info.history(period="1y") 
         
        company_name =df.iloc[i]['企業名'] 
        history['企業名'] = company_name 
        historical_data[company_name] = history[['Close', '企業名']] 
    return historical_data 
 
historical_prices = get_historical_prices(df_pf) 
 
df_history = pd.DataFrame() 
for name, data in historical_prices.items(): 
    df_history = pd.concat([df_history, data]) 
 
def forecast_prices(df, periods=30): 
    forecast_data = {} 
    for i in range(len(df)): 
        s_code = str(int(df.iloc[i]['証券コード'])) + ".T" 
        ticker_info = yf.Ticker(s_code) 
        history = ticker_info.history(period="1y")['Close'] 
        rolling_mean = history.rolling(window=5).mean().dropna() 
        last_price = history.iloc[-1] 
        predictions = [last_price + (np.mean(rolling_mean[-5:]) - last_price) * (i/periods) for i in range(1, periods+1)] 
        future_dates = [history.index[-1] + timedelta(days=i) for i in range(1, periods+1)] 
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': predictions}) 
        forecast_df['企業名'] = df_pf['企業名'] 
        forecast_data[df.iloc[i]['企業名']] = forecast_df 
    return forecast_data 
 
forecast_data = forecast_prices(df_pf) 
 
app = dash.Dash(__name__, suppress_callback_exceptions=True) 
 
app.layout = html.Div([ 
    dcc.Tabs(id='tabs-example', value='tab-1', children=[ 
        dcc.Tab(label='総資産', value='tab-1'), 
        dcc.Tab(label='銘柄別', value='tab-2'), 
    ]), 
    html.Div(id='tabs-content-example') 
]) 
 
@app.callback(Output('tabs-content-example', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        # 総資産のデータを計算
        df_pf['現単価(円)'] = [data['Close'].iloc[-1] for data in historical_prices.values()]
        df_pf['現総資産(円)'] = df_pf['現単価(円)'] * df_pf['取得数']
        df_pf['評価損益(円)'] = (df_pf['現単価(円)'] - df_pf['取得単価(円)'])*df_pf['取得数']
        
        # 年間配当金データを追加
        df_pf['年間配当単価(円)'] = [yf.Ticker(str(int(row['証券コード'])) + ".T").info.get('dividendRate', 0) for index, row in df_pf.iterrows()]
        df_pf['年間配当金(円)'] = df_pf['年間配当単価(円)'] * df_pf['取得数']
        
        # 次回配当金データを追加（仮のデータとして年間配当金の1/2を使用）
        df_pf['次回配当金(円)'] = df_pf['年間配当金(円)'] / 2
        
        # 1か月後の予想金額を取得（予想金額 * 取得数）
        df_pf['1か月後の予想金額(円)'] = [int(forecast_data[row['企業名']]['Forecast'].iloc[-1] * row['取得数']) for index, row in df_pf.iterrows()]
        
        # 配当利回りを追加
        df_pf['配当利回り (%)'] = (df_pf['年間配当単価(円)'] / df_pf['現単価(円)']) * 100
        
        # 金額を小数点第1位で切り捨て
        df_pf['現単価(円)'] = df_pf['現単価(円)'].apply(lambda x:int(x))
        df_pf['現総資産(円)'] = df_pf['現総資産(円)'].apply(lambda x:int(x))
        df_pf['年間配当金(円)'] = df_pf['年間配当金(円)'].apply(lambda x:int(x))
        df_pf['次回配当金(円)'] = df_pf['次回配当金(円)'].apply(lambda x:int(x))
        df_pf['配当利回り (%)'] = df_pf['配当利回り (%)'].apply(lambda x: round(x, 2))
        
        # '年間配当単価(円)'と'現単価(円)'を数値型に変換
        df_pf['年間配当単価(円)'] = pd.to_numeric(df_pf['年間配当単価(円)'], errors='coerce')
        df_pf['現単価(円)'] = pd.to_numeric(df_pf['現単価(円)'], errors='coerce')

        # 配当利回り (%) を再計算
        df_pf['配当利回り (%)'] = (df_pf['年間配当単価(円)'] / df_pf['現単価(円)']) * 100
      
        #CSSスタイル追加
        cell_style = {
            'border': '1px solid black',
            'padding': '8px',
            'text-align': 'center'            
        }
        
        # テーブルの作成
        table_header = [
            html.Thead(html.Tr([html.Th("銘柄名", style=cell_style),
                                html.Th("取得単価", style=cell_style),
                                html.Th("取得数", style=cell_style),
                                html.Th("現単価", style=cell_style),
                                html.Th("現総資産", style=cell_style),
                                html.Th("評価損益", style=cell_style),
                                html.Th("1ヶ月後予想価格", style=cell_style),
                                html.Th("年間配当金額", style=cell_style),
                                html.Th("次回配当金額", style=cell_style),
                                html.Th("配当利回り (%)", style=cell_style)]))
        ]

        # 総計データを計算
        total_current_price = int(df_pf['現総資産(円)'].sum())
        total_forecast_price = int(df_pf['1か月後の予想金額(円)'].sum())
        total_annual_dividend = int(df_pf['年間配当金(円)'].sum())
        total_next_dividend = int(df_pf['次回配当金(円)'].sum())
        persentage_dividend = round(total_annual_dividend / total_current_price * 100, 2)
        total_valuation_gain_loss = int(df_pf['評価損益(円)'].sum())

        # 整数をカンマ区切りにする        
        df_pf['現単価(円)'] = df_pf['現単価(円)'].apply(lambda x: "{:,}".format(int(x)))
        df_pf['現総資産(円)'] = df_pf['現総資産(円)'].apply(lambda x: "{:,}".format(int(x)))
        df_pf['評価損益(円)'] = df_pf['評価損益(円)'].apply(lambda x: "{:,}".format(int(x)))
        df_pf['1か月後の予想金額(円)'] = df_pf['1か月後の予想金額(円)'].apply(lambda x: "{:,}".format(int(x)))
        df_pf['年間配当金(円)'] = df_pf['年間配当金(円)'].apply(lambda x: "{:,}".format(int(x)))
        df_pf['次回配当金(円)'] = df_pf['次回配当金(円)'].apply(lambda x: "{:,}".format(int(x)))
        df_pf['配当利回り (%)'] = df_pf['配当利回り (%)'].apply(lambda x: round(x, 2))
        
        # 総資産行を追加
        rows = []
        rows.insert(0, html.Tr([html.Td("総資産", style=cell_style),
                                html.Td("", style=cell_style),
                                html.Td("", style=cell_style),
                                html.Td("", style=cell_style),
                                html.Td(f"{total_current_price:,}円", style=cell_style),
                                html.Td(f"{total_valuation_gain_loss:,}円", style=cell_style),
                                html.Td(f"{total_forecast_price:,}円", style=cell_style),
                                html.Td(f"{total_annual_dividend:,}円", style=cell_style),
                                html.Td(f"{total_next_dividend:,}円", style=cell_style),
                                html.Td(f"{persentage_dividend}%")], style=cell_style))

        # 各銘柄の行を追加
        for index, row in df_pf.iterrows():
            stock_name, sector_name = get_company_info(row['証券コード'])
            rows.append(html.Tr([html.Td([html.Span(f"証券コード: {row['証券コード']}"), html.Br(), html.Span(stock_name), html.Br(), html.Span(sector_name)], style=cell_style),
                                 html.Td(row['取得単価(円)'], style=cell_style),
                                 html.Td(row['取得数'], style=cell_style), 
                                 html.Td(f"{row['現単価(円)']}円", style=cell_style),
                                 html.Td(f"{row['現総資産(円)']}円", style=cell_style),
                                 html.Td(f"{row['評価損益(円)']}円", style=cell_style),
                                 html.Td(f"{row['1か月後の予想金額(円)']}円", style=cell_style),
                                 html.Td(f"{row['年間配当金(円)']}円", style=cell_style),
                                 html.Td(f"{row['次回配当金(円)']}円", style=cell_style),
                                 html.Td(f"{row['配当利回り (%)']}%")], style=cell_style))

        # テーブル作成
        table_body = [html.Tbody(rows)]
        table = html.Table(table_header + table_body, style={'border': '1px solid black', 'border-collapse': 'collapse', 'width': '100%', 'text-align': 'center'})

        # 総資産のグラフを作成
        def calculate_total_asset_over_time(df, historical_data):
            # 日付ごとの総資産を記録する辞書
            total_asset_per_day = {}

            # 各企業について総資産を計算
            for i in range(len(df)):
                company_name = df.iloc[i]['企業名']
                acquisition_amount = df.iloc[i]['取得数']
                close_prices = historical_data[company_name]['Close']  # 過去1年の株価

                # 各日付ごとの総資産を計算
                for date, price in close_prices.items():
                    if date not in total_asset_per_day:
                        total_asset_per_day[date] = 0
                    total_asset_per_day[date] += price * acquisition_amount  # 株価 * 取得数 でその日の資産を加算

            # 日付と総資産をデータフレームに変換
            total_asset_df = pd.DataFrame(list(total_asset_per_day.items()), columns=['Date', 'TotalAsset'])
            total_asset_df = total_asset_df.sort_values('Date')
            
            return total_asset_df

        # 総資産の時系列データを取得
        total_asset_df = calculate_total_asset_over_time(df_pf, historical_prices)

        # 総資産グラフを作成
        fig_portfolio = px.line(total_asset_df, x='Date', y='TotalAsset', title='ポートフォリオの総資産推移')

        # グラフとテーブルを返す
        return html.Div([
            dcc.Graph(figure=fig_portfolio),
            table
        ])

 
    elif tab == 'tab-2': 
        sorted_df = df_pf.sort_values(by='証券コード') 
        graphs = [] 
        for name in sorted_df['企業名']: 
            individual_df = df_history[df_history['企業名'] == name] 
            forecast_df = forecast_data[name] 
            y_min = min(individual_df['Close'].min(), forecast_df['Forecast'].min()) * 0.95 
            y_max = max(individual_df['Close'].max(), forecast_df['Forecast'].max()) * 1.05 
            fig = px.line(individual_df, x=individual_df.index, y='Close', title=f'{name}の過去1年の推移') 
            fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='予測') 
            fig.update_layout(yaxis=dict(range=[y_min, y_max])) 
            graphs.append(dcc.Graph(figure=fig)) 
        return html.Div(graphs) 
 
if __name__ == '__main__': 
    app.run_server(debug=True)