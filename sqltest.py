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

# データベース設定
db_file = 'portfolio.db'
engine = create_engine(f'sqlite:///{db_file}')
metadata = MetaData()

# ポートフォリオテーブルの定義
portfolio_table = Table(
    'portfolio', metadata,
    Column('id', Integer, primary_key=True),
    Column('証券コード', String, nullable=False),
    Column('取得単価', Float, nullable=False),
    Column('取得数', Integer, nullable=False)
)

# テーブル作成
metadata.create_all(engine)

# セッションを作成
Session = sessionmaker(bind=engine)
session = Session()

# ポートフォリオデータの取得
def get_portfolio_data():
    df_pf = pd.read_sql_table('portfolio', engine)
    return df_pf if not df_pf.empty else pd.DataFrame(columns=['証券コード', '取得単価', '取得数'])

# データフレームに変換
df_pf = get_portfolio_data()

# データが存在しない場合は空のデータフレームを使う
if df_pf.empty:
    df_pf = pd.DataFrame(columns=['証券コード', '取得単価', '取得数'])

# 取得総額列を追加
if not df_pf.empty:
    df_pf['取得総額(円)'] = df_pf['取得単価'] * df_pf['取得数']

# 企業名の取得
def get_company_info(security_code):
    ticker_symbol = f"{security_code}.T"
    ticker = yf.Ticker(ticker_symbol)
    company_name = ticker.info.get('shortName', 'Unknown')
    sector = ticker.info.get('sector', 'Unknown')
    return company_name, sector

# 企業名を取得してデータフレームに追加
if not df_pf.empty:
    df_pf['企業名'], df_pf['セクター'] = zip(*df_pf['証券コード'].apply(get_company_info))

# 過去1年分のデータ取得
def get_historical_prices(df):
    historical_data = {}
    if df.empty:
        return historical_data
    for i in range(len(df)):
        s_code = str(int(df.iloc[i]['証券コード'])) + ".T"
        ticker_info = yf.Ticker(s_code)
        history = ticker_info.history(period="1y")
        company_name = df.iloc[i]['企業名']
        history['企業名'] = company_name
        historical_data[company_name] = history[['Close', '企業名']]
    return historical_data

historical_prices = get_historical_prices(df_pf)

# 過去の株価データを結合
df_history = pd.DataFrame()
for name, data in historical_prices.items():
    df_history = pd.concat([df_history, data])

# ダッシュアプリのセットアップ
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='総資産', value='tab-1'),
        dcc.Tab(label='銘柄別', value='tab-2'),
    ]),
    html.Div(id='tabs-content-example'),
    # データ追加用フォーム
    html.Div([
        html.H3("データ追加"),
        html.Label("証券コード"),
        dcc.Input(id='new_code', type='text'),
        html.Label("取得単価"),
        dcc.Input(id='new_price', type='number'),
        html.Label("取得数"),
        dcc.Input(id='new_amount', type='number'),
        html.Button('追加', id='add-button', n_clicks=0),
    ], style={'padding': '20px'}),
    html.Div(id='add-output'),
])

# データの追加
@app.callback(
    Output('add-output', 'children'),
    Input('add-button', 'n_clicks'),
    [Input('new_code', 'value'),
     Input('new_price', 'value'),
     Input('new_amount', 'value')]
)
def add_data(n_clicks, new_code, new_price, new_amount):
    if n_clicks > 0 and new_code and new_price and new_amount:
        # データベースに新しい銘柄を追加
        new_entry = {'証券コード': new_code, '取得単価': new_price, '取得数': new_amount}
        df_new = pd.DataFrame([new_entry])
        df_new.to_sql('portfolio', engine, if_exists='append', index=False)
        return f"新しい銘柄: {new_code} が追加されました。"
    return ""

# コールバックでコンテンツをレンダリング
@app.callback(Output('tabs-content-example', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    df_pf = get_portfolio_data()
    if df_pf.empty:
        return html.Div("ポートフォリオにデータがありません。銘柄を追加してください。")

    if tab == 'tab-1':
        # 総資産のデータを計算
        df_pf['現単価(円)'] = [data['Close'].iloc[-1] for data in historical_prices.values()]
        df_pf['現総資産(円)'] = df_pf['現単価(円)'] * df_pf['取得数']
        df_pf['評価損益(円)'] = (df_pf['現単価(円)'] - df_pf['取得単価'])*df_pf['取得数']
        
        # 年間配当金データを追加
        df_pf['年間配当単価(円)'] = [yf.Ticker(str(int(row['証券コード'])) + ".T").info.get('dividendRate', 0) for index, row in df_pf.iterrows()]
        df_pf['年間配当金(円)'] = df_pf['年間配当単価(円)'] * df_pf['取得数']
        
        # グラフの作成（ポートフォリオの総資産推移）
        def calculate_total_asset_over_time(df, historical_data):
            total_asset_per_day = {}
            for i in range(len(df)):
                company_name = df.iloc[i]['企業名']
                acquisition_amount = df.iloc[i]['取得数']
                close_prices = historical_data[company_name]['Close']
                for date, price in close_prices.items():
                    if date not in total_asset_per_day:
                        total_asset_per_day[date] = 0
                    total_asset_per_day[date] += price * acquisition_amount
            total_asset_df = pd.DataFrame(list(total_asset_per_day.items()), columns=['Date', 'TotalAsset'])
            total_asset_df = total_asset_df.sort_values('Date')
            return total_asset_df
        
        total_asset_df = calculate_total_asset_over_time(df_pf, historical_prices)
        fig_portfolio = px.line(total_asset_df, x='Date', y='TotalAsset', title='ポートフォリオの総資産推移')

        return html.Div([
            dcc.Graph(figure=fig_portfolio)
        ])

    elif tab == 'tab-2':
        sorted_df = df_pf.sort_values(by='証券コード')
        graphs = []
        for name in sorted_df['企業名']:
            individual_df = df_history[df_history['企業名'] == name]
            forecast_df = forecast_data[name]
            y_min = min(individual_df['Close'].min(), forecast_df['Forecast'].min()) * 0.95
            y_max = max(individual_df['Close'].max(), forecast_df['Forecast'].max()) * 1.05
            fig = px.line(individual_df, x=individual_df.index, y='Close', title=f'{name}の推移')
            fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='予測')
            fig.update_layout(yaxis=dict(range=[y_min, y_max]))
            graphs.append(dcc.Graph(figure=fig))
        return html.Div(graphs)

if __name__ == '__main__':
    app.run_server(debug=True)
