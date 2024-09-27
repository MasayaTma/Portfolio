import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
from sqlalchemy import create_engine, Column, Integer, String, Float, MetaData, Table

def create_app():
    # SQLiteデータベースの設定
    db_file = 'portfolio.db'
    engine = create_engine(f'sqlite:///{db_file}')
    metadata = MetaData()

    # ポートフォリオテーブルの作成
    portfolio_table = Table(
        'portfolio', metadata,
        Column('id', Integer, primary_key=True),
        Column('証券コード', String, nullable=False),
        Column('取得単価', Float, nullable=False),
        Column('取得数', Integer, nullable=False)
    )
    metadata.create_all(engine)

    # データ取得関数
    def get_portfolio_data():
        df_pf = pd.read_sql_table('portfolio', engine)
        return df_pf if not df_pf.empty else pd.DataFrame(columns=['証券コード', '取得単価', '取得数'])

    # 企業情報取得関数
    def get_company_info(security_code):
        ticker_symbol = f"{security_code}.T"
        ticker = yf.Ticker(ticker_symbol)
        company_name = ticker.info.get('shortName', 'Unknown')
        sector = ticker.info.get('sector', 'Unknown')
        return company_name, sector

    # 過去1年分のデータ取得関数
    def get_historical_prices(df):
        historical_data = {}
        if df.empty:
            return historical_data
        for i in range(len(df)):
            try:
                s_code = str(int(df.iloc[i]['証券コード'])) + ".T"
                ticker_info = yf.Ticker(s_code)
                history = ticker_info.history(period="1y")
                
                if history.empty:
                    print(f"データが取得できませんでした: {s_code}")
                    continue
                
                company_name = df.iloc[i]['証券コード']
                historical_data[company_name] = history[['Close']]
            except Exception as e:
                print(f"エラーが発生しました: {s_code}, エラー: {e}")
                continue
        return historical_data

    # Dashアプリの設定
    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    app.layout = html.Div([
        dcc.Tabs(id='tabs-example', value='tab-1', children=[
            dcc.Tab(label='総資産', value='tab-1'),
            dcc.Tab(label='銘柄別', value='tab-2'),
            dcc.Tab(label='銘柄登録', value='tab-3')
        ]),
        html.Div(id='tabs-content-example'),
        html.Div(id='output-state')  # output-state to display success messages
    ])

    # コンテンツをレンダリング
    @app.callback(
        [Output('tabs-content-example', 'children'),
        Output('output-state', 'children')],
        [Input('tabs-example', 'value')],
        [State('submit-button', 'n_clicks'),
        State('edit-button', 'n_clicks'),
        State('delete-button', 'n_clicks'),
        State('input-id', 'value'),
        State('input-code', 'value'),
        State('input-price', 'value'),
        State('input-quantity', 'value')],
        prevent_initial_call=True
    )
    
    def render_tab_content(tab, submit_clicks, edit_clicks, delete_clicks, id, code, price, quantity):
        ctx = dash.callback_context
        if not ctx.triggered:
            return html.Div(), ""
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        #データ追加の処理
        if button_id == 'submit-button' and submit_clicks:
            if code and price is not None and quantity is not None:
                with engine.connect() as conn:
                    conn.execute(portfolio_table.insert().values(
                        証券コード=code,
                        取得単価=price,
                        取得数=quantity
                    ))
                return render_add_portfolio(), "データが追加されました"
        
        #データ更新の処理
        if button_id == 'edit-button' and edit_clicks:
            if id and code and price is not None and quantity is not None:
                with engine.connect() as conn:
                    conn.execute(portfolio_table.update().where(portfolio_table.c.id==id).values(
                        証券コード = code,
                        取得単価 = price,
                        取得数 = quantity
                    ))
                return render_add_portfolio(),"データが更新されました"

        #データ削除の処理
        if button_id == 'delete-button' and delete_clicks:
            if id and code and price is not None and quantity is not None:
                with engine.connect() as conn:
                    conn.execute(portfolio_table.delete().where(portfolio_table.c.id==id))
                return render_add_portfolio(),"データが削除されました"
        
        df_pf = get_portfolio_data()
        historical_prices = get_historical_prices(df_pf)
        
        if tab == 'tab-1':
            if df_pf.empty:
                return html.Div("ポートフォリオにデータがありません。銘柄を追加してください。")

            if historical_prices:
                df_pf['現単価(円)'] = [data['Close'].iloc[-1] for data in historical_prices.values()]
            else:
                df_pf['現単価(円)'] = np.nan
            
            df_pf['現総資産(円)'] = df_pf['現単価(円)'] * df_pf['取得数']
            df_pf['評価損益(円)'] = (df_pf['現単価(円)'] - df_pf['取得単価']) * df_pf['取得数']
            
            # 総資産グラフの作成
            def calculate_total_asset_over_time(df, historical_data):
                total_asset_per_day = {}
                for i in range(len(df)):
                    company_name = df.iloc[i]['証券コード']
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

            # テーブル表示
            portfolio_table = df_pf.to_dict('records')
            table_rows = [html.Tr([html.Td(row['証券コード']),
                                   html.Td(row['取得単価']),
                                   html.Td(row['取得数'])]) for row in portfolio_table]

            return html.Div([
                dcc.Graph(figure=fig_portfolio),
                html.Hr(),
                html.Table([
                    html.Thead(html.Tr([html.Th("証券コード"), html.Th("取得単価"), html.Th("取得数")])),
                    html.Tbody(table_rows)
                ])
            ])

        elif tab == 'tab-2':
            if df_pf.empty:
                return html.Div("ポートフォリオにデータがありません。銘柄を追加してください。")

            sorted_df = df_pf.sort_values(by='証券コード')
            graphs = []
            for code in sorted_df['証券コード']:
                if code in historical_prices:
                    individual_df = historical_prices[code]
                    fig = px.line(individual_df, x=individual_df.index, y='Close', title=f'{code}の株価推移')
                    graphs.append(dcc.Graph(figure=fig))
            return html.Div(graphs)

        elif tab == 'tab-3':
            return render_add_portfolio()

    # データ追加コールバック
    @app.callback(
        Output('output-state', 'children'),
        Output('tabs-content-example', 'children'),
        Input('submit-button', 'n_clicks'),
        State('input-code', 'value'),
        State('input-price', 'value'),
        State('input-quantity', 'value'),
        State('tabs-example', 'value'),
        prevent_initial_call=True
    )
    def add_portfolio_data(n_clicks, code, price, quantity, tab):
        if n_clicks and code and price is not None and quantity is not None:
            # データベースに新しいレコードを追加
            with engine.connect() as conn:
                conn.execute(portfolio_table.insert().values(
                    証券コード=code,
                    取得単価=price,
                    取得数=quantity
                ))
            df_pf = get_portfolio_data()  # データ追加後に再取得
            return render_add_portfolio(), "データが追加されました"
        return render_tab_content(tab), ""

    # データ編集・削除コールバック
    @app.callback(
        Output('output-state', 'children'),
        Output('tabs-content-example', 'children'),
        Input('edit-button', 'n_clicks'),
        Input('delete-button', 'n_clicks'),
        State('input-id', 'value'),
        State('input-code', 'value'),
        State('input-price', 'value'),
        State('input-quantity', 'value'),
        State('tabs-example', 'value'),
        prevent_initial_call=True
    )
    def edit_delete_portfolio_data(edit_clicks, delete_clicks, id, code, price, quantity, tab):
        ctx = dash.callback_context
        if not ctx.triggered:
            return render_tab_content(tab), ""
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'edit-button' and edit_clicks:
            with engine.connect() as conn:
                conn.execute(portfolio_table.update().where(portfolio_table.c.id == id).values(
                    証券コード=code,
                    取得単価=price,
                    取得数=quantity
                ))
            return render_add_portfolio(), "データが更新されました"
        
        if button_id == 'delete-button' and delete_clicks:
            with engine.connect() as conn:
                conn.execute(portfolio_table.delete().where(portfolio_table.c.id == id))
            return render_add_portfolio(), "データが削除されました"
        
        return render_tab_content(tab), ""

    # 銘柄追加フォームのレンダリング関数
    def render_add_portfolio():
        df_pf = get_portfolio_data()
        
        # 現在のポートフォリオデータを表示するためのテーブル
        portfolio_table = df_pf.to_dict('records')
        table_rows = [html.Tr([html.Td(row['id']),
                               html.Td(row['証券コード']),
                               html.Td(row['取得単価']),
                               html.Td(row['取得数'])]) for row in portfolio_table]

        return html.Div([
            html.H3("銘柄追加"),
            dcc.Input(id='input-id', type='number', placeholder='ID (編集・削除用)'),
            dcc.Input(id='input-code', type='text', placeholder='証券コード'),
            dcc.Input(id='input-price', type='number', placeholder='取得単価'),
            dcc.Input(id='input-quantity', type='number', placeholder='取得数'),
            html.Button('確定', id='submit-button', n_clicks=0),
            html.Button('編集', id='edit-button', n_clicks=0),
            html.Button('削除', id='delete-button', n_clicks=0),
            html.Hr(),
            html.H4("現在のポートフォリオ"),
            html.Table([
                html.Thead(html.Tr([html.Th("ID"), html.Th("証券コード"), html.Th("取得単価"), html.Th("取得数")])),
                html.Tbody(table_rows)
            ])
        ])

    return app

if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True)
