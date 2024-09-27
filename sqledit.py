import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import sqlite3
import pandas as pd

# データベースの初期化
def init_db():
    conn = sqlite3.connect('portfolio.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY,
        Stock_Code TEXT,
        Unit_Price REAL,
        Quantity INTEGER
    )
    ''')
    conn.commit()
    conn.close()

# データベースを初期化
init_db()

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Total Assets', value='tab-1'),
        dcc.Tab(label='By Security', value='tab-2'),
        dcc.Tab(label='Register Security', value='tab-3')
    ]),
    html.Div(id='tabs-content-example'),
    html.Div(id='output-state')  # output-state to display success messages
])

@app.callback(
    Output('tabs-content-example', 'children'),
    Input('tabs-example', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Total Assets Content')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('By Security Content')
        ])
    elif tab == 'tab-3':
        return html.Div([
            dcc.Input(id='stock_code', type='text', placeholder='Stock Code'),
            dcc.Input(id='unit_price', type='number', placeholder='Unit Price'),
            dcc.Input(id='quantity', type='number', placeholder='Quantity'),
            html.Button('Submit', id='submit-val', n_clicks=0),
            html.Button('Edit', id='edit-val', n_clicks=0),
            html.Button('Delete', id='delete-val', n_clicks=0),
            html.Div(id='output'),
            html.Hr(),
            html.Button('Refresh Data', id='refresh-data', n_clicks=0),
            html.Div(id='data-table')
        ])

@app.callback(
    Output('output', 'children'),
    [Input('submit-val', 'n_clicks'),
     Input('edit-val', 'n_clicks'),
     Input('delete-val', 'n_clicks')],
    [State('stock_code', 'value'),
     State('unit_price', 'value'),
     State('quantity', 'value')]
)
def update_output(submit_clicks, edit_clicks, delete_clicks, stock_code, unit_price, quantity):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ''
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    conn = sqlite3.connect('portfolio.db')
    cursor = conn.cursor()

    if button_id == 'submit-val' and stock_code and unit_price is not None and quantity is not None:
        cursor.execute('''
        INSERT INTO portfolio (Stock_Code, Unit_Price, Quantity) VALUES (?, ?, ?)
        ''', (stock_code, unit_price, quantity))
        conn.commit()
        conn.close()
        return f'Data inserted: {stock_code}, {unit_price}, {quantity}'

    elif button_id == 'edit-val' and stock_code and unit_price is not None and quantity is not None:
        cursor.execute('''
        UPDATE portfolio SET Unit_Price = ?, Quantity = ? WHERE Stock_Code = ?
        ''', (unit_price, quantity, stock_code))
        conn.commit()
        conn.close()
        return f'Data updated: {stock_code}, {unit_price}, {quantity}'

    elif button_id == 'delete-val' and stock_code:
        cursor.execute('''
        DELETE FROM portfolio WHERE Stock_Code = ?
        ''', (stock_code,))
        conn.commit()
        conn.close()
        return f'Data deleted: {stock_code}'

@app.callback(
    Output('data-table', 'children'),
    Input('refresh-data', 'n_clicks')
)
def display_data(n_clicks):
    conn = sqlite3.connect('portfolio.db')
    df = pd.read_sql_query('SELECT * FROM portfolio', conn)
    conn.close()
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][col]) for col in df.columns
            ]) for i in range(len(df))
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
