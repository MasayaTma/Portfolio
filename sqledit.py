import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import sqlite3

app = dash.Dash(__name__)



app.layout = html.Div([
    dcc.Input(id='code', type='text', placeholder='code'),
    dcc.Input(id='price', type='number', placeholder='price'),
    dcc.Input(id='quantity', type='number', placeholder='quantity'),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='output')
])

@app.callback(
    Output('output', 'children'),
    Input('submit-val', 'n_clicks'),
    State('code', 'value'),
    State('price', 'value'),
    State('quantity', 'value')
)
def update_output(n_clicks, code, price, quantity):
    if n_clicks > 0:
        conn = sqlite3.connect('example.db')
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO users (code, price, quantity) VALUES (?, ?)
        ''', (code, price, quantity))
        conn.commit()
        conn.close()
        return f'Data inserted: {code}, {price},{quantity}'

if __name__ == '__main__':
    app.run_server(debug=True)
