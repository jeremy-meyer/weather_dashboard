from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(children=[
    dbc.Row(children=[
        dbc.Col([
            html.H4('GRAPH COL 1'),
        ], width=7),
        dbc.Col([
            html.H4('GRAPH COL 2'),
        ], width=5)
    ]),
    # dbc.Row([
    #     dbc.Col([
    #         html.H4('Test2')
    #     ], width=6),
    #     dbc.Col([
    #         html.H4('Test3')
    #     ], width=6)
    # ]),
])

app.run(debug=True, port=8083)