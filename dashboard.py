
"""
Created on Wed Jun 20 16:48:41 2018

@author: bukowskio
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from IPython.display import Markdown


app = dash.Dash()

df=pd.read_csv('~/Oskar/energy-prices-forecasting/predictions.csv')
df2=pd.read_csv('~/Oskar/energy-prices-forecasting/dane.csv')

df2=df2.dropna()


# RDN forecast calculations

rdn_errors=df[['rdn_xgb', 'rdn_lstm']].apply(lambda x:x-df['rdn'])
cro_errors=df[['cro_xgb', 'cro_lstm']].apply(lambda x:x-df['cro'])

rdn_errors.columns=['xgb','lstm']
cro_errors.columns=['xgb','lstm']

rdn_errors2=np.abs(rdn_errors)
cro_errors2=np.abs(rdn_errors)

rdn_errors3=rdn_errors2.apply(np.mean,axis=0).sort_values()
cro_errors3=cro_errors2.apply(np.mean,axis=0).sort_values()


# descriptive analysis

df2['cro_peak']=np.where(df2['cro']>np.percentile(df2['cro'],90),'cro_peak','cro')
df2['rdn_peak']=np.where(df2['rdn']>np.percentile(df2['rdn'],90),'rdn_peak','rdn')

x01=pd.pivot_table(df2,index='rdn_peak',columns='cro_peak',values='demand',aggfunc=np.mean)
x02=pd.pivot_table(df2,index='rdn_peak',columns='cro_peak',values='supply1',aggfunc=np.mean)
x03=pd.pivot_table(df2,index='rdn_peak',columns='cro_peak',values='supply2',aggfunc=np.mean)
x04=pd.pivot_table(df2,index='rdn_peak',columns='cro_peak',values='wind',aggfunc=np.mean)
x05=pd.pivot_table(df2,index='rdn_peak',columns='cro_peak',values='reserve',aggfunc=np.mean)
x06=pd.pivot_table(df2,index='rdn_peak',columns='cro_peak',values='se',aggfunc=np.mean)


# plottings

app.layout = html.Div(children=[
    
    html.H1(children='Forecst report'),

    html.H3(children='RDN - CRO compariosn'),

           
    dcc.Graph(
        id='rdn-cro',
        figure={
            'data': [
                go.Scatter(
                        #x=df['data'],
                        y=df['rdn'],
                        line=dict(color='rgb(20,20,200)'),
                        name='rdn'),
                go.Scatter(
                        #x=df['data'],
                        y=df['cro'],
                        line=dict(color='rgb(20,200,20)'),
                        name='cro')
            ],
            'layout': go.Layout(
                title='RDN - CRO',
                autosize=False,
                width=1200,
                height=600
            )
        }
    ),       

    # first graph

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                go.Bar(
                        x=list(rdn_errors3.index),
                        y=list(rdn_errors3.values),
                        marker=dict(color='rgb(20,150,200)'),
                        name='rdn'),
                go.Bar(
                        x=list(rdn_errors3.index), 
                        y=list(rdn_errors3.values), 
                        marker=dict(color='rgb(150,30,200)'),
                        name='cro')
            ],
            'layout': go.Layout(
                title='RDN forecasts mean percentage errors',
                autosize=False,
                width=1200,
                height=600
            )
        }
    ),
    #html.H2('RDN forecasts'),

    dcc.Graph(
        id='eee1',
        figure={
            'data': [
                go.Scatter(
                        y=df['rdn'],
                        line=dict(color='rgb(20,20,200)'),
                        name='real'),
                go.Scatter(
                        y=df['rdn_xgb'],
                        line=dict(color='rgb(20,150,200)'),
                        name='xgb'),
                go.Scatter(
                        y=df['rdn_lstm'],
                        line=dict(color='rgb(150,20,200)'),
                        name='lstm')
            ],
            'layout': go.Layout(
                title='RDN real + forecasts',
                autosize=False,
                width=1200,
                height=600
            )
        }
    ),
        
    dcc.Graph(
        id='hist1',
        figure={
            'data': [
                go.Histogram(
                        x=rdn_errors['xgb'],
                        marker=dict(color='rgb(20,150,200)'),
                        name='xgb'),
                go.Histogram(
                        x=rdn_errors['lstm'],
                        marker=dict(color='rgb(150,20,200)'),
                        name='lstm')
            ],
            'layout': go.Layout(
                title='RDN real + forecasts',
                autosize=False,
                width=1200,
                height=600
            )
        }
    ),
        
    dcc.Graph(
        id='hist2',
        figure={
            'data': [
                go.Histogram(
                        x=rdn_errors2['xgb'],
                        marker=dict(color='rgb(20,150,200)'),
                        name='xgb'),
                go.Histogram(
                        x=rdn_errors2['lstm'],
                        marker=dict(color='rgb(150,20,200)'),
                        name='lstm')
            ],
            'layout': go.Layout(
                title='RDN real + forecasts',
                autosize=False,
                width=1200,
                height=600
            )
        }
    ),
        
    dcc.Graph(
        id='eee2',
        figure={
            'data': [
                go.Scatter(
                        y=df['cro'],
                        line=dict(color='rgb(20,20,200)'),
                        name='real'),
                go.Scatter(
                        y=df['cro_xgb'],
                        line=dict(color='rgb(20,150,200)'),
                        name='xgb'),
                go.Scatter(
                        y=df['cro_lstm'],
                        line=dict(color='rgb(150,20,200)'),
                        name='lstm')
            ],
            'layout': go.Layout(
                title='CRO real + forecasts',
                autosize=False,
                width=1200,
                height=600
            )
        }
    ),
        
    dcc.Graph(
        id='hist3',
        figure={
            'data': [
                go.Histogram(
                        x=cro_errors['xgb'],
                        marker=dict(color='rgb(20,150,200)'),
                        name='xgb'),
                go.Histogram(
                        x=cro_errors['lstm'],
                        marker=dict(color='rgb(150,20,200)'),
                        name='lstm')
            ],
            'layout': go.Layout(
                title='RDN real + forecasts',
                autosize=False,
                width=1200,
                height=600
            )
        }
    ),
        
    dcc.Graph(
        id='hist4',
        figure={
            'data': [
                go.Histogram(
                        x=cro_errors2['xgb'],
                        marker=dict(color='rgb(20,150,200)'),
                        name='xgb'),
                go.Histogram(
                        x=cro_errors2['lstm'],
                        marker=dict(color='rgb(150,20,200)'),
                        name='lstm')
            ],
            'layout': go.Layout(
                title='RDN real + forecasts',
                autosize=False,
                width=1200,
                height=600
            )
        }
    ),
        
    html.H2('descriptive analysis'),
     
     dcc.Graph(
        id='zapotrzebowanie heat-map',
        figure={
            'data': [
                go.Heatmap(
                        z=np.array(x01),
                        x=x01.columns,
                        y=x01.index,
                        colorscale = 'Viridis')
            ],
            'layout': go.Layout(
                title='relation between prices and demand',
                autosize=False,
                width=800,
                height=500
            )
        }
    ),
    
    dcc.Graph(
        id='jwcd heat-map',
        figure={
            'data': [
                go.Heatmap(
                        z=np.array(x02),
                        x=x02.columns,
                        y=x02.index,
                        colorscale = 'Viridis')
            ],
            'layout': go.Layout(
                title='relation between prices and jwcd',
                autosize=False,
                width=800,
                height=500
            )
        }
    ),
        
    dcc.Graph(
        id='njwcd heat-map',
        figure={
            'data': [
                go.Heatmap(
                        z=np.array(x03),
                        x=x03.columns,
                        y=x03.index,
                        colorscale = 'Viridis')
            ],
            'layout': go.Layout(
                title='relation between prices and njwcd',
                autosize=False,
                width=800,
                height=500
            )
        }
    ),
        
    dcc.Graph(
        id='wind heat-map',
        figure={
            'data': [
                go.Heatmap(
                        z=np.array(x04),
                        x=x04.columns,
                        y=x04.index,
                        colorscale = 'Viridis')
            ],
            'layout': go.Layout(
                title='relation between prices and wind',
                autosize=False,
                width=800,
                height=500
            )
        }
    ),
        
    dcc.Graph(
        id='SE heat-map',
        figure={
            'data': [
                go.Heatmap(
                        z=np.array(x06),
                        x=x06.columns,
                        y=x06.index,
                        colorscale = 'Viridis')
            ],
            'layout': go.Layout(
                title='relation between prices and SE prices',
                autosize=False,
                width=800,
                height=500
            )
        }
    ),
        
    dcc.Graph(
        id='LT heat-map',
        figure={
            'data': [
                go.Heatmap(
                        z=np.array(x05),
                        x=x05.columns,
                        y=x05.index,
                        colorscale = 'Viridis')
            ],
            'layout': go.Layout(
                title='relation between prices and LT prices',
                autosize=False,
                width=800,
                height=500
            )
        }
    )
        

])


if __name__ == '__main__':
    app.run_server(debug=True)