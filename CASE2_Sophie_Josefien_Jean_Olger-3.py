#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
#import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
#from plotly.offline import plot
from dash import dcc
from jupyter_dash import JupyterDash
from dash import html
#from dash.dependencies import Input, Output
from dash.dependencies import Input, Output, State


# In[2]:


df = pd.read_csv('Dataset_case2_airplanecrash.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


#het aantal vliegtuigcrashes per jaar in een lijndiagram
df['Year'] = df["Date"].astype(str).str[6:10]
df.head()
aantal_y = df.groupby(['Year'])
aantal_y = pd.DataFrame(aantal_y['Date'].count())
print(aantal_y)


# In[6]:


fig = px.line(aantal_y, title='Aantal vliegtuigongelukken per jaar in de jaren 1908-2019')
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout({'xaxis': {'title': {'text': 'Jaar'}},
                   'yaxis': {'title':{'text': 'Aantal vliegtuigongelukken'}},
                   'legend': {'title':{'text': 'Verloop van het aantal'}}})    
fig.show()
st.plotly_chart(fig)


# In[7]:


#het aantal vliegtuigcrashes per maand in een barplot

df['Month'] = df["Date"].astype(str).str[0:2]
df.head()
aantal_m = df.groupby(['Month'])
aantal_m = pd.DataFrame(aantal_m['Date'].count())
print(aantal_m)


# In[8]:


fig = px.bar(aantal_m, title='Aantal vliegtuigongelukken per maand in de jaren 1908-2019')
fig.update_layout({'xaxis': {'title': {'text': 'Maand'}},
                   'yaxis': {'title':{'text': 'Aantal vliegtuigongelukken'}},
                   'legend': {'title':{'text': 'Aantal'}}})   
fig.show()
st.plotly_chart(fig)


# In[9]:


#het aantal vliegtuigcrashes per dag van de week in een barplot
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day_of_week
df.head()
aantal_d = df.groupby(['Day'])
aantal_d = pd.DataFrame(aantal_d['Date'].count())
print(aantal_d)


# In[10]:


fig = px.bar(aantal_d, title='Aantal vliegtuigongelukken per dag in de jaren 1908-2019')
fig.update_layout({'xaxis': {'title': {'text': 'Dag'}},
                   'yaxis': {'title':{'text': 'Aantal vliegtuigongelukken'}},
                   'legend': {'title':{'text': 'Aantal'}}})   
fig.show()
st.plotly_chart(fig)


# In[11]:


# remove NaN values within two columns
df_drop = df.dropna(subset=['Aboard', 'Fatalities'])




# get total people aboard
aboard = df_drop['Aboard'].sum()
fatalities = df_drop['Fatalities'].sum()

# get total people died
total_fatal = df_drop['Fatalities'].sum()

# survived = aboard minus the people that died
total_survived = aboard - fatalities



# fatalities within crew
crew_fatal = df_drop['Fatalities Crew'].sum()

# total crew aboard
ab_crew = df_drop['Aboard Crew'].sum()

# survival crew
crew_survived = ab_crew - crew_fatal



# fatalities within passengers
passangers_fatal = df_drop['Fatalities Passangers'].sum()

# total passangers aboard
ab_pass = df_drop['Aboard Passangers'].sum()

# survival passengers
passangers_survived = ab_pass - passangers_fatal


# In[12]:


# imports


# data
labels = ['Fatalities', 'Survived']
values = [total_fatal, total_survived]
values_crew = [crew_fatal, crew_survived]
values_pas = [passangers_fatal, passangers_survived]


# plotly setup
fig = go.Figure()

# Add one ore more traces
fig.add_traces(go.Pie(labels=labels, values=values)) # values 1 (first plot)
fig.add_traces(go.Pie(labels=labels, values=values_crew)) # values 1b (second plot)
fig.add_traces(go.Pie(labels=labels, values=values_pas)) # values 1b (second plot)


dropdown_buttons = [
    {'method': 'update', 'label': 'Total',
    'args': [{'visible': [True, False, False]},
            {'title': 'Total'}]},
    {'method': 'update', 'label': 'Crew',
    'args': [{'visible': [False, True, False]},
            {'title': 'Crew'}]}, 
    {'method': 'update', 'label': 'Passengers',
    'args': [{'visible': [False, False, True]},
            {'title': 'Passengers'}]}, 
]

fig.update_layout({
    'updatemenus':[{'type': "dropdown",
        'x': 1.3,
        'y': 0.5,
        'showactive': True,
        'buttons': dropdown_buttons}]})

fig.show()
st.plotly_chart(fig)


# In[13]:


df_time = df.dropna(subset=['Time'])
df_time['Hour'] = df_time["Time"].astype(str).str[0:2]
df_time.drop(df_time[df_time['Hour'] > '23'].index, inplace=True)
aantal_h = df_time.groupby(['Hour'])
aantal_h = pd.DataFrame(aantal_h['Date'].count())
print(aantal_h)


# In[14]:


tijd1 = df_time.loc[(df_time['Hour'] >= '00') & (df_time['Hour'] <= '08')]
groep1 = tijd1.groupby(['Hour'])
groep1 = pd.DataFrame(groep1['Date'].count())


tijd2 = df_time.loc[(df_time['Hour'] >= '09') & (df_time['Hour'] <= '16')]
groep2 = tijd2.groupby(['Hour'])
groep2 = pd.DataFrame(groep2['Date'].count())


tijd3 = df_time.loc[(df_time['Hour'] >= '17') & (df_time['Hour'] <= '23')]
groep3 = tijd3.groupby(['Hour'])
groep3 = pd.DataFrame(groep3['Date'].count())


# In[15]:


fig = px.line(aantal_h, title='Aantal vliegtuigongelukken per uur op een dag in de jaren 1908-2019')
dcc.Checklist(aantal_h.columns, aantal_h.columns.values)
fig.update_layout({'xaxis': {'title': {'text': 'Uur'}},
                   'yaxis': {'title':{'text': 'Aantal vliegtuigongelukken'}},
                   'legend': {'title':{'text': 'Verloop van het aantal'}}})    
fig.show()
st.plotly_chart(fig)


# In[16]:


# !pip install jupyter-dash



# Build App
app = JupyterDash(__name__)
app.layout = html.Div(
    [
        dcc.Checklist(
            id="navy",
            options=[{"label": "Military - U.S. Navy", "value": "navy"}],
            value=[],
            labelStyle={"display": "inline-block"},
        ),
        dcc.Checklist(
            id="private",
            options=[{"label": "Private", "value": "private"}],
            value=[],
            labelStyle={"display": "inline-block"},
        ),
        html.Table([
        html.Tr([html.Td(['Gemiddeld']), html.Td(id='output_mean')]),
        html.Tr([html.Td(['Maximum']), html.Td(id='output_max')]),
    ])
    ]
)



@app.callback(
    Output("output_mean", "children"),
    Output("output_max", "children"),
    [Input("navy", "value")],
    [Input("private", "value")],
)

def get_input(check_1, check_2):
    
    df_check = pd.read_csv('Dataset_case2_airplanecrash.csv')

    
    res_1 = 0
    res_2 = 0
    
    if check_1 == ["navy"] and check_2 == ["private"]:
        
        # select both
        df_check = df_check[df['Operator'].isin(['Military - U.S. Navy', 'Private'])]
        
        res_1 = round(df_check['Aboard'].mean(),2)
        res_2 = round(df_check['Aboard'].max(),2)
        
        print("both")
            
    elif check_1 == ["navy"]:
        
        # select navy
        df_check = df_check[df_check['Operator'] == "Military - U.S. Navy"]
        
        res_1 = round(df_check['Aboard'].mean(),2)
        res_2 = round(df_check['Aboard'].max(),2)
        
                
    elif check_2 == ["private"]:
        
        # select navy
        df_check = df_check[df_check['Operator'] == "Private"]
        
        res_1 = round(df_check['Aboard'].mean(),2)
        res_2 = round(df_check['Aboard'].max(),2)
                

    return res_1, res_2



# Run app and display result inline in the notebook
app.run_server(mode='inline')
streamlit run app.py

# In[ ]:




