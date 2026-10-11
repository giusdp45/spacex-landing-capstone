"""Interactive launch-site success pie charts and payload/outcome scatter plot."""
from pathlib import Path
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "spacex_launch_dash.csv")
app = Dash(__name__)
app.layout = html.Div([
    html.H1("SpaceX launch analytics", style={"marginBottom": "4px"}),
    html.P("56-launch historical dashboard sample | Explore site, payload and landing outcome"),
    dcc.Dropdown(id="site-dropdown", options=[{"label": "All Sites", "value": "ALL"}] +
                 [{"label": s, "value": s} for s in sorted(df['Launch Site'].unique())], value="ALL", clearable=False),
    dcc.Graph(id="success-pie-chart", style={"height": "360px"}),
    html.Label("Payload range (kg)"),
    dcc.RangeSlider(id="payload-slider", min=0, max=10000, step=500, value=[0, 10000],
                    marks={i: str(i) for i in range(0, 10001, 2000)}, tooltip={"placement": "bottom", "always_visible": True}),
    dcc.Graph(id="success-payload-scatter-chart", style={"height": "360px"}),
    html.P("Class 1 = successful landing label; class 0 = all remaining outcomes. Site comparisons reflect different missions and launch eras.")
], style={"fontFamily": "Arial", "maxWidth": "1150px", "margin": "20px auto", "padding": "0 25px"})


@app.callback(Output("success-pie-chart", "figure"), Input("site-dropdown", "value"))
def success_pie(site):
    if site == "ALL":
        grouped = df.groupby('Launch Site', as_index=False)['class'].sum()
        figure = px.pie(grouped, values='class', names='Launch Site', title='Share of successful landings by launch site (24 total)')
    else:
        counts = df[df['Launch Site'] == site]['class'].value_counts().rename_axis('Class').reset_index(name='Launches')
        counts['Outcome'] = counts['Class'].map({0: 'Class 0: unsuccessful / other', 1: 'Class 1: successful'})
        figure = px.pie(counts, values='Launches', names='Outcome', title=f'{site}: landing outcome distribution',
                        color='Outcome', color_discrete_map={'Class 0: unsuccessful / other': '#ef4444', 'Class 1: successful': '#2563eb'})
    figure.update_traces(textinfo='label+percent', textposition='inside')
    figure.update_layout(margin=dict(l=20, r=20, t=45, b=15))
    return figure


@app.callback(Output("success-payload-scatter-chart", "figure"), Input("site-dropdown", "value"), Input("payload-slider", "value"))
def payload_scatter(site, payload_range):
    filtered = df[df['Payload Mass (kg)'].between(*payload_range)]
    if site != 'ALL':
        filtered = filtered[filtered['Launch Site'] == site]
    figure = px.scatter(filtered, x='Payload Mass (kg)', y='class', color='Booster Version Category',
                        hover_data=['Launch Site', 'Flight Number'], title=f'Payload mass vs landing outcome: {site} ({len(filtered)} launches)',
                        labels={'class': 'Landing class'})
    figure.update_traces(marker={'size': 10, 'opacity': .75})
    figure.update_yaxes(tickvals=[0, 1], range=[-.15, 1.15])
    figure.update_layout(margin=dict(l=45, r=20, t=45, b=40))
    return figure


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8051)
