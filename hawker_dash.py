import dataiku
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
from dash import no_update
import dash_table
import plotly.figure_factory as ff
import numpy as np

# use the style of examples on the Plotly documentation
# app.config.external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

dataset = dataiku.Dataset("hawker_stacked_prepared")
df = dataset.get_dataframe()

Food = df.food_name.unique().tolist()


# url, root-url and first-loading are used for routing
url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='root-url', style={'display': 'none'}),
    html.Div(id='first-loading', style={'display': 'none'}),
    html.Div(id='page-content')
])

layout_index = html.Div(children=[
    html.H1(children='Healthiest and Most Unhealthy Singapore Hawker Food 🇸🇬 🍽️', style={
        'textAlign': 'center','font-family':'sans-serif'}, className="header-title"),
    html.H3(children='Learn about the nutritional content of popular Singapore hawker foods here! Data source: healthxchange.sg.', style={
        'textAlign': 'center','font-family':'sans-serif'}),

    html.Div(html.P([html.Br()])),

    html.Plaintext(children='Filter the data in the table below to select foods of interest.', style={
    'font-family':'sans-serif', 'font-size':15,'color': 'blue'}),

    dcc.Markdown('''
     Eg. You can enter `Chinese` under "type" column and `<500` under "kcal" column. You can also sort the values for each column by clicking the arrow buttons. 
    ''', style={
    'font-family':'sans-serif', 'font-size':15,'color': 'blue'}),

    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": False} for i in ['food','type','kcal',"protein(g)", "fat(g)", "carbs(g)","cholesterol(mg)","sodium(mg)"]#df.columns
        ],
        data=df.to_dict('records'),
        style_cell={'fontSize':13, 'font-family':'sans-serif'},
        style_cell_conditional=[
        {
            'if': {'column_id': c},
            'textAlign': 'left'
        } for c in ['food', 'type']
        ],
        editable=True,
        filter_action="native",
        filter_options={"case": "insensitive"},
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10,
        export_format="csv",
    ),
    html.Div(id='datatable-interactivity-container'),

    html.Div(html.P([html.Br()])),

    html.Plaintext(children='Filter to get health tips related.', style={
    'font-family':'sans-serif', 'font-size':15,'color': 'blue'}),

    dcc.Dropdown(
            id='filter_dropdown',
            options=[{'label':st, 'value':st} for st in Food],
            value = Food[0],
            style={'fontSize':13, 'font-family':'sans-serif','textAlign': 'left','color': 'black'},
            ),

#https://dash.plotly.com/datatable/interactivity    
#https://github.com/plotly/dash-table/pull/916
    dash_table.DataTable(id='table-container',
        css=[dict(selector="p", rule="margin: 0px;")],
        data=df.to_dict("records"),
        columns = [{"id": "image", "name": "", "presentation": "markdown"},
                   {"id":"comments","name":""}],                 
                   #{"name": i, "id": i} for i in ['comments']], 
        markdown_options={"html": True},
        #style_table={"height": 80},
        style_cell={'fontSize':15, 'font-family':'sans-serif','textAlign': 'left'},
        style_header={'height': 0, 'padding': 0, 'border': 'none'},  # Make headers invisible
        style_data={
        'color': 'white',
        'backgroundColor': 'grey',
        'whiteSpace': 'normal', 'height': 'auto'
    },
                        ), 

    html.Div(html.P([html.Br()])),

    dcc.Link('Next >>>', href='page-2',
             style={'fontSize':13, 'font-family':'sans-serif',
                    'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center',
                    'color': 'black'}),
])


### Page 2 charts
#app = dash.Dash(__name__)
#fig = px.bar(df, x="", y="", color="type", barmode="group")

fig = px.scatter(df, x="kcal", y="protein(g)", color="type",
                 size='food_weight', hover_data=['food_name'])

fig_weighted = px.scatter(df, x="kcalperg", y="proteinperg", color="type",
                 size='food_weight', hover_data=['food_name'],
                 labels={
                     "kcalperg": "kcal per 100g",
                     "proteinperg": "protein(g) per 100g",
                 },)

fig_2 = px.scatter_matrix(df,
    dimensions=['kcal',"protein(g)", "fat(g)","saturatedfat(g)","dietaryfibre(g)", "carbs(g)","cholesterol(mg)","sodium(mg)"],
    color="type",hover_data=['food_name'])

df2 = df[['kcal',"protein(g)", "fat(g)", "saturatedfat(g)","dietaryfibre(g)","carbs(g)","cholesterol(mg)","sodium(mg)"]]
corr = df2.corr().round(3)
mask = np.triu(np.ones_like(corr, dtype=bool))
df_mask = corr.mask(mask)

fig_3 = ff.create_annotated_heatmap(z=df_mask.to_numpy(), 
                                  x=list(df_mask[['kcal',"protein(g)", "fat(g)","saturatedfat(g)","dietaryfibre(g)", "carbs(g)","cholesterol(mg)","sodium(mg)"]]),#df_mask.columns.tolist(),
                                  y=list(df_mask[['kcal',"protein(g)", "fat(g)","saturatedfat(g)","dietaryfibre(g)", "carbs(g)","cholesterol(mg)","sodium(mg)"]]),#df_mask.columns.tolist(),
                                  #colorscale=px.colors.diverging.RdBu,
                                  hoverinfo="none", #Shows hoverinfo for null values
                                  showscale=True, ygap=1, xgap=1
                                 )


layout_page_2 = html.Div([
    html.H1(children='Healthiest and Most Unhealthy Singapore Hawker Food 🇸🇬 🍽️', style={
        'textAlign': 'center','font-family':'sans-serif'}, className="header-title"),

    html.H3(children='The following charts show the relationships between nutritional values.', style={
        'textAlign': 'center','font-family':'sans-serif'}),

    html.Div(html.P([html.Br()])),

    html.Plaintext(children='Hover over the points to see what food they are. Note: The size indicates the weight of the food.', style={
    'font-family':'sans-serif', 'font-size':15,'color': 'blue'}),

    dcc.Graph(
        id='example-graph',
        figure=fig, 
        clear_on_unhover=True
    ),
    dcc.Tooltip(id="graph-tooltip"),

    dcc.Graph(
        id='weighted-graph',
        figure=fig_weighted, 
        clear_on_unhover=True
    ),
    dcc.Tooltip(id="graph-tooltip-2"),

    dcc.Graph(
        id='scatter-graph',
        figure=fig_2

    ),

    dcc.Graph(
        id='corr-graph',
        figure=fig_3

    ),

    html.Plaintext(children='Only fat and carbs are strongly correlated with kcal, and fat with saturated fat, while the rest of the nutritional contents are not very strongly correlated with one another.', style={
    'font-family':'sans-serif', 'font-size':15}),

    html.Div(html.P([html.Br()])),

    dcc.Link('<<< Back', id='page-1-root-link', href='',
            style={'fontSize':13, 'font-family':'sans-serif',
                    'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center',
                    'color': 'black'})
])

fig.update_layout(
    title='Overview of hawker food by kcal, protein and weight',
    font=dict(size=11),
    width=1000,
    #height=900,
)

fig_weighted.update_layout(
    title='Overview of hawker food by kcal, protein per 100g',
    font=dict(size=11),
    width=1000,
    #height=900,
)

#https://dash.plotly.com/dash-core-components/tooltip
fig.update_traces(hoverinfo="none", hovertemplate=None)
fig_weighted.update_traces(hoverinfo="none", hovertemplate=None)


fig_2.update_layout(
    title='Scatterplot of nutritional content',
    font=dict(size=11),
    width=1200,
    height=1100,
)


fig_3.update_xaxes(side="bottom")

fig_3.update_layout(
    title_text='Correlation matrix of nutritional content',
    font=dict(size=11),
    #title_x=0.5, 
    width=600, 
    height=600,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis_zeroline=False,
    yaxis_zeroline=False,
    yaxis_autorange='reversed',
    template='plotly_white'
)

# NaN values are not handled automatically and are displayed in the figure
# So we need to get rid of the text manually
for i in range(len(fig_3.layout.annotations)):
    if fig_3.layout.annotations[i].text == 'nan':
        fig_3.layout.annotations[i].text = ""


# index layout
app.layout = url_bar_and_content_div

# "complete" layout, need at least Dash 1.12
app.validation_layout = html.Div([
    url_bar_and_content_div,
    layout_index,
    layout_page_2,
])

# The following callback is used to dynamically instantiate the root-url
@app.callback([dash.dependencies.Output('root-url', 'children'), dash.dependencies.Output('first-loading', 'children')],
              dash.dependencies.Input('url', 'pathname'),
              dash.dependencies.State('first-loading', 'children')
              )
def update_root_url(pathname, first_loading):
    if first_loading is None:
        return pathname, True
    else:
        raise PreventUpdate

# We can now use the hidden root-url div to update the link in page-1 and page-2
@app.callback(dash.dependencies.Output('page-1-root-link', 'href'),
              [dash.dependencies.Input('root-url', 'children')])
def update_root_link(root_url):
    return root_url

@app.callback(dash.dependencies.Output('page-2-root-link', 'href'),
              [dash.dependencies.Input('root-url', 'children')])
def update_root_link(root_url):
    return root_url

# This is the callback doing the routing
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [
                  dash.dependencies.Input('root-url', 'children'),
                  dash.dependencies.Input('url', 'pathname')
              ])
def display_page(root_url, pathname):
    if root_url + "page-2" == pathname :
        return layout_page_2
    else:
        return layout_index

### Page 1 callbacks
@app.callback(
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('datatable-interactivity', 'selected_columns')
)
def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

@app.callback(
    Output('datatable-interactivity-container', "children"),
    Input('datatable-interactivity', "derived_virtual_data"),
    Input('datatable-interactivity', "derived_virtual_selected_rows"))
def update_graphs(rows, derived_virtual_selected_rows):
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncrasy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = df if rows is None else pd.DataFrame(rows)

    colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'
              for i in range(len(dff))]

    return [
        dcc.Graph(
            id=column,
            figure={
                "data": [
                    {
                        "x": dff["food_name"],
                        "y": dff[column],
                        "type": "bar",
                        "marker": {"color": colors}, 
                    }
                ],
                "layout": {
                    "xaxis": {"automargin": True, "rangeselector_font_size": 6}, #
                    "yaxis": {
                        "automargin": True,
                        "title": {"text": column}
                    },
                    "height": 450,
                    "margin": {"t": 10, "l": 10, "r": 10},
                    "font": {"size": 11}
                },
            },
        )

        # check if column exists - user may have deleted it
        # If `column.deletable=False`, then you don't
        # need to do this check.
        for column in ['kcal',"protein(g)", "fat(g)", "carbs(g)","cholesterol(mg)","sodium(mg)"] if column in dff
    ]

#if __name__ == '__main__':
#    app.run_server(debug=True)



@app.callback(
    Output('table-container', 'data'),
    [Input('filter_dropdown', 'value') ] )
def display_table(state):
    dff = df[df.food_name==state]

    #return [dash_table.DataTable(data=dff, columns=columns)]
    return dff.to_dict('records')



### Page 2 callbacks
@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("example-graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    x = pt["x"]
    y = pt["y"]
    bbox = pt['bbox']

    df_row = df[(df['kcal']==x) & (df['protein(g)']==y)]
    img_src = df_row['image_link']
    food_weight = str(df_row['food_weight'].values[0])
    name = str(df_row['food_name'].values[0])
    form = str(df_row['protein(g)'].values[0])
    kcal = str(df_row['kcal'].values[0])
    #desc = df_row['DESC']
    #if len(desc) > 300:
    #    desc = desc[:100] + '...'

    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "100%"}),
            html.H2(f"{name}, {food_weight}g", style={"color": "darkblue",'fontSize':14,'font-family':'sans-serif'}),
            #html.Plaintext(f"food weight(g)={food_weight}", style={'fontSize':12,'font-family':'sans-serif'}),
            html.Plaintext(f"protein(g)={form}, kcal={kcal}", style={'fontSize':13,'font-family':'sans-serif'}),
            #html.P(f"kcal={kcal}"),
        ], style={'width': '200px', 'white-space': 'normal','fontSize':15,'font-family':'sans-serif'})
    ]

    return True, bbox, children


@app.callback(
    Output("graph-tooltip-2", "show"),
    Output("graph-tooltip-2", "bbox"),
    Output("graph-tooltip-2", "children"),
    Input("weighted-graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    x = pt["x"]
    y = pt["y"]
    bbox = pt['bbox']

    df_row = df[(df['kcalperg']==x) & (df['proteinperg']==y)]
    img_src = df_row['image_link']
    food_weight = str(df_row['food_weight'].values[0])
    name = str(df_row['food_name'].values[0])
    form = str(df_row['protein(g)'].values[0])
    kcal = str(df_row['kcal'].values[0])
    #desc = df_row['DESC']
    #if len(desc) > 300:
    #    desc = desc[:100] + '...'

    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "100%"}),
            html.H2(f"{name}, {food_weight}g", style={"color": "darkblue",'fontSize':14,'font-family':'sans-serif'}),
            #html.Plaintext(f"food weight(g)={food_weight}", style={'fontSize':12,'font-family':'sans-serif'}),
            html.Plaintext(f"protein(g)={form}, kcal={kcal}", style={'fontSize':13,'font-family':'sans-serif'}),
            #html.P(f"kcal={kcal}"),
        ], style={'width': '200px', 'white-space': 'normal','fontSize':15,'font-family':'sans-serif'})
    ]

    return True, bbox, children
