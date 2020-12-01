import praw
import nltk
from nltk.corpus import stopwords
from nltk import bigrams

from textblob import TextBlob
import pandas as pd
import itertools
import collections

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input
import dash_table
import plotly.express as px
import plotly.graph_objects as go

from io import BytesIO
from wordcloud import WordCloud
import base64

#nltk.doDELETEwnload('stopwords')
# uncomment the above line and remove delete if running locally for the first time
# heroku will not allow the mentioning of that line, even in the comments if it is to deploy properly


reddit = praw.Reddit(
    client_id="q3i_JQwcT6LiVQ",
    client_secret="0WBVfamfuSueQRVx23poC-Eao-c",
    user_agent="my user agent"
)

# counts words post
def word_count_df(submission):
    split_lower = []
    for top_level_comment in submission.comments:
        split_lower.append(top_level_comment.body.lower().split())

    stop_words = set(stopwords.words('english'))
    split_lower_no_stop = tweets_nsw = [[word for word in split_lower_words if not word in stop_words]
                                        for split_lower_words in split_lower]
    all_words = list(itertools.chain(*split_lower_no_stop))
    all_words_counts = collections.Counter(all_words)
    all_words_counts.most_common(10)
    clean_df = pd.DataFrame(all_words_counts.most_common(10), columns=['words', 'count'])
    return clean_df

# wordcloud
def plot_wordcloud(dataframe):
    d = {a:x for a,x in dataframe.values}
    wc = WordCloud(background_color='#1a1c23')
    wc.fit_words(d)
    return wc.to_image()


# constructs bigrams from post
def bigram_df(submission):
    split_lower = []
    for top_level_comment in submission.comments:
        split_lower.append(top_level_comment.body.lower().split())

    stop_words = set(stopwords.words('english'))
    split_lower_no_stop = [[word for word in split_lower_words if not word in stop_words]
                                        for split_lower_words in split_lower]

    terms_bigram = [list(bigrams(comment)) for comment in split_lower_no_stop]
    bigrams_c = list(itertools.chain(*terms_bigram))
    bigram_counts = collections.Counter(bigrams_c)
    bigram_df = pd.DataFrame(bigram_counts.most_common(5),columns=['bigram', 'count'])
    return bigram_df


tab_style = {
    'font-size': '30px',
    'font-weight': 'bold',
    'color': '#d46161',
    'backgroundColor':'#1a1c23',
    'border-top':'none',
    'border-left':'none',
    'border-right':'none',
    'border-bottom': '2px grey none',
}


tab_selected_style = {
    'font-size': '30px',
    'font-weight': 'bold',
    'color': '#d46161',
    'backgroundColor':'#4f4f4f',
    'border-top':'none',
    'border-left':'none',
    'border-right':'none',
    'border-bottom':'2px grey none',
}

app = dash.Dash(__name__, )
app.title = "Reddit NLP Analysis"
app.config.suppress_callback_exceptions = True
server = app.server

app.layout = html.Div([
    html.H1('A Natural Language Processing Application for Reddit', className='title'),
    dcc.Tabs(
        id='main_tabs',
        value='post_sentiment',
        children=[
            dcc.Tab(value='post_sentiment', label='Reddit Sentiment Analysis',style=tab_style, selected_style=tab_selected_style ),
            dcc.Tab(value='subreddit_sentiment', label='Topic Modelling using LDA',style=tab_style, selected_style=tab_selected_style)
        ], style={'width':'100%', 'margin':'auto'}),
    html.Div(id='render'),
    html.Div([
        html.Br(),
        html.Br(),
        html.A("@TheRealMapleJordan", href = "https://therealmaplejordan.com/"),
        html.Br(),
    ],style={'float':'right'})
],)

post_page = html.Div([
    html.Div(
        className='sub-heading-container',
        children=[
            html.H4('Use AI to find the sentiment of a Reddit post or comment!', className='sub-heading')
    ]),
    html.Div(
        className='input-container',
        children=[
            html.P('Start by entering the URL of a Reddit post here:', className='url-input-heading'),
            dcc.Input(id="post_url", type="url", placeholder="url", size='50', className='url-input-inputs' ,
                  value='https://www.reddit.com/r/Coronavirus/comments/k3weur/absolutely_remarkable_no_one_who_got_modernas/'),
            html.P('Number of Comments:', className='url-input-label'),
            dcc.Input(id='post_limit', type='number', placeholder='post number', value=20, size='5', className='url-input-inputs'),
    ]),
    # hidden div
    html.Div(id='word_count_value', style={'display': 'none'}),
    html.Div(id='polarity_value', style={'display': 'none'}),
    html.Div(id='bigram_value', style={'display':'none'}),
    html.Div(
        className='row-container',
        children=[
            html.H3('Comment Statistics', className='mini-heading'),]),
    html.Div(
        className='row-container',
        children=[
            html.Div(
                className='box-statistics-container',
                children=[
                    html.Div(id='box-statistics')
                ]),
            html.Div(
                className='user-text-input-container',
                children=[
                    html.H3('Enter a piece of text here to see what the sentiment and subjectivity is!', className='input-heading'),
                    dcc.Textarea(id='user_text_input', placeholder='insert some text here',
                            value='TheRealMapleJordan creates amazing content!', draggable=False, className='comment-input'),
                    html.Div(id='user_text_input_polarity')
                ]),
        ]),
        # second row
    html.Div(
        className='row-container',
        children=[
            html.H3('Word Count', className='mini-heading'),
            html.Div(id='word_count_graph', ),
            html.Div(id='word_cloud', className='word-cloud-container'),
        ]),
        # third row
    html.Div(
        className='row-container',
        children=[
            html.H3('Bigram and Upvotes', className='mini-heading'),
            html.Div(id='bigram_count', className='bigram-count'),
            html.Div(
               className='polarity-histogram',
               children=[
                    html.Div(id='post_polarity_graph'),
                    ]),
            html.Div(
                className='upvotes-regression-graph',
                children=[
                    html.Div(id='upvotes_polarity_graph')
                    ]),
        ]),
    html.Div(
        className='row-container',
        children=[
            html.H3('Postive or Negative Feed', className='mini-heading'),
        ]),
    html.Div(
        className='radioitem-container',
        children=[
            dcc.RadioItems(id='negative_positive_radioitem',
                   options=[{'label': i, 'value': i} for i in ['Positive', 'Negative']],
                   value='Positive'
                           )
        ]),
    html.Div(
        className='row-container',
        children=[
            html.Div(className='datatable'
                     ,id='polarity_datatable'),
            html.Div(
                className='TextBox',
                children=[
                    html.H3('Notes about Sentiment Analysis', className='TextBox-heading'),
                    html.P('The sentiment analysis program used in this application is VADER (Valence Aware Dictionary and Sentiment Reasoner). VADER was created specifically for analyzing social media platforms. However one main issue with all sentiment analysis tools is they have a difficult time figuring out sarcasm. Try it yourself by clicking the negative option on the left and see how many sarcastic comments are label with a negative polarity!'),
                    html.Br(),
                ]
            )
        ]),
])

subreddit_page = html.Div(
    html.H1('IN PROGRESS'),
)


# for data processing
@app.callback(
    Output('word_count_value', 'children'),
    Output('polarity_value', 'children'),
    Output('bigram_value', 'children'),
    [Input('post_limit', 'value'),
     Input('post_url', 'value')]
)
def process_data(post_limit, post_url):
    submission = reddit.submission(url=post_url)
    submission.comments.replace_more(limit=post_limit)

    ## polarity calculations
    # HEROKU not working for sentiment_df calculations?
    #####################################
    sentiment_objects = [TextBlob(top_level_comment.body) for top_level_comment in submission.comments]
    sentiment_values = [[comment.sentiment.polarity, str(comment)] for comment in sentiment_objects]
    sentiment_df = pd.DataFrame(sentiment_values, columns=["Polarity", "Comment"])
    sentiment_df = sentiment_df.round(2)
    #upvotes for comments
    upvotes = [top_level_comment.score for top_level_comment in submission.comments]
    sentiment_df['Upvotes'] = upvotes

    return word_count_df(submission).to_json(orient='split'), sentiment_df.to_json(orient='split'), bigram_df(submission).to_json(orient='split')


## box statistics
@app.callback(
    Output('box-statistics', 'children'),
    [Input('polarity_value', 'children')]
)
def box_stat(jsonified_cleaned_data):
    sentiment_df = pd.read_json(jsonified_cleaned_data, orient='split')
    polarity_no_zeros_df = sentiment_df[sentiment_df['Polarity'] != 0].round(2)
    most_upvotes_df = polarity_no_zeros_df.sort_values(by='Upvotes', ascending=False)
    most_upvotes_comment = most_upvotes_df.iloc[0]['Comment']
    most_upvotes_upvotes = most_upvotes_df.iloc[0]['Upvotes']
    most_upvotes_polarity = most_upvotes_df.iloc[0]['Polarity']

    highest_polarity_df =  polarity_no_zeros_df.sort_values(by='Polarity', ascending=False)
    highest_polarity_comment = highest_polarity_df.iloc[0]['Comment']
    highest_polarity_upvotes = highest_polarity_df.iloc[0]['Upvotes']
    highest_polarity_polarity = highest_polarity_df.iloc[0]['Polarity']

    lowest_polarity_comment = highest_polarity_df.iloc[-1]['Comment']
    lowest_polarity_upvotes = highest_polarity_df.iloc[-1]['Upvotes']
    lowest_polarity_polarity = highest_polarity_df.iloc[-1]['Polarity']

    return html.Div([
                html.Div(
                    children=[
                        html.P('Comment With the Most Upvotes', className='comment-headings'),
                        html.P(most_upvotes_comment, className='comment'),
                    ]),
                    html.P('Upvotes: {}, Polarity: {}' .format(most_upvotes_upvotes, most_upvotes_polarity), className='comment2'),
                html.Div(
                    children=[
                        html.P('Comment With the Highest Polarity', className='comment-headings'),
                        html.P(highest_polarity_comment, className='comment'),
                        html.P('Upvotes: {}, Polarity: {}' .format(highest_polarity_upvotes, highest_polarity_polarity), className='comment2')
                    ]),
                html.Div(
                    children=[
                        html.P('Comment With the Lowest Polarity', className='comment-headings' ),
                        html.P(lowest_polarity_comment, className='comment'),
                        html.P('Upvotes: {}, Polarity: {}' .format(lowest_polarity_upvotes, lowest_polarity_polarity), className='comment2')
                    ]),
                ])


## user input results
@app.callback(
    Output('user_text_input_polarity', 'children'),
    [Input('user_text_input', 'value')]
)
def input_polarity(user_text_input):
    TextBlob_object = TextBlob(user_text_input)
    text_polarity = round(TextBlob_object.sentiment.polarity,2)
    text_subjectivity = round(TextBlob_object.sentiment.subjectivity,2)

    return html.Div(
                className='user-results',
                children=[
                    html.P('Input: {}'.format(user_text_input)),
                    html.P('Polarity: {}' .format(text_polarity)),
                    html.P('Subjectivity: {}' .format(text_subjectivity))
                ])


#### post sentiment analysis
## wordcloud
@app.callback(Output('word_cloud', 'children'), [Input('word_count_value', 'children')])
def make_image(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    img = BytesIO()
    plot_wordcloud(dataframe=df).save(img, format='PNG')
    wordcloud_pic = 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
    return html.Img(id="image_wc", src = wordcloud_pic, className='wordcloud'),


### for histogram
@app.callback(
    Output('post_polarity_graph', 'children'),
    [Input('polarity_value', 'children')]
)
def polarity_graph(jsonified_cleaned_data):
    sentiment_df = pd.read_json(jsonified_cleaned_data, orient='split')
    polarity_no_zeros_df = sentiment_df[sentiment_df['Polarity'] != 0]
    fig = go.Figure(data=go.Histogram(x=polarity_no_zeros_df['Polarity'], marker=dict(
                    color='#415085')
                    ))
#    fig = px.histogram(polarity_no_zeros_df, x='Polarity')
    fig.update_layout(
        title='Post Polarity (removing neutral polarity)',
        paper_bgcolor='#1a1c23',
        plot_bgcolor='#1a1c23',
        font=dict(
            color='#d46161'),
        xaxis=dict(
            title='Polarity',
        ),
        yaxis=dict(
            title='Count',
            gridcolor='#3a3d4a'),

    )

    return html.Div(
        dcc.Graph(figure=fig)
    )


# polarity datatable
@app.callback(
    Output('polarity_datatable', 'children'),
    [Input('polarity_value', 'children'),
     Input('negative_positive_radioitem', 'value')]
)
def polarity_datatable(jsonified_cleaned_data, neg_or_pos):
    sentiment_df = pd.read_json(jsonified_cleaned_data, orient='split')
    polarity_no_zeros_df = sentiment_df[sentiment_df['Polarity'] != 0]
    if neg_or_pos == 'Positive':
        datatable_df = polarity_no_zeros_df[polarity_no_zeros_df['Polarity'] > 0].head(6)
        datatable_df = datatable_df.sort_values(by='Polarity', ascending=False)
        if datatable_df.empty:
            return html.P('There are no positive comments for this search result :(')
        else:
            return html.Div([dash_table.DataTable(
                columns=[{'name': i, 'id': i} for i in datatable_df.columns],style_table={'overflowX': 'auto'},
                    data=datatable_df.to_dict('records'), style_cell={'textAlign': 'left', 'whiteSpace': 'normal',
                    'height': 'auto','backgroundColor':'#1a1c23', 'color':'lightgrey' }, style_header = {'font-weight':'bold', }
                                                 )])
    elif neg_or_pos == 'Negative':
        datatable_df = polarity_no_zeros_df[polarity_no_zeros_df['Polarity'] < 0].head(6)
        datatable_df = datatable_df.sort_values(by='Polarity', ascending=True)
        if datatable_df.empty:
            return html.P('There are no negative comments for this search result :)')
        else:
            return html.Div([dash_table.DataTable(
                columns=[{'name': i, 'id': i} for i in datatable_df.columns],style_table={'overflowX': 'auto'},
                    data=datatable_df.to_dict('records'), style_cell={'textAlign': 'left', 'whiteSpace': 'normal',
                     'height': 'auto', 'backgroundColor':'#1a1c23', 'color':'lightgrey'}, style_header = {'font-weight':'bold'})
                             ])


# for word count
@app.callback(
Output("word_count_graph", 'children'),
[Input('word_count_value', 'children')]
)
def update_word_count(jsonified_cleaned_data):
    clean_df = pd.read_json(jsonified_cleaned_data, orient='split')
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=clean_df['count'].iloc[::-1],
            y=clean_df['words'].iloc[::-1],
            orientation='h',
            marker=dict(
                color='#db8a27'
            )
        )
    )
    fig.update_layout(
        title='Most Common Words (minus stopwords)',
        paper_bgcolor='#1a1c23',
        plot_bgcolor='#1a1c23',
        font=dict(
            color='#d46161'),
        xaxis=dict(
            title='Count',
            gridcolor='#3a3d4a'),
        yaxis=dict(
            title='Words',
            gridcolor='#1a1c23'
            )
    )
    return html.Div(
                className='word-count-graph',
                children=[
                    dcc.Graph(figure=fig)], )


# regression graph
@app.callback(
    Output('upvotes_polarity_graph', 'children'),
    [Input('polarity_value', 'children')]
)
def regression_graph(jsonified_cleaned_data):
    data_df = pd.read_json(jsonified_cleaned_data, orient='split')
    fig =  go.Figure(
        go.Scatter(x=data_df.Polarity, y=data_df.Upvotes, marker=dict(color='#415085'), mode='markers')
    )
#    fig = px.scatter(data_df, x='Polarity', y='Upvotes', #trendline='ols', trendline_color_override='#d46161'    )
    fig.update_layout(
        title='Upvotes vs Polarity',
        paper_bgcolor='#1a1c23',
        plot_bgcolor='#1a1c23',
        font=dict(
            color='#d46161'
        ),
        xaxis=dict(
            title='Polarity',
            gridcolor='#3a3d4a',
            zeroline=False,
        ),
        yaxis=dict(
            title='Upvotes',
            gridcolor='#3a3d4a',
            zeroline=False,
        )
    )
    return html.Div(
        dcc.Graph(figure=fig)
    )


#bigram
@app.callback(
    Output('bigram_count', 'children'),
    [Input('bigram_value', 'children')]
)

def create_bigram(jsonified_cleaned_data):
    bigram_df = pd.read_json(jsonified_cleaned_data, orient='split')
    return html.Div(
                className='top-bigrams',
                children=[
                    #html.P('Bigrams are pairs of words that follow each other'),
                    html.H3('Most common bigrams (word pairings) are the following:', className='TextBox-heading'),
                    html.P('({0}-{1}), count: {2}' .format(bigram_df['bigram'][0][0], bigram_df['bigram'][0][1], bigram_df['count'][0])),
                    html.P('({0}-{1}), count: {2}' .format(bigram_df['bigram'][1][0], bigram_df['bigram'][1][1], bigram_df['count'][1])),
                    html.P('({0}-{1}), count: {2}' .format(bigram_df['bigram'][2][0], bigram_df['bigram'][2][1], bigram_df['count'][2])),
                    html.P('({0}-{1}), count: {2}' .format(bigram_df['bigram'][3][0], bigram_df['bigram'][3][1], bigram_df['count'][3])),
                    html.P('({0}-{1}), count: {2}' .format(bigram_df['bigram'][4][0], bigram_df['bigram'][4][1], bigram_df['count'][4])),

                ])


#### tab control
@app.callback(
    Output('render', 'children'),
    [Input('main_tabs', 'value')]
)
def render_page(tab_value):
    if tab_value == 'subreddit_sentiment':
        return subreddit_page
    else:
        return post_page


if __name__ == "__main__":
    app.run_server(debug=True)






#
