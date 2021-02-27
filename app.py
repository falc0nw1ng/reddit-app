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

import gensim
import gensim.corpora as corpora
from gensim.parsing.preprocessing import preprocess_documents
import sys
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')
stopwords = stopwords.words('english')
stopwords.extend(['from', 'subject', 're', 'edu', 'use', 'www', 'http'])
tokenizer = RegexpTokenizer(r'\w+')

# uncomment the above line and remove delete if running locally for the first time
# heroku will not allow the mentioning of that line, even in the comments if it is to deploy properly


reddit = praw.Reddit(
    client_id="q3i_JQwcT6LiVQ",
    client_secret="0WBVfamfuSueQRVx23poC-Eao-c",
    user_agent="my user agent"
)

# counts words post
def word_count_df(submission):
    lower_comments=[]
    for comment in submission.comments.list():
        lower_comments.append(comment.body.lower())
    # remove punctuations
    no_punc = [tokenizer.tokenize(comment) for comment in lower_comments]
    # remove stopwords
    # stop_words = stopwords.words('english')
    # stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'www',])
    no_stopwords_punc = [[word for word in comment if not word in stopwords] for comment in no_punc]
    flattened = [word for comment in no_stopwords_punc for word in comment]
    word_count = collections.Counter(flattened)

    terms_bigram = [list(bigrams(comment)) for comment in no_stopwords_punc]
    bigrams_c = list(itertools.chain(*terms_bigram))
    bigram_counts = collections.Counter(bigrams_c)

    # combine together
    clean_df = pd.DataFrame(word_count.most_common(10), columns=['Word', 'Count'])
    temp_df = pd.DataFrame(bigram_counts.most_common(10), columns=['Bigram', 'Count (Bigram)'])
    clean_df['Bigram'] = temp_df['Bigram']
    clean_df['Count (Bigram)'] = temp_df['Count (Bigram)']
    return clean_df


def LDA_df(submission):
    lower_comments=[]
    for comment in submission.comments.list():
        lower_comments.append(comment.body.lower())

    tokenizer = RegexpTokenizer(r'\w+')
    no_punc = [tokenizer.tokenize(comment) for comment in lower_comments]
    no_stopwords = [[word for word in comment if not word in stopwords]for comment in no_punc]
    id2word = corpora.Dictionary(no_stopwords)
    corpus = [id2word.doc2bow(text) for text in no_stopwords]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=20,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)

    topics = lda_model.print_topics(num_topics=5, num_words=10)
    temp_df = pd.DataFrame(topics, columns=[['Number', 'Topics']])
    topic_list = []
    for index_value in range(0,5):
        topic_list.append(temp_df.Topics.loc[index_value].str.findall(r'[a-zA-Z]+'))

    weight_list = []
    for index_value in range(0,5):
        weight_list.append(temp_df.Topics.loc[index_value].str.findall(r'[0-9.0-9]+'))
    df = pd.DataFrame({'Topics':topic_list, 'Weight':weight_list})
    return df

selected_style = {
    'font-size': '20px',
    'font-weight': 'lighter',
    'color': '#ff9100',
    'backgroundColor':'#212121',
    'borderTop': '3px solid #ff9100',

}

app = dash.Dash(__name__, )
app.title = "Reddit NLP Analysis"
app.config.suppress_callback_exceptions = True
server = app.server

app.layout = html.Div([
    html.Div(
        className='container1',
        children=[
            html.Div(
                className='input-container',
                children=[
                    html.Span('A ', className='title'),
                    html.A('Reddit ', href='https://therealmaplejordan.com/', className='title-span', target='_blank'),
                    html.Span('NLP Analysis Application', className='title'),
                    html.H3('Enter Reddit post URL', className='input-heading'),
                    dcc.Input(id="post_url", type="url", placeholder="url", size='30' ,
                          value='https://www.reddit.com/r/Coronavirus/comments/k3weur/absolutely_remarkable_no_one_who_got_modernas/', className='input-url'),
                    html.H3('Number of branches of comments', className='input-heading'),
                    dcc.Input(id='threshold', type='number', placeholder='post number', value=20, size='5', className='input-branch'),
                    dcc.Store(id='word_count_value'),
                    dcc.Store(id='polarity_value',),
                    dcc.Store(id='LDA_value')
                ]),
                html.Div(
                    className='post-statistics-container',
                    children=[
                        html.H3('Post Statistics', className='post-statistics-heading'),
                        html.Div(id='post_statistics')
                        ]
                ),
        ]
    ),
    html.Div(
        className='tab-container',
        children=[
            dcc.Tabs(
                className = 'main-tab',
                id='sentiment-topic-tabs',
                value='sentiment_tab',
                children=[
                    dcc.Tab(value='sentiment_tab', label='Sentiment Analysis', className='tab-style', selected_style=selected_style,),
                    dcc.Tab(value='topic_tab', label='Topic Modeling', className='tab-style', selected_style=selected_style,),
                    dcc.Tab(value='other', label='Word/Bigram Frequency', className='tab-style', selected_style=selected_style,)
                ]),
            html.Div(id='tab_output', className='tab-output')
        ]),
    html.Div(
        className='LSI-container',
        children=[
            html.Div(
                className='hover-container',
                children=[
                    html.H3('Latent Semantic Indexing', className='LSI-heading'),
                    html.Span('Enter query parameters here to find similar comments in subreddit post (Be patient!)', className='LSI-hover'),
                ]
            ),
            dcc.Textarea(id='LSI_input', placeholder='Enter query parameters', value='', draggable=False, className='LSI-textbox'),
            html.Div(id='LSI_output', className='LSI-output-container')
        ]),
],)


# for data processing
@app.callback(
    Output('word_count_value', 'data'),
    Output('polarity_value', 'data'),
    [Input('threshold', 'value'),
     Input('post_url', 'value')]
)

def process_data(threshold, post_url):
    submission = reddit.submission(url=post_url)
    submission.comments.replace_more(limit=threshold)
    ## polarity calculations

    sentiment_objects = [TextBlob(comment.body) for comment in submission.comments.list()]
    sentiment_values = [[comment.sentiment.polarity, comment.sentiment.subjectivity, str(comment)] for comment in sentiment_objects]
    sentiment_df = pd.DataFrame(sentiment_values, columns=["Polarity", "Subjectivity", "Comment"])
    sentiment_df = sentiment_df.round(2)
    #upvotes for comments
    upvotes = [comment.score for comment in submission.comments.list()]
    sentiment_df['Upvotes'] = upvotes

    return word_count_df(submission).to_json(orient='split'), sentiment_df.to_json(orient='split'),


# number of comments, average senitment value, average upvotes,
@app.callback(
    Output('post_statistics', 'children'),
    [Input('polarity_value', 'data')]
)

def post_stats(jsonified_cleaned_data):
    df = pd.read_json(jsonified_cleaned_data, orient='split')
    number_of_comments = df.shape[0]
    non_zero_comments = df[df['Polarity'] != 0]
    av_polarity_per_comment = round(non_zero_comments.Polarity.mean(), 2)
    av_subjectivity_per_comment = round(df.Subjectivity.mean(), 2)
    av_upvotes_per_comment = round(df.Upvotes.mean(), 2)

    return html.Div(
        className='placeholder',
        children=[
            html.P('Number of comments: {}' .format(number_of_comments), className='post-statistics-stats'),
            html.P('Average polarity per non-zero polarity comments: {}' .format(av_polarity_per_comment), className='post-statistics-stats'),
            html.P('Average subjectivity per comment: {}' .format(av_subjectivity_per_comment), className='post-statistics-stats'),
            html.P('Average number of upvotes per comment: {}' .format(av_upvotes_per_comment), className='post-statistics-stats')
        ])


#### tab control
@app.callback(
    Output('tab_output', 'children'),
    [Input('sentiment-topic-tabs', 'value'),
    Input('polarity_value', 'data'),
    Input('word_count_value', 'data'),
    Input('threshold', 'value'),
    Input('post_url', 'value')
    ])
def render_page(tab_value, polarity_cleaned_data, wordcount_cleaned_data, threshold, post_url):
    if tab_value == 'sentiment_tab':
        df = pd.read_json(polarity_cleaned_data, orient='split')
        no_zeroes_df = df[df['Polarity'] != 0]

        #polarity hisotgram
        fig1 = go.Figure(data=go.Histogram(x=no_zeroes_df['Polarity'], marker=dict(color='#ffb300')))
        fig1.update_layout(
            title='Polarity Histogram (non-zero)',
            font=dict(
                color='#ff9100',
            ),
            xaxis=dict(
                title='Polarity'
            ),
            yaxis=dict(
                title='Count',
                gridcolor='darkgray'
            ),
            paper_bgcolor='#212121',
            plot_bgcolor='#212121',
        ),
        fig1.update_traces(opacity=.75),

        fig2 = go.Figure(data=go.Scatter(x=df['Polarity'], y=df['Upvotes'], mode='markers', marker=dict(color='#FF5700')))
        fig2.update_layout(
            title='Upvotes vs Polarity',
            font=dict(
                color='#FF5700'
                ),
            xaxis=dict(
                title='Polarity',
                gridcolor='darkgray'
                ),
            yaxis=dict(
                title='Upvotes',
                gridcolor='darkgray'
                ),
            plot_bgcolor='#212121',
            paper_bgcolor='#212121'
        )
        return html.Div([
            dcc.Graph(figure=fig1, className='polarity-graph'),
            dcc.Graph(figure=fig2, className='polarity-upvotes-graph')]
        )
    elif tab_value =='topic_tab':
        submission = reddit.submission(url=post_url)
        submission.comments.replace_more(limit=threshold)
        ds = LDA_df(submission)

        fig3a = go.Figure(go.Sunburst(
        labels=['Topic 1', 'Topic 2', 'Topic 3', ds.Topics[0][0][0], ds.Topics[0][0][1], ds.Topics[0][0][2],ds.Topics[0][0][3], ds.Topics[1][0][0], ds.Topics[1][0][1], ds.Topics[1][0][2],ds.Topics[1][0][3], ds.Topics[2][0][0], ds.Topics[2][0][1], ds.Topics[2][0][2], ds.Topics[2][0][3]],
        parents=['', '', '', 'Topic 1', 'Topic 1', 'Topic 1', 'Topic 1', 'Topic 2', 'Topic 2', 'Topic 2', 'Topic 2', 'Topic 3', 'Topic 3', 'Topic 3', 'Topic 3',  ]
        ))
        fig3a.update_layout(
            margin = dict(t=0, l=0, r=0, b=0),
            plot_bgcolor='#212121',
            paper_bgcolor='#212121'
        ),
        list_of_topics = []
        list_of_weights = []
        for i in range(0,3):
            for j in range(0,4):
                list_of_topics.append(ds.Topics[i][0][j])
                list_of_weights.append(ds.Weight[i][0][j])
        fig3b = go.Figure(go.Bar(x=list_of_topics, y=list_of_weights, marker=dict(color='#ffb300')))
        fig3b.update_layout(
            title='Weight vs Subtopics',
            plot_bgcolor='#212121',
            paper_bgcolor='#212121',
            xaxis=dict(
                title='Subtopics'
            ),
            yaxis=dict(
                title='Weight'
            ),
            font=dict(
                color='#ffb300'
                ),
        )

        return html.Div(
            className='topic-modeling-container',
            children=[
                dcc.Graph(figure=fig3a, className='word-frequency'),
                dcc.Graph(figure=fig3b, className='word-frequency')
            ])

    elif tab_value == 'other':
        de = pd.read_json(wordcount_cleaned_data, orient='split')
        fig4 = go.Figure(go.Bar(x=de.Count.iloc[:8], y=de.Word.iloc[:8], orientation='h', marker=dict(color='#FF5700')))
        fig4.update_layout(
            title='Most Common Words',
            xaxis=dict(
                title='Count',
                gridcolor='darkgray'
                ),
            yaxis=dict(
                title='Word'
                ),
            font=dict(
                color='#FF5700'
                ),
            plot_bgcolor='#212121',
            paper_bgcolor='#212121'
        )

        fig5 = go.Figure(go.Bar(x=de['Count (Bigram)'].iloc[:8], y=de.Bigram.iloc[:8], orientation='h', marker=dict(color='#ffb300')))
        fig5.update_layout(
            title='Bigram Frequency',
            xaxis=dict(
                title='Count',
                gridcolor='darkgray'
                ),
            yaxis=dict(
                title='Bigram'
                ),
            font=dict(
                color='#ffb300'
                ),
            plot_bgcolor='#212121',
            paper_bgcolor='#212121'
        )

        fig5.update_traces(opacity=.75)
        return html.Div(
            className='word-frequency-container',
            children=[
                dcc.Graph(figure=fig4, className='word-frequency'),
                dcc.Graph(figure=fig5, className='bigram-frequency')
            ]
        )
    else:
        html.Div(html.P('ERROR!!!'))

# LSI
@app.callback(
    Output('LSI_output', 'children'),
    [Input('polarity_value', 'data'),
    Input('LSI_input', 'value')
    ])

def LSI(polarity_cleaned_data, LSI_input):
    df = pd.read_json(polarity_cleaned_data, orient='split')
    text_corpus = df['Comment']

    processed_corpus = preprocess_documents(text_corpus)
    dictionary = gensim.corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    lsi = gensim.models.LsiModel(bow_corpus, num_topics=200)
    index = gensim.similarities.MatrixSimilarity(lsi[bow_corpus])

    new_doc = gensim.parsing.preprocessing.preprocess_string(LSI_input)
    new_vec = dictionary.doc2bow(new_doc)
    vec_bow_tfidf = lsi[new_vec]

    sims = index[vec_bow_tfidf]

    comment_list=[]
    cosine_similarity=[]
    comment_polarity=[]
    comment_subjectivity=[]
    comment_upvotes=[]
    for s in sorted(enumerate(sims), key=lambda item: -item[1])[:10]:
        comment_list.append(f"{df['Comment'].iloc[s[0]]}")
        cosine_similarity.append(s[1])
        comment_polarity.append(df['Polarity'].iloc[s[0]])
        comment_subjectivity.append(df['Subjectivity'].iloc[s[0]])
        comment_upvotes.append(df['Upvotes'].iloc[s[0]])

    d = {'Cosine Similarity':cosine_similarity, 'Comments':comment_list, 'Polarity':comment_polarity, 'Subjectivity': comment_subjectivity, 'Upvotes':comment_upvotes}
    LSI_df = pd.DataFrame(d)

## averages for top 10 comment results
    columns = ['Polarity', 'Subjectivity', 'Cosine Similarity']
    averages = [round(LSI_df['Polarity'].mean(),2), round(LSI_df['Subjectivity'].mean(),2), round(LSI_df['Cosine Similarity'].mean(), 2) ]

    fig5 = go.Figure(
                data=[
                    go.Bar(x=columns, y=averages, marker=dict(color='#ffb300'))
                  ])
    fig5.update_layout(
        font=dict(
            color='#ff9100'
        ),
        title='Statistical Averages for Search Results',
        xaxis=dict(
            title='Comments (from highest cosine similarity to lowest)',
        ),
        yaxis=dict(
            title='Polarity, Subjectivity and Cosine Similarity Averages',
            gridcolor='darkgray'
        ),
        plot_bgcolor='#212121',
        paper_bgcolor='#212121'

    ),
    fig5.update_traces(opacity=.75)

    return html.Div(
            children=[
                html.Div(
                    children=[
                    dash_table.DataTable(
                        columns=[{'name': i, 'id': i} for i in LSI_df.columns],style_table={'overflow': 'auto'},
                        data=LSI_df.to_dict('records'), style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'font-family':'Helvetica', 'font-weight':'ligher',
                        'height': 'auto','backgroundColor':'#1a1a1a', 'color':'darkgray' }, style_header = {'font-weight':'bold', },
                        css=[{'selector': '.dash-spreadsheet td div',
                            'rule': '''
                                line-height: 15px;
                                max-height: 30px; min-height: 30px; height: 30px;
                                display: block;
                                overflow-y: hidden;
                            '''
                            }],tooltip_duration=None,
                                tooltip_data=[{
                                    column: {'value': str(value), 'type': 'markdown'}
                                    for column, value in row.items()
                                            } for row in LSI_df.to_dict('records')],
                                    )
                                ],
                    className='datatable'),
                html.Div(
                    className='LSI-bar',
                    children=[
                        dcc.Graph(figure=fig5)

                    ])

            ])


if __name__ == "__main__":
    app.run_server(debug=True)






#
