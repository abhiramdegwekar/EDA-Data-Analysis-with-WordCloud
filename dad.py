import streamlit as st
import pandas as pd
from tqdm import tqdm
import altair as alt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
import nltk

nltk.download('punkt')
nltk.download('stopwords')
warnings.filterwarnings("ignore")

st.set_option('deprecation.showPyplotGlobalUse', False)
def convert_df(df):
    return df.to_csv().encode('utf-8')

def remove_stopwords(text):
    '''
    Returns a sentence after removing stopwords
    
    input: [str] sentence with stop words
    output: [str] sentence without stop words
    '''
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(text)

    filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]
    return ' '.join(filtered_sentence)

def clean_text_data(df):
    '''
    Returns a list of cleaned consumer complaint text
    Works only for 'Consumer complaint narrative' column
    
    input: [dataframe] dataframe containing consumer complaints
    output: [list] list of complaints after text preprocessing
    '''
    
    text_series = df['Consumer complaint narrative']


    #removing XX
    text_series = text_series.str.replace('XX',"")

    #lowercase #remove special characters
    text_series = text_series.str.replace('\d+', '').str.replace('?', '').str.replace('\W', ' ').str.lower()  #lowercase #remove punctuation

    #removing white space
    text_series = pd.Series(map(lambda x: " ".join(x.split()), text_series))

    return [remove_stopwords(text) for text in tqdm(text_series)]

def make_wordcloud(short_df,maximum):

    '''
    Returns wordcloud object 
    
    input: [dataframe] dataframe containing consumer complaints
           [int] maximum number of words to show in word cloud
    output: [object] word cloud object
    '''
    
    text = " ".join(short_df.cleaned_complaint_text)

    # Create and generate a word cloud image:
    wordcloud_object = WordCloud(max_words = maximum,background_color ='black',min_font_size = 9).generate(text)
    return wordcloud_object,text



def get_word_count(text_corpus):
    '''
    Returns dict with word count in corpus
    input: [str] corpus of words
    output: [dict] count of words
    '''
    dct = {}

    for i in text_corpus.split(' '):

        if i in dct:
            dct[i] += 1

        else:
            dct[i] = 1
    return dct

def get_count(df, column_name):
    '''
    Returns a dict with element as key and count in column as value
    
    input: dataframe, column name(from df)
    output: [dict] count of elements in column name
    '''
    lst = df[column_name]
    dct = {}

    for i in lst:
        if i in dct:
            dct[i] += 1

        else:
            dct[i] = 1
    return dct

def get_bar_plot(df,column_name):
    '''
    Returns a bar plot for a column
    
    input: dataframe, column name(from df)
    output: [fig] bar plot for the column
    '''
    print(column_name)
    count_dct = get_count(df,column_name)
    return pd.DataFrame(count_dct.items(), columns=[column_name, 'count'])


def app():
    st.header("Data Analysis Dashboard")  # sourcery skip: extract-method
    df=pd.read_csv("./data/main_sheet.csv")
    count_df=get_bar_plot(df,'Issue')
    final_df=count_df.sort_values(by="count",ascending=False)[:5]
    count_df=get_bar_plot(df,'Issue')
    bar_chart=alt.Chart(final_df).mark_bar().encode(
        y="count",
        x="Issue",
        # column= alt.Column(header=alt.Header(labelAngle=270))
    )
    st.subheader("Top 5 Issues")
    st.altair_chart(bar_chart,use_container_width=True)
    select_box=st.empty()
    remove=["Problem with a credit reporting company's investigation into an existing problem","Opening, managing, closing an account","Threats related to information sharing, taking legal action","Payment and bill payment related problems","Other features, terms, or service problem","Applying, refinancing, struggle in payment, closing a mortgage"]
    full_list=list(count_df["Issue"])
    for item in remove:
        if item in full_list:
            full_list.remove(item)

    full_list.insert(0,"Select")
    issue_name = select_box.selectbox("Choose your Issue",options=full_list)
    if st.button("Submit",key=9):
        if issue_name=="Select":
            st.subheader("You must select an Issue category!")
        else:
            short_df = df[df['Issue'] == issue_name]
            count_df_specific=get_bar_plot(short_df,"Issue_sub_group")
            bar_chart_specific=alt.Chart(count_df_specific,width=600).mark_bar().encode(
                y="count",
                x="Issue_sub_group"
            )
            
            st.subheader("Specific Sub-Issue Count")
            st.altair_chart(bar_chart_specific,use_container_width=True)         
                
            maximum = 300
            short_df['cleaned_complaint_text'] = clean_text_data(short_df)
            wordcloud_object,text = make_wordcloud(short_df,maximum)
            fig=plt.imshow(wordcloud_object, interpolation='bilinear')
            fig=plt.axis("off")
            fig=plt.show()
            st.subheader("WordCloud")
            st.pyplot()
            st.subheader("Word Count")
            final_dict=get_word_count(text_corpus=text)

            words=list(final_dict.keys())
            count=list(final_dict.values())
            word_df = pd.DataFrame(list(zip(words,count)),columns =['Words', 'Count'])
            top_words_df=word_df.sort_values(by="Count",ascending=False)[:10]
            word_bar_chart=alt.Chart(top_words_df,height=400,width=600).mark_bar().encode(
                y="Count",
                x="Words", 
            )
            st.altair_chart(word_bar_chart)

app()