import streamlit as st
from tqdm import tqdm
import pandas as pd
import os
from textblob import Word
import nltk
from nltk.corpus import stopwords
import joblib


working_dir=os.getcwd()

@st.cache_resource(show_spinner=True)
def training():
    return joblib.load("./data/Models/Hier/bow_tfidf.joblib")

def convert_df(df):
    return df.to_csv().encode('utf-8')

def savefile(path,uploaded_file):
    save_path = f"{working_dir}/{path}"
    with open(os.path.join(save_path, uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer())
    return f"{working_dir}/{path}{str(uploaded_file.name)}"

def single_complaint_processing(df):
    df['consumer_complaint_narrative'] =df['consumer_complaint_narrative'].apply(lambda x: ' '.join([i.lower() for i in str(x).split()]))
    df['consumer_complaint_narrative'] =df['consumer_complaint_narrative'].str.replace(r'[^\w\s]',"")
    stop = stopwords.words('english')
    df['consumer_complaint_narrative'] =df['consumer_complaint_narrative'].apply(lambda x: ' '.join([i for i in str(x).split() if i not in stop]))
    df['consumer_complaint_narrative'] =df['consumer_complaint_narrative'].apply(lambda x:' '.join([Word(i).lemmatize() for i in str(x).split()]))
    return df

def testing_df_processing(TEST_DF):
    
    TEST_DF.rename(columns = {'Consumer complaint narrative':'consumer_complaint_narrative'}, inplace = True)
    TEST_DF.rename(columns = {'Issue_sub_group':'Issue Sub Group'}, inplace = True)
    TEST_DF['consumer_complaint_narrative'] =TEST_DF['consumer_complaint_narrative'].apply(lambda x: ' '.join([i.lower() for i in str(x).split()]))
    TEST_DF['consumer_complaint_narrative'] =TEST_DF['consumer_complaint_narrative'].str.replace(r'[^\w\s]',"")
    stop = stopwords.words('english')
    TEST_DF['consumer_complaint_narrative'] =TEST_DF['consumer_complaint_narrative'].apply(lambda x: ' '.join([i for i in str(x).split() if i not in stop]))
    TEST_DF['consumer_complaint_narrative'] =TEST_DF['consumer_complaint_narrative'].apply(lambda x:' '.join([Word(i).lemmatize() for i in str(x).split()]))
    return TEST_DF
   

def app():    

    st.header("Slot Classifier")
    processing_option=st.selectbox("Choose:",options=["Single Complaint","Dataframe"],key=5)


    if processing_option=="Single Complaint":
        DEFAULT_TEXT="""I took out a personal loan from Wells Fargo, first payments stated XXXX. XXXX is the amount. They wanted me to set this up on auto payments. I would go in every month and put another payment down on top of this minimum auto payment. Problems so far... It started to seem that my min payment was not being made. Every time I went in I made sure it was all set up for auto payment and that they were being made. The tellers EVERYTIME would tell me yes, they are all set up and working. Finally a teller suggested I make the second payment on top of the min payment " principal only '' payments, that would help me pay off loan faster. Started to do that. WF then said on XXXX XXXX ( when I started to do it this way ) they said after careful review, we can confirm the principal only payment option was removed from all channels??? THEN... a few months later, another teller was trying to help me, she pointed out that you can ONLY make overpayments the ONLY DAY THE AUTO PAYMENT GOES THROUGH! So In good faith, I started to put XXXX on top of the minimum payment of XXXX. In all I was giving WF XXXX a month... HERE IS THE A BIG PROBLEM... since I was putting down more than minimum, they were DIFFERING the auto payment to the next month. So not acknowledging the min payment. Then taking interest out of the extra payment amount!! OVERALL... by not acknowledging the min payment that added up to be and was XXXX NOT PUT TOWARDS MY LOAN per year!! By rights I should have this loan almost paid off by now!! I just noticed they get XXXX PER YEAR INTEREST FROM ME. Now I know that banks are in business to make money. But this??? I am in good faith trying to pay off my loan as fast as I can. That only seems logical?? So far, I have a case number with WFXXXX. Now I am sure with all their lawyers and being a huge company, they are all in the legal here... BUT the way they are going around and setting up all these road blocks trying to slow me down in paying off the loan is pretty despicable????? Hopefully by issuing a complaint others will see and be alerted to these shenanigans being pulled by WF. Thank you for taking time out on this matter."""
        input_issue=st.text_area("Enter your Query",DEFAULT_TEXT,height=300)
        temp_list = [input_issue]
        temp_df=pd.DataFrame(temp_list,columns=["consumer_complaint_narrative"])

        if st.button("Submit",key=7):
            single_run(temp_df)
    if processing_option=="Dataframe":
        bow_tfidf_logit_pipeline=training()
        df=st.file_uploader("Enter your Test Dataframe",type=[".xlsx"])
        if st.button("Submit") and df is not None:
            dataframe_run(df, bow_tfidf_logit_pipeline)

def dataframe_run(df, bow_tfidf_logit_pipeline):
    filename=savefile(path="data/Complaint/",uploaded_file=df)
    user_df=pd.read_excel(filename)
    proccesed_user_df=testing_df_processing(user_df)
    test_pred_level_2 = bow_tfidf_logit_pipeline.predict(proccesed_user_df['consumer_complaint_narrative'])
    predicted_issue = []
    predicted_issue_group=[]
    for ele in test_pred_level_2:
        l = ele.split('/')
        predicted_issue.append(l[0])
        predicted_issue_group.append(l[1])

    proccesed_user_df['Predicted Issue']=predicted_issue
    proccesed_user_df['Predicted Issue Sub Group']=predicted_issue_group
    proccesed_user_df.rename(columns = {'consumer_complaint_narrative':'Consumer Complaint Narrative'}, inplace = True)
    proccesed_user_df.rename(columns = {'Issue':'Actual Issue'}, inplace = True)
    proccesed_user_df.rename(columns = {'Issue Sub Group':'Actual Issue Sub Group'}, inplace = True)


    proccesed_user_df_final=proccesed_user_df[["Consumer Complaint Narrative","Predicted Issue","Predicted Issue Sub Group","Actual Issue","Actual Issue Sub Group"]]
    st.dataframe(proccesed_user_df_final)
    proccesed_user_df_final=convert_df(proccesed_user_df_final)
    st.download_button("Download",data=proccesed_user_df_final,file_name="test_results.csv")

def single_run(temp_df):
    training()
    bow_tfidf_logit_pipeline=training()
    temp_df=single_complaint_processing(temp_df)
    prediction=bow_tfidf_logit_pipeline.predict(temp_df["consumer_complaint_narrative"])
    predicted_issue_single = []
    predicted_issue_group_single=[]
    for ele in prediction:
        p = ele.split('/')
        predicted_issue_single.append(p[0])
        predicted_issue_group_single.append(p[1])

        st.header("Predicted issue:")
        st.subheader(predicted_issue_single[0])
        st.header("Predicted issue sub-group:")
        st.subheader(predicted_issue_group_single[0])
app()