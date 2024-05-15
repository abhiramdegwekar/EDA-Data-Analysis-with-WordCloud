import streamlit as st
from tqdm import tqdm
import pandas as pd
import os
from textblob import Word
from nltk.corpus import stopwords
from stqdm import stqdm
working_dir=os.getcwd()


@st.cache_resource(show_spinner=True)
def load_model():
    import torch
    from transformers import T5ForConditionalGeneration, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('./data/Models/Precis/')
    model = T5ForConditionalGeneration.from_pretrained('./data/Models/Precis/')
    device = torch.device('cpu')
    return model,tokenizer,device

def savefile(path,uploaded_file):
    save_path = f"{working_dir}/{path}"
    with open(os.path.join(save_path, uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer())
        filename = f"{working_dir}/{path}{str(uploaded_file.name)}"
    return filename

def generate_df(MAIN_DF):
    MAIN_DF.rename(columns = {'Consumer complaint narrative':'consumer_complaint_narrative'}, inplace = True)
    data1=MAIN_DF['consumer_complaint_narrative']
    data1 =data1.apply(lambda x: ' '.join([i.lower() for i in str(x).split()]))
    data1 =data1.str.replace(r'[^\w\s]',"")
    stop = stopwords.words('english')
    data1 =data1.apply(lambda x: ' '.join([i for i in str(x).split() if i not in stop]))
    data1 =data1.apply(lambda x:' '.join([Word(i).lemmatize() for i in str(x).split()]))
    data1 = data1.str.replace(r"xx+\s","")
    return data1
def convert_df(df):
    return df.to_csv().encode('utf-8')
def summarizer(text,model,tokenizer,device):
  t5_prepared_Text = "summarize: "+ text

  tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
  summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=70,
                                    early_stopping=True)
  return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def app():  # sourcery skip: extract-method, merge-list-append
    st.header("Precis Creator")
    functions=st.selectbox("Choose:",options=["Single Complaint","Dataframe"])
    if functions=="Dataframe":
        data_frame_file=st.file_uploader("Enter your Excel sheet",type=["xlsx"])
        if st.button("Submit"):
            if data_frame_file is not None:
                filename=savefile(path="data/Precis/",uploaded_file=data_frame_file)      
                data = pd.ExcelFile(filename)
                dfs = {sheet_name: data.parse("Sheet1") 
                for sheet_name in tqdm(data.sheet_names)}
                data_main = dfs['Sheet1']
                model,tokenizer,device=load_model()
                data1=generate_df(MAIN_DF=data_main)

                
                summarize_lst=[]
                for row in stqdm(data1,desc="Making changes to all of the rows:"):
                    summarize = summarizer(text=row,model=model,tokenizer=tokenizer,device=device)
                    summarize_lst.append(summarize)

                data_main.rename(columns = {'consumer_complaint_narrative':'Consumer Complaint Narrative'}, inplace = True)
                data_main['Precis Complaint']=summarize_lst
                st.dataframe(data_main)
                data_main=convert_df(data_main)
                st.download_button("Download",data=data_main,file_name="precis_results.csv")
            else:
                st.subheader("Please enter a XLSX file above.")
    else:
        DEFAULT_TEXT="""in accordance with the Fair Credit Reporting act XXXX Account # XXXX, has violated my rights. 15 U.S.C 1681 section 602 A. States I have the right to privacy. 15 U.S.C 1681 Section 604 A Section 2 : It also states a consumer reporting agency can not furnish a account without my written instructions XXXX XXXX XXXX  XXXX & XXXX"""
        custom_text=st.text_area("Enter your Complaint",DEFAULT_TEXT,height=50)
        if st.button("Submit",key=2):
            model,tokenizer,device=load_model()
            custom_text_list=[]
            custom_text_list.append(custom_text)
            data=pd.DataFrame(custom_text_list,columns=["Consumer complaint narrative"])
            processed_data=generate_df(MAIN_DF=data)

            summarize_lst=[]
            for row in processed_data:
                summarize = summarizer(text=row,model=model,tokenizer=tokenizer,device=device)
                summarize_lst.append(summarize)
            
            processed_data['Summary']=summarize_lst
            st.subheader("Precis :")
            st.write(summarize_lst[0])

app()