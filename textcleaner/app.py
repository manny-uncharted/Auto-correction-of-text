# Core Packages
import streamlit as st

# NLP
import neattext.functions as nfx

# EDA
import pandas as pd

# Text Downloader
import base64
import time

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
from wordcloud import WordCloud

# NLP Packages
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
matplotlib.use("Agg")

timestr = time.strftime("%Y%m%d-%H%M%S")


def get_most_common_tokens(docx,num=10):
    word_freq = Counter(docx.split())
    most_common_tokens = word_freq.most_common(num)
    return dict(most_common_tokens)

def plot_wordcloud(text):
    wordcloud = WordCloud().generate(text)
    fig = plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig)

def text_analyzer(my_text):
    docx = nlp(my_text)
    allData = [(token.text, token.shape_, token.pos_, token.tag_, token.dep_, token.lemma_, token.is_alpha, token.is_stop) for token in docx]
    df = pd.DataFrame(allData, columns=["Text", "Shape", "POS", "Tag", "Dep", "Lemma", "Alpha", "Stopword"])
    return df

def get_entities(my_text):
    docx = nlp(my_text)
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    return entities


HTML_WRAPPER = """ <div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem"> {} </div> """


def render_entities(raw_text):
    docx = nlp(raw_text)
    html = displacy.render(docx, style="ent")
    html = html.replace("\n\n", "\n")
    result = HTML_WRAPPER.format(html)
    return result

def text_downloader(raw_text):
    b64 = base64.b64encode(raw_text.encode()).decode()
    new_filename = "cleaned_text_result_{}_.txt".format(timestr)
    st.markdown("#### Download File ####")
    href = f'<a href="data:file/txt:base64,{b64}" download="{new_filename}">Click here!!</a>'
    st.markdown(href, unsafe_allow_html=True)

def make_downloadable(data):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    new_filename = "nlp_result_{}_.csv".format(timestr)
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download csv file</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    st.title("Text Correction App")

    menu = ["TextCleaner", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "TextCleaner":
        st.subheader("Text Cleaning")

        text_file = st.file_uploader("Upload Txt file", type=['txt']) # Textfile upload

        # The cases to consider to format the text.
        normalize_case = st.sidebar.checkbox("Normalize Case")
        clean_stopwords = st.sidebar.checkbox("Stopwords")
        clean_punctuations = st.sidebar.checkbox("Punctuations")
        clean_emails = st.sidebar.checkbox("Emails")
        clean_specials = st.sidebar.checkbox("Special Characters")
        clean_numbers = st.sidebar.checkbox("Numbers")
        clean_urls = st.sidebar.checkbox("URLS/Links")
        clean_hashtags = st.sidebar.checkbox("Hashtags")
        clean_html_tags = st.sidebar.checkbox("Html tags")
        clean_emojis = st.sidebar.checkbox("Emojis")

        if text_file is not None:

            # Show the file details
            file_details = {"Filename":text_file.name, "Filesize":text_file.size, "Filetype":text_file.type}
            st.write(file_details)

            # Columns for display text
            col1, col2 = st.columns(2)
            # Decode Text
            raw_text = text_file.read().decode('utf-8')
            
            with col1:
                with st.expander("Original Text"):
                    st.write(raw_text)

            with col2:
                with st.expander("Processed Text"):
                    if normalize_case:
                       raw_text = raw_text.lower()

                    if clean_stopwords:
                        raw_text = nfx.remove_stopwords(raw_text)

                    if clean_emails:
                        raw_text = nfx.remove_emails(raw_text)

                    if clean_hashtags:
                        raw_text = nfx.remove_hashtags(raw_text)

                    if clean_html_tags:
                        raw_text = nfx.remove_html_tags(raw_text)
                    
                    if clean_emojis:
                        raw_text = nfx.remove_emojis(raw_text)

                    if clean_punctuations:
                        raw_text = nfx.remove_punctuations(raw_text)

                    if clean_specials:
                        raw_text = nfx.remove_special_characters(raw_text)
                    
                    if clean_numbers:
                        raw_text = nfx.remove_numbers(raw_text)

                    if clean_urls:
                        raw_text = nfx.remove_urls(raw_text)

                    st.write(raw_text)
                    text_downloader(raw_text)
                    # st.button("Download", text_downloader(raw_text))

            # Show the text analysis
            with st.expander("Text Analysis"):
                token_result_df = text_analyzer(raw_text)
                st.dataframe(token_result_df)
                make_downloadable(token_result_df)

            with st.expander("Plot Wordcloud"):
                plot_wordcloud(raw_text)

            with st.expander("Plot POS Tags"):
                fig = plt.figure()
                token_result_df['POS'] = token_result_df['POS'].astype('category')
                plots = (token_result_df['POS'].value_counts())
                sns.barplot(x=plots.index, y=plots.values, palette='Set2')
                plt.xticks(rotation=45)
                st.pyplot(fig)
    else:
        st.subheader("About")



if __name__ == '__main__':
    main()