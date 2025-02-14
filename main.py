import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
import spacy
import networkx as nx
import io
from langdetect import detect
from googletrans import Translator
import numpy as np

# Install NLTK data if not downloaded
nltk.download('punkt')

# Load spaCy model (lazily to prevent deployment issues)
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

st.title("📰 NewsIntelGraph: AI-Powered Insight Analyzer")

# User input for topic
topic = st.text_input("Enter a news topic:")

if topic:
    # Scrape news from Bing
    search_url = f'https://www.bing.com/news/search?q={topic.replace(" ", "+")}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    articles = soup.find_all('a', class_='title')[:5]

    if not articles:
        st.error("No news articles found! Try another topic.")
    else:
        st.subheader("Top Articles:")

        full_text = ""
        article_sentiments = []
        entity_freq = {}

        nlp = load_spacy()

        for idx, article in enumerate(articles):
            title = article.text
            link = article['href']
            st.markdown(f"[{idx + 1}. {title}]({link})")
            full_text += title + " "

            # Sentiment Analysis
            blob = TextBlob(title)
            sentiment_score = blob.sentiment.polarity
            article_sentiments.append(sentiment_score)

            # Named Entity Recognition
            doc = nlp(title)
            for ent in doc.ents:
                entity_freq[ent.text] = entity_freq.get(ent.text, 0) + 1

        # Sentiment Analysis Summary
        sentiment_score = np.mean(article_sentiments)
        st.subheader("Sentiment Analysis:")
        if sentiment_score > 0:
            st.success("Overall Sentiment: Positive 😊")
        elif sentiment_score < 0:
            st.error("Overall Sentiment: Negative 😞")
        else:
            st.warning("Overall Sentiment: Neutral 😐")

        # Sentiment Distribution
        st.subheader("Sentiment Score Distribution:")
        fig, ax = plt.subplots()
        ax.hist(article_sentiments, bins=10, color='skyblue', edgecolor='black')
        ax.set_xlabel("Sentiment Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Sentiment Distribution of News Articles")
        st.pyplot(fig)

        # Word Cloud
        if full_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(full_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.subheader("Word Cloud of News Headlines:")
            st.pyplot(fig)

        # Summarization
        if len(full_text.split()) > 10:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summary = summarizer(full_text[:1024], max_length=50, min_length=25, do_sample=False)[0]['summary_text']
            st.subheader("AI Summary of Topic:")
            st.write(summary)

        # Language Detection and Translation
        lang = detect(full_text)
        st.subheader("Language Detection:")
        st.write(f"Detected Language: {lang}")

        if lang != 'en':
            translator = Translator()
            translated_text = translator.translate(full_text, src=lang, dest='en').text
            st.subheader("Translated News Text:")
            st.write(translated_text)

        # Knowledge Graph Generation
        st.subheader("Knowledge Graph of Named Entities:")
        graph = nx.DiGraph()
        for entity in entity_freq.keys():
            graph.add_edge(topic, entity)

        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10, font_weight='bold')
        st.pyplot(plt)

        # Trend Analysis (Top Entities)
        sorted_entities = sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        st.subheader("Top 10 Most Frequent Entities:")
        for entity, freq in sorted_entities:
            st.write(f"{entity}: {freq}")

        # Named Entity Frequency Distribution
        st.subheader("Named Entity Frequency Distribution:")
        if entity_freq:
            entity_names = list(entity_freq.keys())
            entity_counts = list(entity_freq.values())

            fig, ax = plt.subplots()
            ax.barh(entity_names, entity_counts, color='lightgreen')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Named Entities')
            ax.set_title('Frequency Distribution of Named Entities')
            st.pyplot(fig)

        # Trending Keywords
        if entity_freq:
            keyword_cloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(entity_freq)
            fig, ax = plt.subplots()
            ax.imshow(keyword_cloud, interpolation='bilinear')
            ax.axis("off")
            st.subheader("Trending Keywords in News:")
            st.pyplot(fig)
