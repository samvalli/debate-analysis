from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import tokenize
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from nltk import tokenize


def tex_blob_sentiment(merged_data,sentence_split=False):
    items_subjectivity=[]
    items=merged_data['item'].tolist()
    
    for item in items:
        textblob = TextBlob(item)
        polarity=0.0
        subjectivity=0.0
        if sentence_split==True:
            sentences=textblob.sentences
            for sent in sentences:
                if len(sent)>5:
                    pred = sent.sentiment
                    polarity+=pred[0]
                    subjectivity+=pred[1]
                else: 
                    continue
            items_subjectivity.append(round(subjectivity/len(sentences),3))  
        
        else:
            if len(item)>5:
                pred = textblob.sentiment
                polarity=pred[0]
                subjectivity=pred[1]
                items_subjectivity.append(subjectivity)
            else:
                items_subjectivity.append(0.0)

    return items_subjectivity 

def VADER_sentiment(merged_data,sentence_split=False):
    compound=[]
    items=merged_data['item'].tolist()

    for item in items:
        analyzer = SentimentIntensityAnalyzer()
        comp=0.0

        if sentence_split==True:
            sentences = tokenize.sent_tokenize(item)
            for sent in sentences:
                if len(sent)>5:
                    scores = analyzer.polarity_scores(sent)
                    comp += scores['compound']
                else:
                    continue
            compound.append(comp/len(sentences))
        
        else:
            if len(item)>5:
                scores=analyzer.polarity_scores(item)

                comp = scores['compound']
                compound.append(comp)
            else:
                print(f"short item found: {item}")
                compound.append(0.0)
                
    return compound
