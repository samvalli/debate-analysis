from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import tokenize
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from nltk import tokenize


def tex_blob_sentiment(items,platform,sentence_split=False):
    items_polarity=[]
    items_subjectivity=[]
    platform=[]
    for item in items:
        platform.append(platform)
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
            items_polarity.append(round(polarity/len(sentences),3))
            items_subjectivity.append(round(subjectivity/len(sentences),3))  
        
        else:
            if len(item)>5:
                pred = textblob.sentiment
                polarity=pred[0]
                subjectivity=pred[1]
                items_polarity.append(polarity)
                items_subjectivity.append(subjectivity)
            else:
                items_polarity.append(0.0)
                items_subjectivity.append(0.0)

    return items_polarity,items_subjectivity 

def VADER_sentiment(items,sentence_split=False):
    negative=[]
    neutral=[]
    positive=[]
    compound=[]

    for item in items:
        analyzer = SentimentIntensityAnalyzer()
        
        neg=0.0
        neu=0.0
        pos=0.0
        comp=0.0

        if sentence_split==True:
            sentences = tokenize.sent_tokenize(item)
            for sent in sentences:
                if len(sent)>5:
                    scores = analyzer.polarity_scores(sent)
                    neg +=scores['neg']
                    neu +=scores['neu']
                    pos +=scores['pos']
                    comp += scores['compound']
                else:
                    continue
            negative.append(neg/len(sentences))
            neutral.append(neu/len(sentences))
            positive.append(pos/len(sentences))
            compound.append(comp/len(sentences))
        
        else:
            if len(item)>5:
                scores=analyzer.polarity_scores(item)
                neg =scores['neg']
                neu =scores['neu']
                pos =scores['pos']
                comp = scores['compound']

                negative.append(neg)
                neutral.append(neu)
                positive.append(pos)
                compound.append(comp)
            else:
                print(f"short item found: {item}")
                negative.append(0.0)
                positive.append(0.0)
                neutral.append(1.0)
                compound.append(0.0)
                
    return positive,negative,neutral,compound
