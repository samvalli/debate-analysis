from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import tokenize
import pandas as pd


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from nltk import tokenize

def VADER_sentiment(data,sentence_split=False):
    negative=[]
    neutral=[]
    positive=[]
    compound=[]
    sentence_split=False
    analyzer = SentimentIntensityAnalyzer()
    items=data['item'].tolist()
    for item in items:
        
        
        neg=0.0
        neu=0.0
        pos=0.0
        comp=0.0
        if sentence_split==True:
            sentences = tokenize.sent_tokenize(item)
            for sent in sentences:
                scores = analyzer.polarity_scores(sent)
                neg +=scores['neg']
                neu +=scores['neu']
                pos +=scores['pos']
                comp += scores['compound']

            negative.append(neg/len(sentences))
            neutral.append(neu/len(sentences))
            positive.append(pos/len(sentences))
            compound.append(comp/len(sentences))
        else:
            scores=analyzer.polarity_scores(item)
            neg =scores['neg']
            neu =scores['neu']
            pos =scores['pos']
            comp = scores['compound']

            negative.append(neg)
            neutral.append(neu)
            positive.append(pos)
            compound.append(comp)

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
    sentence_split=False
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


def merge_datasets(wiki_data,kialo_data,cmv_data):
    wiki_data=wiki_data[['id','page_id','item','parent_id','quantile','title','length','level','z_scores_page_it','z_scores_page_mod','num_child','thread_id','categories']]
    cmv_pages=cmv_data['page_id'].unique().tolist()
    wiki_data=wiki_data[wiki_data['page_id'].isin(cmv_pages)]
    kialo_data=kialo_data[kialo_data['page_id'].isin(cmv_pages)]
    kialo_data=kialo_data[['page_id','title','id','item','length','level']]
    cmv_data=cmv_data[['page_id','title','id','item','length','level']]

    data_frames=[wiki_data,kialo_data,cmv_data]
    platforms=['wiki','kialo','cmv']

    blob_pol=[]
    blob_subj=[]
    vader_pos=[]
    vader_neg=[]
    vader_neu=[]
    vader_comp=[]
    for i,data in enumerate(data_frames):
        platform = platforms[i]
        print(f"New platform: {platform}")
        items = data['item'].tolist()
        items_polarity,items_subjectivity= tex_blob_sentiment(items,platform,False)
        positive,negative,neutral,compound = VADER_sentiment(items,False)

        blob_pol+=items_polarity
        blob_subj+=items_subjectivity
        vader_pos+=positive
        vader_neg+=negative
        vader_neu+=neutral
        vader_comp+=compound
    
    platform_list = ['wiki']*len(wiki_data)+['kialo']*len(kialo_data)+['cmv']*len(cmv_data)

    merged_data = pd.concat([wiki_data,kialo_data,cmv_data],ignore_index=True)
    merged_data=merged_data.assign(blob_pol=blob_pol,blob_subj=blob_subj,vader_pos=vader_pos,vader_neg=vader_neg,vader_neu=vader_neu,vader_comp=vader_comp,platform=platform_list)
    merged_data.to_csv('data/global/wiki_kialo_cmv_data.csv')
