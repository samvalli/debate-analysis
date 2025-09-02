from lexical_diversity import lex_div as ld
import collections
import re
from nltk.corpus import stopwords
import html
from nltk.stem import PorterStemmer
from nltk import tokenize
from textstat import textstat

def readability_score(data):
    readability_score=[]
    for item in data['item'].tolist():
        score=textstat.flesch_reading_ease(item)
        readability_score.append(score)
    return readability_score

def tokenize_and_preprocess(text):
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = re.split(r"[^0-9A-Za-z\-'_]+", text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [word for word in tokens if not word.isdigit()]
    return tokens

def get_yules(s):
    """ 
    Returns a tuple with Yule's K and Yule's I.
    (cf. Oakes, M.P. 1998. Statistics for Corpus Linguistics.
    International Journal of Applied Linguistics, Vol 10 Issue 2)
    In production this needs exception handling.
    """
    tokens = tokenize_and_preprocess(s)
    token_counter = collections.Counter(tok.upper() for tok in tokens)
    m1 = sum(token_counter.values())
    m2 = sum([freq ** 2 for freq in token_counter.values()])
    i = (m1*m1) / (m2-m1)
    k = 1/i * 10000
    return k, i, len(tokens)


def get_yules_k_paper(s):
    """ 
    Calculates Yule's K as defined in the paper.
    K = 10^4 * [ -1/N + sum(V(i, N) * (i/N)^2) ]
    where V(i, N) is the number of words appearing exactly i times.
    """
    tokens = tokenize(s)
    N = len(tokens)  # Total number of tokens
    if N == 0:
        raise ValueError("The input text must contain at least one token.")

    # Count frequencies of each word
    token_counter = collections.Counter(tok.upper() for tok in tokens)

    # Build V(i, N): a frequency distribution of word occurrences
    freq_counter = collections.Counter(token_counter.values())

    # Calculate Yule's K
    sum_component = sum(freq_counter[i] * (i / N) ** 2 for i in freq_counter)
    K = 10**4 * (-1 / N + sum_component)

    return K



def mltd(merged_data):
    items=merged_data['item'].tolist()
    MLTD=[]
    for item in items:
        tok = ld.tokenize(item)
        MLTD.append(ld.mtld(tok))
    return MLTD

#UN idea potrebbe essere quella di sommare tutti le frasi riguardanti un certo topic e calcolare la yule di ciascuna pagina invece che di ciascun item
def get_yule_k_info(merged_data,platforms):
    yule_dict={'wiki':[],'kialo':[],'cmv':[]}
    page_yule=[]
    for platform in platforms:
        print("============= new platform ===========")
        data = merged_data[merged_data['platform']==platform]
        for page_id in merged_data['page_id'].unique().tolist():
            text=''
            page_data = data[data['page_id']==page_id]
            for item in page_data['item'].tolist():
                text+=item+'\n'
            yule_k,yule_i,page_words=get_yules(text)
            #yule_k_paper=get_yules_k_paper(text)
            print(f"page: {page_id}")
            print(page_words)
            print(yule_k)
            num_items=len(page_data)
            words_per_item=page_words/num_items
            yule_dict[platform].append([page_id,yule_k,yule_i,page_words,num_items,words_per_item])

    k_yule=[]
    i_yule=[]
    page_words=[]
    page_items=[]
    words_per_item=[]
    for i,row in merged_data.iterrows():
        platform = row['platform']
        platform_dict=yule_dict[platform]
        for elem in platform_dict:
            if elem[0]==row['page_id']:
                k_yule.append(elem[1])
                i_yule.append(elem[2])
                page_words.append(elem[3])
                page_items.append(elem[4])
                words_per_item.append(elem[5])
    
    return k_yule,i_yule,page_words,page_items,words_per_item