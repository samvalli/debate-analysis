from lexical_diversity import lex_div as ld
from textstat import textstat

def mltd(merged_data):
    items=merged_data['item'].tolist()
    MLTD=[]
    for item in items:
        tok = ld.tokenize(item)
        MLTD.append(ld.mtld(tok))
    return MLTD

def readability_score(data):
    readability_score=[]
    for item in data['item'].tolist():
        score=textstat.flesch_reading_ease(item)
        readability_score.append(score)
    return readability_score

