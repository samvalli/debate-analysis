from lexical_diversity import lex_div as ld
from textstat import textstat
import pandas as pd

def get_complexity_scores(merged_data,platforms,complexity,readability):

    merged_data=merged_data.assign(complexity=complexity,readability=readability)
    complexity_score=[]
    readability_core_array = []

    for platform in platforms:
        plat_data = merged_data[merged_data['platform']==platform]
        for page_id in merged_data['page_id'].unique():
            pg_data = plat_data[plat_data['page_id']==page_id]
            for debate_id in pg_data['debate_id'].unique():
                debate_data=pg_data[pg_data['debate_id']==debate_id]
                complexity_score.append(debate_data['complexity'].mean())
                readability_core_array.append(debate_data['readability'].mean())
    
    return [x / 100 for x in complexity_score],[x / 100 for x in readability_core_array]

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

