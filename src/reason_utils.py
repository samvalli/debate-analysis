import numpy as np
import pandas as pd

def binarize_knowledge(data,trehsold):
    
    binazrized=[]
    for i,row in data.iterrows():
        if row['knowledge']>trehsold:
            binazrized.append(1)
        else:
            binazrized.append(0)

    data = data.assign(binarized_know = binazrized,debate_id =data['debate_id'].tolist())
    return data


def assign_reason_score(data,platforms,trehsold=0.95):
    dimension_df = binarize_knowledge(data,trehsold)
    debate_wise_knowledge_binary_percentage=[]

    for platform in platforms:
        plat_data = dimension_df[dimension_df['platform']==platform]
        for page_id in dimension_df['page_id'].unique():
            pg_data = plat_data[plat_data['page_id']==page_id]
            knowledge_debates=[]
            for debate_id in pg_data['debate_id'].unique():
                debate_data = pg_data[pg_data['debate_id']==debate_id]
                knowledge_debates.append(debate_data['binarized_know'].sum()/len(debate_data))
            
            debate_wise_knowledge_binary_percentage+=knowledge_debates
    
    return debate_wise_knowledge_binary_percentage