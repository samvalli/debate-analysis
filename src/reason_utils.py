import numpy as np
import pandas as pd

def binarize_knowledge(data,dimension_df,trehsold):
    binazrized=[]
    for i,row in dimension_df.iterrows():
        if row['knowledge']>trehsold:
            binazrized.append(1)
        else:
            binazrized.append(0)

    dimension_df = dimension_df.assign(binarized_know = binazrized,debate_id =data['debate_id'].tolist())
    return dimension_df


def assign_reason_score(data,platforms,directory='/data/big/wikidebate/deliberative_measures/data/global/tendims_mean_sampled.csv',trehsold=0.95):
    
    dimension_df = pd.read_csv(directory,index_col=0)
    dimension_df = binarize_knowledge(data,dimension_df,trehsold)
    knowledge_binary_percentage=[]
    debate_wise_knowledge_binary_percentage=[]

    for platform in platforms:
        plat_data = dimension_df[dimension_df['platform']==platform]
        for page_id in dimension_df['page_id'].unique():
            pg_data = plat_data[plat_data['page_id']==page_id]
            knowledge_debates=[]
            for debate_id in pg_data['debate_id'].unique():
                debate_data = pg_data[pg_data['debate_id']==debate_id]
                knowledge_debates.append(debate_data['binarized_know'].sum()/len(debate_data))
                #knowledge_std.append(debate_data['knowledge'].std())
            
            debate_wise_knowledge_binary_percentage+=knowledge_debates
            knowledge_binary_percentage.append(np.mean(knowledge_debates))
    
    return knowledge_binary_percentage,debate_wise_knowledge_binary_percentage