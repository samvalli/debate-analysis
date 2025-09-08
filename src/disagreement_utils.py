import numpy as np

def assign_disagreement_score(data,platforms):
    disagreement_score=[]
    debate_wise_disagreement_score=[]

    for platform in platforms:
        plat_data = data[data['platform']==platform]
        for page_id in data['page_id'].unique():
            page_data = plat_data[plat_data['page_id']==page_id]
            page_disagreement_score=[]
            for debate_id in page_data['debate_id'].unique():
                num_direct_opposing=[]
                debate_data = page_data[page_data['debate_id']==debate_id]
                for i,row in debate_data.iterrows():
                    item_id = row['id']
                    child_data = debate_data[(debate_data['parent_id']==item_id) & (debate_data['stance']=='con')]
                    num_direct_opposing.append(len(child_data))
                page_disagreement_score.append(np.mean(num_direct_opposing))
            disagreement_score.append(np.mean(page_disagreement_score))
            debate_wise_disagreement_score+=page_disagreement_score
    
    return debate_wise_disagreement_score,disagreement_score