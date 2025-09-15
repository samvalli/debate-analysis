import numpy as np
from openai import OpenAI
import pandas as pd

client=OpenAI(api_key='insert your personal key here')

def get_disagreement_score(merged_data,platforms):
    debate_disagreement_score=[]

    for platform in platforms:
        plat_data = merged_data[merged_data['platform']==platform]
        for page_id in merged_data['page_id'].unique():
            page_data = plat_data[plat_data['page_id']==page_id]
            for debate_id in page_data['debate_id'].unique():
                num_direct_opposing=[]
                debate_data = page_data[page_data['debate_id']==debate_id]
                for i,row in debate_data.iterrows():
                    item_id = row['id']
                    child_data = debate_data[(debate_data['parent_id']==item_id)&(debate_data['stance']=='con')]
                    num_direct_opposing.append(len(child_data))
                debate_disagreement_score.append(np.sum(num_direct_opposing)/len(debate_data))

    return debate_disagreement_score


def get_openai_stance_identification(parent_claim,child_claim,topic,client):
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      #response_format={ "type": "json_object" },
      temperature=0.3,
      messages=[
        {"role": "system", "content": f"You are a helpful assistant designed to assign a label to the a child claim and return it as single word."},
        {"role": "user", "content": f"I will provide you with two claims on the same topic. One is the parent claim, and the other is the child claim. The topic is {topic}"+
         """Your task is to determine whether the child claim is supporting, opposing, or neutral with respect to the parent claim.
            Assign one of the following labels based on your assessment:\n"""+
            """ "pro" if the child claim express the same position as the parent claim \n"""+
            """ "con" if the child claim clearly opposes the parent claim or attacks its author \n"""+
            """ "neutral" if the child claim does not take a clear stance or is unrelated.\n"""+
            """If you are not sure you should assing "neutrual" label. Here are the claims: \n"""
            f"PARENT CLAIM: {parent_claim}\n"+
            f"CHILD CLAIM: {child_claim}\n"
         'Reply providing one single word, the assigned label"'},
      ]
    )
  return response.choices[0].message.content

def get_cmv_stance(cmv_data_complete,cmv_data):


    #Using the cmv_data_complete allow us to avoid loosing track of possible discarded post.
    # Merge the dataset with itself to get claim-parent pairs
    couples_df = cmv_data_complete.merge(cmv_data_complete, left_on="parent_id", right_on="id", suffixes=("_child", "_parent"))

    parents=[]
    childs=[]
    topic=[]
    child_id=[]
    
    for i, row in couples_df.iterrows():
        child_id.append(row['id_child'])
        topic.append(row['title_parent'])
        parents.append(row['item_parent'])
        childs.append(row['item_child'])
    
    
    stance_dict={'id':[],'parent_claim':[],'child_claim':[],'stance':[]}
    for id,top,par,chil in zip(child_id,topic,parents,childs):
        stance_dict['id'].append(id)
        stance_dict['parent_claim'].append(par)
        stance_dict['child_claim'].append(chil)
        stance=get_openai_stance_identification(par,chil,top,client)
        stance_dict['stance'].append(stance)
    
    stance_df = pd.DataFrame(stance_dict)
    stance_df = stance_df.drop_duplicates(subset=["id"])

    stance=[]
    for i,row in cmv_data.iterrows():
      level=row['level']
      item_id = row['id']

      if level==0:
        stance.append("neutral")
      else:
        item_stance = stance_df[stance_df['id']==item_id]['stance'].item()
        stance.append(item_stance)

    cmv_data=cmv_data.assign(stance=stance)

    return cmv_data
