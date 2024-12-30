from sentence_transformers import SentenceTransformer, util
import os
import glob
import json
import pandas as pd

def find_related_title(kialo_title_emb,wiki_titles_emb):
    similarity=[]
    for i,title in enumerate(wiki_titles_emb):
        embeddings=[kialo_title_emb,title]    
        similarity.append(util.pytorch_cos_sim(embeddings[0], embeddings[1]).item())
    similarity_max = max(similarity)
    title_index = similarity.index(similarity_max)
    return title_index

def find_matching_titles(dir_files,page_data,wiki_titles,model):
    original_titles=[]
    page_ids=[]
    new_titles=[]
    wiki_embeddings = model.encode(wiki_titles, convert_to_tensor=True)
    for dir in dir_files:
        title=dir[:-9].replace('data/kialo/kialo_page_data/','').replace('-',' ')
        title_emb = model.encode(title, convert_to_tensor=True)
        title_index = find_related_title(title_emb,wiki_embeddings)
        original_titles.append(title)
        title_selected=wiki_titles[title_index]
        print(title_selected)
        print(title)
        print('==================================')
        page_id = page_data[page_data['title']==title_selected]['page_id'].item()
        page_ids.append(page_id)
        new_titles.append(title_selected)
        os.rename(dir,'data/kialo/kialo_page_data/'+'Kialo '+title_selected+'.txt')
    return page_ids,new_titles,original_titles


def create_kialo_csv(dir_files,kialo_page_data):
    page_ids=[]
    ids=[]
    wiki_title=[]
    kialo_original_title=[]
    item=[]
    level=[]
    length=[]
    stance=[]
    for dir in dir_files:
        title = dir.replace('data/kialo/kialo_parsed_content/','').replace('.json','').replace('Kialo','').lstrip()
        print(title)
        page_data = kialo_page_data[kialo_page_data['tile']==title]
        with open(dir,'r') as file:
            data=json.load(file)
        for i,elem in enumerate(data):
            page_ids.append(page_data['page_id'].item())
            ids.append(str(page_ids[-1])+'_k_'+str(i).zfill(4))
            wiki_title.append(page_data['tile'].item())
            kialo_original_title.append(page_data['original_title'].item())
            content=elem['Content'].lstrip()
            item.append(content)
            length.append(len(content))
            level.append(elem['Level'])
            stance.append(elem['Stance'])
    
    kialo_items_dict={
        'page_id':page_ids,
        'title':wiki_title,
        'kialo_title':kialo_original_title,
        'id':ids,
        'item':item,
        'length':length,
        'level':level,
        'stance':stance,
    }

    kialo_items_df = pd.DataFrame(kialo_items_dict)
    kialo_items_df.to_csv('data/argumentation/kialo_all_items.csv')