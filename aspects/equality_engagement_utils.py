import numpy as np
import pandas as pd
from kialo_utils.kialo_authors import *
import collections
import os
import ast

def gini(array):

    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    
    array = array + 0.0000001
    array = np.sort(array)
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))
    
def engagement_and_equality_assignment(merged_data,platforms):
    
    directory = 'data/kialo/kialo_authors'
    wiki_complete = pd.read_csv('data/global/wiki_complete_items.csv',index_col=0)
    cmv_complete = pd.read_csv('data/global/cmv_items.csv',index_col=0)
    wiki_complete['authors'] = wiki_complete['authors'].apply(ast.literal_eval)
    authros_dict={
        'platform':[],
        'page_id':[],
        'debate_num':[],
        'authors':[],
        'num_edits':[],
    }
    engagement_value=[]
    gini_index=[]
    debate_wise_engagement=[]
    debate_gini_index=[]
    debate_wise_gini=[]

    for platform in platforms:
        plat_data = merged_data[merged_data['platform']==platform]
        for page_id in merged_data['page_id'].unique():
            page_data = plat_data[plat_data['page_id']==page_id]
            if platform=='wiki':
                title = page_data['title'].tolist()[0]
                rev_data = pd.read_csv(f'data/argumentation/revisions_data/{title}.csv')
                rev_data[rev_data['minor']==0]
                authros_dict['platform'].append('wiki')
                authros_dict['page_id'].append(page_id)
                authros_dict['debate_num'].append(0)
                ids = page_data['id'].tolist()
                wiki_aut = wiki_complete[wiki_complete['id'].isin(ids)]['authors'].tolist()
                wiki_aut = [aut for sublist in wiki_aut for aut in sublist]
                counts = collections.Counter(wiki_aut)
                authros_dict['authors'].append(list(counts.keys()))
                authros_dict['num_edits'].append(list(counts.values()))
                num_it = len(page_data)
                engagement=len(authros_dict['authors'][-1])/num_it
                inverse_gini=1-gini(np.array(authros_dict['num_edits'][-1]))
                engagement_value.append(engagement)
                debate_wise_engagement.append(engagement)
                gini_index.append(inverse_gini)
                debate_wise_gini.append(inverse_gini)

            if platform == 'kialo':
                complete_dir = directory +'/'+str(page_id)
                files=os.listdir(complete_dir)
                debate_engagement_values=[]
                debate_gini_index=[]
                for i,file in enumerate(files):
                    authors,edits = extract_participants(complete_dir+'/'+file)
                    authros_dict['platform'].append('kialo')
                    authros_dict['page_id'].append(page_id)
                    authros_dict['debate_num'].append(i)
                    authros_dict['authors'].append(authors)
                    authros_dict['num_edits'].append(edits)
                    num_it = count_debate_items_mod('data/kialo/kialo_page_updated/'+str(page_id)+f'/{file}')
                    engagement=len(authros_dict['authors'][-1])/num_it
                    inverse_gini=1.0-gini(np.array(authros_dict['num_edits'][-1]))
                    debate_engagement_values.append(engagement)
                    debate_gini_index.append(inverse_gini)

                engagement_value.append(np.mean(debate_engagement_values))
                debate_wise_engagement+=debate_engagement_values
                gini_index.append(np.mean(debate_gini_index))
                debate_wise_gini+=debate_gini_index

            if platform == 'cmv':
                debate_engagement_values=[]
                debate_gini_index=[]
                page_data = page_data.reset_index()
                level_0 = page_data[page_data['level']==0].index.values.tolist()
                for i in range(len(level_0)):
                    if i==len(level_0)-1:
                        debate_data = page_data[level_0[i]:]
                    else:
                        debate_data = page_data[level_0[i]:level_0[i+1]]
                    ids = debate_data['id'].tolist()
                    cmv_aut = cmv_complete[cmv_complete['id'].isin(ids)]['author'].tolist()
                    counts = collections.Counter(cmv_aut)
                    authros_dict['platform'].append('cmv')
                    authros_dict['page_id'].append(page_id)
                    authros_dict['debate_num'].append(i)
                    authros_dict['authors'].append(list(counts.keys()))
                    authros_dict['num_edits'].append(list(counts.values()))
                    num_it = len(debate_data)
                    engagement=len(authros_dict['authors'][-1])/num_it
                    inverse_gini=1-gini(np.array(authros_dict['num_edits'][-1]))
                    debate_engagement_values.append(engagement)
                    debate_gini_index.append(inverse_gini)

                engagement_value.append(np.mean(debate_engagement_values))
                debate_wise_engagement+=debate_engagement_values
                gini_index.append(np.mean(debate_gini_index))
                debate_wise_gini+=debate_gini_index

    return engagement_value,gini_index,debate_wise_engagement,debate_wise_gini,authros_dict