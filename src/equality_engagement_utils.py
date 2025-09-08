import numpy as np
import pandas as pd
from src.data_collection.kialo_utils import *
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
    
    directory_aut = 'data/raw_data_kialo/kialo_authors/'
    directory_cont = 'data/raw_data_kialo/kialo_page_content/'
    engagement_value=[]
    equality_value=[]

    for platform in platforms:
        plat_data = merged_data[merged_data['platform']==platform]
        for page_id in merged_data['page_id'].unique():
            page_data = plat_data[plat_data['page_id']==page_id]

            #Routine for Wikidebate
            if platform=='wiki':

                wiki_data=merged_data[merged_data['platform']=='wiki']
                wiki_data['author']=wiki_data['author'].apply(ast.literal_eval)

                #Collect all authors
                wiki_aut = wiki_data['author'].tolist()
                wiki_aut = [aut for sublist in wiki_aut for aut in sublist]
                counts = collections.Counter(wiki_aut)
                num_it = len(page_data)

                #Evaluate metrics
                engagement=len(list(counts.keys()))/num_it
                inverse_gini=1-gini(np.array(list(counts.values())))
                engagement_value.append(engagement)
                equality_value.append(inverse_gini)



            #Routine for Kialo
            if platform == 'kialo':

                #Acess raw data to obtain inforation about users activity
                complete_dir_aut = directory_aut +'/'+str(page_id)
                complete_dir_content = directory_cont +'/'+str(page_id)
                files=os.listdir(complete_dir_aut)

                #Needed if more than one debate is found for each topic
                debate_engagement_values=[]
                debate_gini_index=[]

                for i,file in enumerate(files):

                    #Extract information from raw data
                    authors,edits = extract_participants(complete_dir_aut+'/'+file)
                    num_it = count_kialo_items(complete_dir_content+f'/{file}')

                    #Evaluate metrics
                    engagement=len(authors)/num_it
                    inverse_gini=1.0-gini(np.array(edits))
                    debate_engagement_values.append(engagement)
                    debate_gini_index.append(inverse_gini)

                engagement_value.append(np.mean(debate_engagement_values))
                equality_value.append(np.mean(debate_gini_index))



            #Routine for ChangeMyView
            if platform == 'cmv':

                #Needed if more than one debate is found for each topic
                debate_engagement_values=[]
                debate_gini_index=[]

                #Find debates separation points
                page_data = page_data.reset_index()
                level_0 = page_data[page_data['level']==0].index.values.tolist()

                for i in range(len(level_0)):

                    #Separate debates
                    if i==len(level_0)-1:
                        debate_data = page_data[level_0[i]:]
                    else:
                        debate_data = page_data[level_0[i]:level_0[i+1]]
                    
                    #Collect all authors
                    cmv_aut = debate_data['author'].tolist()
                    counts = collections.Counter(cmv_aut)
                    num_it = len(debate_data)

                    #Evaluate metrics
                    engagement=len(list(counts.keys()))/num_it
                    inverse_gini=1-gini(np.array(list(counts.values())))
                    debate_engagement_values.append(engagement)
                    debate_gini_index.append(inverse_gini)

                engagement_value.append(np.mean(debate_engagement_values))
                equality_value.append(np.mean(debate_gini_index))

    return engagement_value,equality_value