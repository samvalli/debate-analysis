import pandas as pd

def get_h_index_of_pages_users(pages_data,user_page_id_dict):
    h_index_dict={}
    for pageid in pages_data['pageid']:
        page_occurences=[]
        for user in user_page_id_dict:
            occurences=user_page_id_dict[user].count(pageid)
            if occurences>0:
                page_occurences.append(occurences)
        title=pages_data[pages_data['pageid']==pageid]['title'].item()
        h_index_dict[title]=h_index(page_occurences)
    h_index_data_frame=pd.DataFrame.from_dict(h_index_dict,orient='index')
    #h_index_data_frame.columns=['h_index']
    return h_index_data_frame


def h_index(occurences_list):
    occurences_list.sort(reverse=True)
    h_index=0
    for i,num in enumerate(occurences_list,1):
        if num >= i:
            h_index = i
        else:
            break
    return h_index