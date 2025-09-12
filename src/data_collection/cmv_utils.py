import re
from html import unescape
import emoji
import pandas as pd
from global_utils.thread_utils import *
import praw
import datetime

def extract_post_id(url):
    return url.split('/')[6]



def praw_cmv_debates_to_dict(id,title,page_id,dict,client_id,client_secret,user_agent,username,password):

    level_dict={}
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password
    )
    submission = reddit.submission(id=id)


    #Shared data
    original_title = submission.title
    time=submission.created
    date= datetime.datetime.fromtimestamp(time)

    #Manage the OP
    original_item = submission.selftext
    preproccesed_text = preprocess_reddit_text(original_item)
    id = submission.id
    dict['title'].append(title)
    dict['page_id'].append(page_id)
    dict['item'].append(preproccesed_text)
    dict['length'].append(len(preproccesed_text))
    dict['original_length'].append(len(original_item))
    dict['original_item'].append(original_item)
    dict['id'].append(id)
    dict['parent_id'].append(0) 
    author_name = submission.author.name if submission.author else "[deleted]"
    dict['author'].append(author_name)
    dict['score'].append(submission.score)
    dict['original_title'].append(original_title)
    dict['level'].append(0)
    dict['date'].append(date)
    level_dict[id]=0
    #print(level_dict)

    submission.comments.replace_more(limit=None)
    comment_list=submission.comments.list()
    for com in comment_list:
        original_item = com.body
        preproccesed_text = preprocess_reddit_text(original_item)
        id = com.id
        parent_id = com.parent().id
        #print(parent_id)
        level = level_dict[str(parent_id)]+1
        dict['title'].append(title)
        dict['page_id'].append(page_id)
        dict['item'].append(preproccesed_text)
        dict['length'].append(len(preproccesed_text))
        dict['original_length'].append(len(original_item))
        dict['original_item'].append(original_item)
        dict['id'].append(id)
        dict['parent_id'].append(parent_id) 
        dict['author'].append(com.author)
        dict['score'].append(com.score)
        dict['original_title'].append(original_title)
        dict['level'].append(level)
        dict['date'].append(date)
        level_dict[id]=level
    return dict

def collect_data_cmv(directory,directory_output,client_id,client_secret,user_agent,username,password):
    cmv_page_data = pd.read_csv(directory)
    cmv_page_data = cmv_page_data[['Wiki_title','page_id','Link_1','Link_2','Link_3','Link_4','Link_5']]
    dict={
        'page_id':[],
        'title':[],
        'original_title':[],
        'id':[],
        'parent_id':[],
        'item':[],
        'length':[],
        'level':[],
        'author':[],
        'score':[],
        'original_length':[],
        'original_item':[],
        'date':[],
    }

    for i,row in cmv_page_data.iterrows():
        links=[]
        page_id = row['page_id']
        title = row['Wiki_title']
        for k in range(1,5):
            if pd.notna(row[f'Link_{k}']):
                links.append(row[f'Link_{k}'])
        for link in links:
            print(link)
            post_id = extract_post_id(link)
            dict = praw_cmv_debates_to_dict(post_id,title,page_id,dict,client_id,client_secret,user_agent,username,password)

    cmv_items_df = pd.DataFrame(dict)
    cmv_items_df.to_csv(directory_output)
    return cmv_items_df


def preprocess_reddit_text(text):
    # Remove lines that contain only underscores
    text = re.sub(r"^_+\s*$", "", text, flags=re.MULTILINE)
    # Remove any extra newlines left after deletion
    text = re.sub(r"\n\s*\n", "\n", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove Reddit-specific mentions
    text = re.sub(r"u/\w+|r/\w+", "", text)
    # Strip HTML entities
    text = unescape(text)
    #Remove quoted items
    text = re.sub(r"^>\s?.*", "", text, flags=re.MULTILINE).strip()
    # Remove special characters
    text = re.sub(r"[^\w\s.,!?']", "", text)
    # Remove extra whitespace
    text = " ".join(text.split())
    # Convert emojis to text
    text = emoji.demojize(text)
    
    return text

def find_cmv_titles(cmv_titles,kialo_titles,model,kialo_page_data):
    original_titles=[]
    page_ids=[]
    new_titles=[]
    kialo_embeddings = model.encode(kialo_titles, convert_to_tensor=True)
    cmv_embeddings = model.encode(cmv_titles,convert_to_tensor=True)
    
    for i,title_emb in enumerate(kialo_embeddings):
        print("=================================")
        title_index = find_related_title(title_emb,cmv_embeddings)
        original_titles.append(cmv_titles[title_index])
        title_selected=kialo_titles[i]
        print(title_selected)
        print(original_titles[-1])
        print('==================================')
        page_id = kialo_page_data[kialo_page_data['title']==title_selected]['page_id'].item()
        page_ids.append(page_id)
        new_titles.append(title_selected)
    return page_ids,new_titles,original_titles


def get_cmv_full_debates_items(conversations,corpus,kialo_titles,cmv_selected_titles,page_ids):   
    length=[]
    original_length=[]
    item=[]
    original_item=[]
    kialo_title=[]
    cmv_title=[]
    page_id=[]
    id = []
    parent_id=[]
    author=[]
    conv_id=[]
    score=[]
    for i,conversation in enumerate(conversations):
        if isinstance(conversation,str):
            continue
        for utt_id in conversation.get_utterance_ids():
            kialo_title.append(kialo_titles[i])
            cmv_title.append(cmv_selected_titles[i])
            page_id.append(page_ids[i])
            #keyword_counter=0
            utt=corpus.get_utterance(utt_id)
            text=utt.text
            original_item.append(text)
            original_length.append(len(text))
            item.append(preprocess_reddit_text(text))
            length.append(len(item[-1]))
            id.append(utt_id)
            parent_id.append(utt.reply_to)
            author.append(utt.speaker.id)
            conv_id.append(utt.conversation_id)
            score.append(utt.meta['score'])
    return page_id,kialo_title,cmv_title,item,length,id,parent_id,author,conv_id,score,original_item,original_length


def remove_nan_imtems(cmv_items):
    id_null_items=cmv_items[cmv_items['item'].isnull()]['id']
    id_to_remove=[]
    for id in id_null_items:
        data = cmv_items[cmv_items['id']==id]
        child_data = cmv_items[cmv_items['parent_id']==id]
        if len(child_data)>0:
            cmv_items.at[data.index[0],'item']=''
        else:
            id_to_remove.append(id)

    cmv_items=cmv_items[~cmv_items['id'].isin(id_to_remove)]


def parentid_and_threaddi_kialo_and_cmv(kialo_data_complete,cmv_data_complete):
    datas = [kialo_data_complete,cmv_data_complete]
    parent_id_k_dict={'id':[],'parent_id':[],'thread_id':[]}
    for i,data in enumerate(datas):
        if i==0:
            platform = 'k'
        else:
            platform = 'c'
        for page_id in data['page_id'].unique():
            page_data = data[data['page_id']==page_id]
            first_parent_id=[]

            j=0
            for i,row in page_data.iterrows():
                if row['level']==0:
                    first_parent_id.append((row['id'],j))
                    parent_id_k_dict['id'].append(row['id'])
                    parent_id_k_dict['parent_id'].append(str(0))
                    parent_id_k_dict['thread_id'].append(str(row['page_id'])+'_'+str(len(first_parent_id)))

                j+=1
            
            page_data = page_data[page_data['level']>0]
            page_data = page_data.reset_index()
            print("=== new page ===")
            print(first_parent_id)
            print(page_id)
            #print(len(page_data))
            page_data = assign_parent_and_thread_id_to_data(page_data,page_id,platform=platform,first_parent_id=first_parent_id)
            parent_id_k_dict['id']+=page_data['id'].tolist()
            parent_id_k_dict['parent_id']+=page_data['parent_id'].tolist()
            parent_id_k_dict['thread_id']+=page_data['thread_id'].tolist()

    obtained_parent_id = pd.DataFrame(parent_id_k_dict)
    obtained_parent_id

    parent_ids=[]
    thread_ids=[]
    for i,row in merged_data.iterrows():
        target_id=row['id']
        if target_id in obtained_parent_id['id'].tolist():
            parent_ids.append(obtained_parent_id.loc[obtained_parent_id['id'] == target_id, 'parent_id'].values[0])
            thread_ids.append(obtained_parent_id.loc[obtained_parent_id['id'] == target_id, 'thread_id'].values[0])
        else:
            parent_ids.append(row['parent_id'])
            thread_ids.append(row['thread_id'])
    merged_data=merged_data.assign(parent_id=parent_ids,thread_id=thread_ids)
    merged_data.to_csv('data/global/merged_data_08_01_25.csv')