import re
from html import unescape
import emoji

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
