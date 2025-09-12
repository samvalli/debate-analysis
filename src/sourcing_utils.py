import re
import mwparserfromhell
from src.data_collection.wikidebate_utils import parse

REDDIT_URLS=['http://www.reddit.com/r/changemyview/wiki/rules',
 'http://www.reddit.com/r/changemyview/wiki/guidelines#wiki_upvoting.2Fdownvoting',
 'http://www.reddit.com/r/changemyview/wiki/populartopics',
 'http://www.reddit.com/message/compose?to=/r/changemyview']

def find_external_sources_cmv(text):
    # Improved regex to exclude trailing punctuation like ')', ']', etc.
    link_pattern = r"https?://[^\s\]\)]+|www\.[^\s\]\)]+"
    links = re.findall(link_pattern, text)
    
    # Filter out links that are in REDDIT_URLS (if applicable)
    links = [link for link in links if link not in REDDIT_URLS]
    
    return len(links), links

def find_references_number_kialo(text):

    # Regex to match [number] patterns
    reference_pattern = r"\[\d+\]"
    references = re.findall(reference_pattern, text)
    return len(references)  


def get_reference_infos(code):
    references=[]
    refered_sentences=[]
    ref_counter=0
    last_setnece=''
    for section in code.get_sections(levels=[3],flat=True):
        tags = code.filter_tags()
        prev_node = section.nodes[0]
        for node in section.nodes:
            if '<ref' in node and node in tags:
                ref_counter +=1
                if node not in references:
                    references.append(str(node))
                if prev_node not in references and '<ref name' not in prev_node:
                    refered_sentences.append(str(prev_node))
                    last_setnece=prev_node
                else:
                    refered_sentences.append(last_setnece)
            elif '<ref name' in node and node in tags :
                if prev_node not in references  and '<ref name' not in prev_node:
                    refered_sentences.append(str(prev_node))
                    last_setnece=prev_node
                else:
                    refered_sentences.append(last_setnece)
            if len(str(node))>5 and node not in code.filter_templates():
                prev_node=node
    return [refered_sentences,len(refered_sentences),ref_counter,len(references),references]

def get_wiki_references(wiki_data):

    titles = wiki_data[wiki_data['platform']=='wiki']['title'].unique().tolist()

    for title in titles:
        code,link = parse(title)
        refered_sentences,num_sent_referenced,ref_counter,num_ref,references = get_reference_infos(code)
        return ref_counter


def get_platforms_reference_number(merged_data,platforms):

    wiki_num_ref=0
    kialo_num_ref=0
    cmv_num_ref=0

    for platform in platforms:
        plat_data = merged_data[merged_data['platform']==platform]
        for page_id in merged_data['page_id'].unique():
            page_data = plat_data[plat_data['page_id']==page_id]

            #Routine for Wikidebate
            if platform=='wiki':
                wiki_num_ref=get_wiki_references(page_data)
            
            #Routine for Kialo
            if platform=='kialo':
                for i,row in page_data.iterrows():
                    text = row['item']
                    title = row['title']
                    nref = find_references_number_kialo(text)
                    kialo_num_ref+=nref

            #Routine for CMV
            if platform=='cmv':
                page_data['original_item']=page_data['original_item'].astype(str)
                for text in plat_data['original_item'].tolist():
                    nref,ref = find_external_sources_cmv(text)
                    cmv_num_ref+=nref

    return wiki_num_ref,kialo_num_ref,cmv_num_ref


def divide_reference_by_page(data,references,platform,merged_data):
    links_by_page=[]
    if platform=='kialo' or platform=='wiki':
        last_index=0
        for i,page_id in enumerate(data['page_id'].unique()):
            page_data = data[data['page_id']==page_id]
            start = i+last_index
            n_items = len(page_data)
            page_ref = []
            for i in range(start,start+n_items):
                page_ref+=references[i]
            links_by_page.append(page_ref)

    if platform=='cmv':
        cmv_merged = merged_data[merged_data['platform']=='cmv']
        page_ref=[]
        last_page_id = cmv_merged['page_id'].tolist()[0]
        for i,row in data.iterrows():
            
            page_id = row['page_id']
            if page_id==1011  or page_id==1078:
                continue
            if page_id != last_page_id:
                links_by_page.append(page_ref)
                page_ref=[]
            if row['id'] in cmv_merged['id'].tolist():
                if row['author']=='autowikibot':
                    last_page_id = page_id
                    continue
                page_ref+=references[i]
            last_page_id = page_id
        links_by_page.append(page_ref)
    return links_by_page


