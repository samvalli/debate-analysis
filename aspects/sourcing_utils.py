import re
import mwparserfromhell

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

def find_references_number_kialo(text,page_id):
    # Regex to match [number] patterns
    reference_pattern = r"\[\d+\]"
    references = re.findall(reference_pattern, text)

    #THIS PART HAS TO BE SEPARATED TO MAKE A FUNCTION TO EXTRACT SOURCES FROM RAW TEXT
    #input_dir = f'data/kialo/kialo_page_updated/{str(page_id)}'
    # with open(input_dir, 'r') as fi:
    #     links_references = []
    #     soureces_flag=False
    #     for line in fi:
    #         if line.lstrip().startswith('Sources:'):
    #             soureces_flag=True
    #         if soureces_flag==False:
    #             continue
    #         for ref in references:
    #             if  line.startswith(ref):
    #                 if "http" in line:
    #                     link=re.findall("http\S+|www\S+",line)[0]
    #                 else: 
    #                     link=re.sub("\[\d+\]",'',line)
    #                     link=link.lstrip()
    #                 links_references.append(link)
    return len(references)  #, links_references

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

def extract_links_wiki(text):
    url_pattern = r'https?://[^\s\|]+'
    wikipedia_pattern = r'\[\[Wikipedia:[^\]]+\]\]'
    links = re.findall(url_pattern, text)
    wikipedia_links = re.findall(wikipedia_pattern, text)
    wikipedia_links = [link[2:-2] for link in wikipedia_links]  # Remove '[[' and ']]'
    all_links = links + wikipedia_links
    
    return all_links


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


def get_platforms_reference_number(wiki_data,kialo_data,cmv_data):
    cmv_references=[]
    cmv_num_ref=[]
    kialo_num_ref=[]
    num_ref=wiki_data['references'].tolist()
    #kialo_references=[]

    cmv_data['original_item']=cmv_data['original_item'].astype(str)
    for text in cmv_data['original_item'].tolist():
        nref,ref = find_external_sources_cmv(text)
        cmv_references.append(ref)
        cmv_num_ref.append(nref)

    for i,row in kialo_data.iterrows():
        text = row['item']
        title = row['title']
        nref = find_references_number_kialo(text,title)
        kialo_num_ref.append(nref)
        #kialo_references.append(ref)

    num_ref+=kialo_num_ref+cmv_num_ref
    return num_ref