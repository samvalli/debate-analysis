import mwparserfromhell
from difflib import SequenceMatcher,ndiff,restore
from sentence_transformers import SentenceTransformer,util
from mwedittypes import StructuredEditTypes
import math
from datetime import datetime
import difflib 
import pandas as pd

EXCEPTION_TITLES=['Are humans omnivores or herbivores?']
NOTABLE_TIMESTAMP_OMNIVORES='2022-11-19T00:00:00Z'

def find_string_difference(string1, string2):
    string1=str(string1.strip('\n'))
    string2=str(string2.strip('\n'))
    matcher = SequenceMatcher(None, string1, string2)
    ratio= matcher.ratio()
    return ratio

def check_remove(oldnode,removed):
    if oldnode in removed:
        removed.remove(oldnode)


def check_added_list(new_node,old_code,index,removed):
    if index==0:
        #print(f"new node was added: {new_node}")
        ratio=0.0
        return "added", ratio #flag for addition
    else:
        old_node=old_code[0]
        ratio=find_string_difference(new_node,old_node) 
        if new_node == old_node:
            # print("found identical node")
            # print(new_node)
            check_remove(old_node,removed)
            ratio=1.0
            return "nodefound", ratio
        else:
            if ratio>0.6:
                # print(f"found difference not from direct, ratio is {find_string_difference(new_node,old_node)}")
                # print(f"previous node was: {old_node}")
                # print(removed)
                check_remove(old_node,removed)
                return "modified", ratio #flag for modification
            else: 
                return check_added_list(new_node,old_code[1:],index-1,removed)

def check_added_list_new(new_node,old_code):
    for old_node in old_code:
        ratio=find_string_difference(new_node,old_node) 
        if new_node == old_node:
            check_remove(old_node,old_code)
            ratio=1.0
            return "nodefound", ratio 
        else:
            if ratio>0.6:
                # print(f"found difference not from direct, ratio is {find_string_difference(new_node,old_node)}")
                # print(f"previous node was: {old_node}")
                # print(removed)
                check_remove(old_node,removed)
                return "modified", ratio #flag for modification
            else: 
                return check_added_list(new_node,old_code[1:],index-1,removed)
            

def obtain_changes_list_mod(newsect,oldsect):
    removed=list(oldsect)
    i=0
    prev_rev_node=oldsect[i]
    count_modifications=0
    count_removed=0
    count_added=0
    tot_ratio=0

    for node in newsect:
        
        remaining_indexes=len(oldsect)-i-1
        if node == prev_rev_node:
            if remaining_indexes>0:
                check_remove(prev_rev_node,removed)
                i+=1
                prev_rev_node=oldsect[i]
            else:
                check_remove(prev_rev_node,removed)
                continue
        else:
            if remaining_indexes<0:
                flag="added"
                count_added+=1
                continue
            ratio=find_string_difference(node,prev_rev_node)
            if ratio>0.5:
                if ratio<1.0:
                    count_modifications+=1
                    tot_ratio+=ratio
                
                check_remove(prev_rev_node,removed)
                if remaining_indexes>0:
                    i+=1
                    prev_rev_node=oldsect[i]
                
                #print(f"found_modification in this node {node}")
            else:
                flag,ratio=check_added_list(node,oldsect[i+1:],remaining_indexes,removed)
                #print(ratio)
                if flag=="added":
                    count_added+=1
                # if flag =="removed":
                #     i+=1
                #     prev_rev_node=oldsect[i]
                #     count_removed+=1
                if flag =="modified":
                    if ratio<1.0:
                        
                        count_modifications+=1
                        tot_ratio+=ratio
    #print(removed)
    count_removed=len(removed)
    if count_modifications>0:
        mean_ratio=1-(tot_ratio/count_modifications)
    else:
        mean_ratio=0.0
    return count_added,count_modifications,count_removed,mean_ratio,removed

def get_diff_ratio(diff,error_flag):
    error_flag=False
    removed_senteces=[]
    removed_words=[]
    paragraphs_curr=[]
    paragraphs_prev=[]
    for edit in diff['text-edits']:
        if edit[0]=='Paragraph':
            if edit[1]=='insert':
                paragraphs_curr.append(edit[2].splitlines(keepends=True))
                #print(edit[2].splitlines(keepends=True))
            if edit[1]=='remove':
                paragraphs_prev.append(edit[2].splitlines(keepends=True))
                #print(edit[2].splitlines(keepends=True))
        if edit[0]=='Sentence' and edit[1]=='remove':
            removed_senteces.append(edit[2])
        if edit[0]=='Word' and edit[1]=='remove':
            removed_words.append(edit[2])
    if len(paragraphs_curr)==len(paragraphs_prev):
        if len(paragraphs_curr)==0:
            error_flag=False
            print("find no content differences in the section")
            return 0,0,0,0.0,error_flag,[],removed_senteces,removed_words
        else:
            if len(paragraphs_curr)>1:
                error_flag=False
                #print(len(paragraphs_curr))
                paragraphs_curr=[sum(paragraphs_curr,[])]
                paragraphs_prev=[sum(paragraphs_prev,[])]
                print("found more than one paragraph merged the paragraphs")
                #return 0,0,0,0.0,error_flag
            # print(f"curr parag {paragraphs_curr[0]}")
            # print(f"prev parag {paragraphs_prev[0]}")
            count_added,count_modifications,count_removed,mean_ratio,removed=obtain_changes_list_mod(paragraphs_curr[0],paragraphs_prev[0])
    else:
        #error_flag=True
        void=''
        print(f"the len of current paragraph is {len(paragraphs_curr)}")
        print(f"the len of prev paragraph is {len(paragraphs_prev)}")
        # print("ERROR: paragraphs don't have same length")
        # return 0,0,0,0,error_flag
        if len(paragraphs_curr)==0:
            paragraphs_curr=[[void]]
        if len(paragraphs_prev)==0:
            paragraphs_prev=[[void]]
            
        #print(len(paragraphs_curr))
        paragraphs_curr=[sum(paragraphs_curr,[])]
        paragraphs_prev=[sum(paragraphs_prev,[])]
        print("found more than one paragraph merged the paragraphs")
        print(f"curr parag {paragraphs_curr[0]}")
        print(f"prev parag {paragraphs_prev[0]}")
        count_added,count_modifications,count_removed,mean_ratio,removed=obtain_changes_list_mod(paragraphs_curr[0],paragraphs_prev[0])
    return count_added,count_modifications,count_removed,mean_ratio,error_flag,removed,removed_senteces,removed_words

def get_headings(code,title,model):
    headings=[]
    wrong_headgins=0
    for_str=['Arguments for','first possibility','Pro','For']
    ag_str=['Arguments against','second possibility','Con','Against','Againt']
    for head in code.filter_headings():
        if any(l.casefold() in head.casefold() for l in for_str[:2]) or any(l in head for l in for_str[2:]):
            headings.append(head)
        if any(l.casefold() in head.casefold() for l in ag_str[:2]) or any(l in head for l in ag_str[2:]):
            headings.append(head)
    if len(headings)==0: #Headings are not standardized
        print("no normalization found")
        headings=find_actual_heads(code,title,model)
    if len(headings)!=2:
        if len(headings)==1:
            print("one head found")
            if headings[0]=='== Arguments against ==':
                if title in EXCEPTION_TITLES_MISSING_WORD_HEAD:
                    print("entred exception")
                    headings.insert(0,'== Arguments ==')
            return headings,wrong_headgins
        wrong_headgins=1
        print(f"found {len(headings)} heads")
  
    return headings,wrong_headgins
            
def find_actual_heads(code,title,model):
    headings=[]
    for head in code.filter_headings():
        if 'votes' in head.casefold():
            return headings
        # if head in headings:
        #     return headings
        if compute_distance(head,title,model)>0.79:
            headings.append(head)
    return headings

def compute_distance(oldnode,newnode,model):
    str_vec=[str(oldnode.strip('=')),str(newnode)]
    embeddings=model.encode(str_vec)
    dist=util.pytorch_cos_sim(embeddings[0],embeddings[1])
    return dist

def check_insert_remove(node_edits):
    num_obj_added=0
    num_obj_removed=0
    num_arg_added=0
    num_arg_removed=0
    num_comments=0
    if 'insert' in node_edits:
        if 'Objection' in node_edits[3]:
            num_obj_added+=1
        if 'Argument' in node_edits[3]:
            num_arg_added+=1
    if 'remove' in node_edits:
        if 'Objection' in node_edits[3]:
            num_obj_removed+=1
        if 'Argument' in node_edits[3]:
            num_arg_removed+=1
    return num_arg_added,num_arg_removed,num_obj_added,num_obj_removed

def check_modifications_routine(diff,dict,flag):
    counted_mod=[0,0,0,0]
    for edit in diff['node-edits']:
        if edit[0]=='Template':
            curr_mod=list(check_insert_remove(edit))
            counted_mod = [sum(i) for i in zip(counted_mod, curr_mod  )] 
    if flag==0:
        dict['Arg for'].append(counted_mod)
        flag+=0.5
    if flag==1:
        dict['Arg ag'].append(counted_mod)
        flag-=1.5
    flag=math.ceil(flag)
    
    return counted_mod,flag

def find_right_section(head,sections):
    for section in sections:
        if not section.nodes:
            continue
        elif section.nodes[0]==head:
            return section
        
def check_insert_remove_one_head(node_edits):
    num_obj_added=0
    num_obj_removed=0
    num_pro_arg_added=0
    num_pro_arg_removed=0
    num_con_arg_added=0
    num_con_arg_removed=0
    pro_str=['pro']
    con_str=['against','con']
    if 'insert' in node_edits:
        if 'Objection' in node_edits[3]:
            num_obj_added+=1
        if 'argument' in node_edits[3].casefold():
            if any(strin in node_edits[3].casefold() for strin in pro_str):
                num_pro_arg_added+=1
            if any(strin in node_edits[3].casefold() for strin in con_str):
                num_con_arg_added+=1
            else:
                print("ERROR: no sentiment of the argument specified")
    if 'remove' in node_edits:
        if 'Objection' in node_edits[3]:
            num_obj_removed+=1
        if 'argument' in node_edits[3].casefold():
            if any(strin in node_edits[3].casefold() for strin in pro_str):
                num_pro_arg_removed+=1
            if any(strin in node_edits[3].casefold() for strin in con_str):
                num_con_arg_removed+=1
            else:
                print("ERROR: no sentiment of the argument specified")
            
    return num_pro_arg_added,num_pro_arg_removed,num_con_arg_added,num_con_arg_removed,num_obj_added,num_obj_removed


def check_modifications_routine_one_head(diff,dict):
    counted_mod=[0,0,0,0,0,0]
    for edit in diff['node-edits']:
        if edit[0]=='Template':
            curr_mod=list(check_insert_remove_one_head(edit))
            print(curr_mod)
            counted_mod = [sum(i) for i in zip(counted_mod, curr_mod  )] 
    dict['Arg for'].append([counted_mod[0],counted_mod[4],counted_mod[1],counted_mod[5]])
    dict['Arg for 2'].append([counted_mod[0] +counted_mod[4],counted_mod[1]+counted_mod[5]])
    dict['Arg ag'].append([counted_mod[2],counted_mod[4],counted_mod[3],counted_mod[5]])
    dict['Arg ag 2'].append([counted_mod[2] +counted_mod[4],counted_mod[3]+counted_mod[5]])
    dict['Arg for mods'].append([0,0.0])
    dict['Arg ag mods'].append([0,0.0])
    
    return counted_mod


def one_head_routine(prev_head,curr_head,prev_sections,curr_sections,modif_dict):
    print("entred one header head section")
    prev_sec=find_right_section(prev_head,prev_sections)
    curr_sec=find_right_section(curr_head,curr_sections)
    et=StructuredEditTypes(prev_sec,curr_sec,lang='en')
    diff=et.get_diff()
    counted_mod=check_modifications_routine_one_head(diff,modif_dict)
    return counted_mod


def create_removed_items_dict(username,removed_content,r_s,r_w,timestamp,revid,parentid,comment):
    removed_items_dict={
        "username":username,
        "removed content":removed_content,
        "removed or modified sentences":r_s,
        "removed words": r_w,
        "num removed items":len(removed_content),
        "num removed or modified sent":len(r_s),
        "num removed words":len(r_w),
        "timestamp":timestamp,
        "revid":revid,
        "parentid":parentid,
        "comment":comment,
    }
    return removed_items_dict

def convert_timestamps(timestamps):
    cleaned_timestamps=[]
    format="%Y-%m-%dT%H:%M:%S"
    for time in timestamps:
        timestamp = time.replace('Z','')
        timestamp = datetime.strptime(timestamp,format)
        cleaned_timestamps.append(timestamp)
    return cleaned_timestamps

def compare_lists(old_list, new_list):
    # Initialize lists to store results
    deleted = []
    added = []
    modified = []
    unchanged = []  # To track items that haven't changed
    
    # Track items that have been matched to avoid double-counting
    matched_new_items = set()

    # Compare items from old_list to new_list using difflib
    for old_item in old_list:
        best_match = None
        best_ratio = 0
        
        for new_item in new_list:
            if new_item not in matched_new_items:
                match_ratio = difflib.SequenceMatcher(None, old_item, new_item).ratio()
                
                # Find the best matching item in the new list
                if match_ratio > best_ratio:
                    best_match = new_item
                    best_ratio = match_ratio
        
        # If the best match is exact (ratio of 1), consider it unchanged
        if best_ratio == 1.0:
            unchanged.append(old_item)
            matched_new_items.add(best_match)  # Mark the new item as matched
        # If the best match is close enough (> 0.6), consider it modified
        elif best_ratio > 0.5:
            modified.append((old_item, best_match,best_ratio))
            matched_new_items.add(best_match)  # Mark the new item as matched
        else:
            deleted.append(old_item)  # No good match, so consider it deleted

    # Any remaining items in new_list that weren't matched are considered added
    for new_item in new_list:
        if new_item not in matched_new_items:
            added.append(new_item)

    return {
        'deleted': deleted,
        'added': added,
        'modified': modified,
        'unchanged': unchanged  # Items that are identical in both lists
    }

def obtain_correct_ids(curr_items,prev_items,modifications_dict,items_ids,page_id,num_rev,secid):
    new_items_id = []
    deleted_ids = []
    modified_ids = []
    added_ids = []
    old_items_tuple = list(zip(prev_items,items_ids))
    added = modifications_dict['added']
    deleted = modifications_dict['deleted']
    modified = modifications_dict['modified']
    j=0

    for prev_tuple in old_items_tuple:
        if prev_tuple[0] in deleted:
            deleted_ids.append(prev_tuple[1])
        

    for item in curr_items:
        found_flag = 0
        if item in added:
            num_comment = str(j).zfill(4)
            item_id = page_id+'_'+f'{num_rev}'+'_'+f'{secid}'+'_'+f'{num_comment}'
            new_items_id.append(item_id)
            added_ids.append(item_id)
            j+=1
            continue

        for mod_elem in modified:
            if item in mod_elem:
                old_item = mod_elem[0]
                for old_elem in old_items_tuple: 
                    if old_item in old_elem:
                        new_items_id.append(old_elem[1])
                        modified_ids.append(old_elem[1])
                        found_flag=1
                        break
            if found_flag==1:
                break
        if found_flag == 1:
            continue
        else: 
            for old_elem in old_items_tuple:
                if item in old_elem:
                    new_items_id.append(old_elem[1])
                    break
        
    return new_items_id, added_ids, deleted_ids, modified_ids


def manage_different_section_number(prev_items,prev_ids,curr_items,page_id,num_rev):
    flatten_prev_items = [item for sublist in prev_items for item in sublist]
    flatten_prev_ids = [item for sublist in prev_ids for item in sublist]
    zipped_tuple = list(zip(flatten_prev_items,flatten_prev_ids))
    new_ids_list=[]
    deletedids = []
    modids = []
    addedids = []
    flatten_curr_items = [item for sublist in curr_items for item in sublist]
    modifications = compare_lists(flatten_prev_items,flatten_curr_items)
    j=0

    for removed_item in modifications['deleted']:
        for elem in zipped_tuple:
            if removed_item == elem[0]:
                deletedids.append(elem[1])
                break

    for secid,sec_items in enumerate(curr_items):
        new_ids_sec = []
        for item in sec_items:
            found_flag=0
            if item in modifications['unchanged']:
                for elem in zipped_tuple:
                    if item == elem[0]:
                        new_ids_sec.append(elem[1])
                        break
                continue

            elif item in modifications['added']:
                num_comment = str(j).zfill(4)
                item_id = page_id+'_'+f'{num_rev}'+'_'+f'{secid}'+'_'+f'{num_comment}'
                new_ids_sec.append(item_id)
                addedids.append(item_id)
                j+=1
                continue

            else:
                for mod_tuple in modifications['modified']:
                    if item in mod_tuple:
                        for elem in zipped_tuple:
                            if mod_tuple[0] in elem:
                                new_ids_sec.append(elem[1])
                                modids.append(elem[1])
                                found_flag=1
                                break
                    if found_flag==1:
                        break

        new_ids_list.append(new_ids_sec)
            
            

    return new_ids_list,curr_items,addedids,deletedids,modids,modifications

def get_revisions_with_proprieties(dir_files,num_rev,num_users):
    title_more_than_10_rev=[]
    for file in dir_files:
        rev_data = pd.read_csv(file,index_col=0)
        title = file.replace('data/argumentation/revisions_data/','').replace('.csv','')
        num_users = len(rev_data['user'].unique())
        
        if len(rev_data)>=num_rev and num_users>=num_users:
            title_more_than_10_rev.append(title)
            #print("more than 10 revisions and 5 users involved")
            #print(f"creation date: {converted_starting_date}")
    return title_more_than_10_rev


def obtain_correct_sections(code,title,timestamp):

    if title == EXCEPTION_TITLES[0]:
        timestamps=convert_timestamps([timestamp,NOTABLE_TIMESTAMP_OMNIVORES])
        if timestamps[0]<timestamps[1]:
            return code.get_sections(levels=[2])[:2]

    curr_sections = code.get_sections(levels=[3])
    #levels=3
    if len(curr_sections)==0:
        #levels=2
        new_sections = code.get_sections(levels=[2])
        if len(new_sections) in range(3,5):
            curr_sections = new_sections[:2]
        elif len(new_sections) >=5:
            curr_sections = new_sections[:-2]
        else: 
            curr_sections = new_sections[0:1]
    return curr_sections

def manage_votes_presence(code):
    #Manage presence of Votes section
    found_votes_flag=0
    found_see_also_flag=0
    found_index_see_also=0
    found_index_votes=0
    for head in code.filter_headings():
        if 'see also' in head.casefold():
            print("FOUND SEE ALSO SECTION")
            found_see_also_flag=1
            sections = code.get_sections(levels=[2])
            for j,sect in enumerate(sections):
                if sect.nodes[0]==head:
                    index_see_also=j
                    found_index_see_also=1
        if 'votes' in head.casefold() or 'vote' in head.casefold():
            print("FOUND VOTES")
            found_votes_flag=1
            sections = code.get_sections(levels=[2])
            for j,sect in enumerate(sections):
                if sect.nodes[0]==head:
                    index_votes=j
                    found_index_votes=1
    if found_votes_flag==1 and found_index_votes==1:
        sections.pop(index_votes)
        content = [str(sec) for sec in sections]
        to_parse = ''
        for elem in content:
            to_parse+=elem
        code = mwparserfromhell.parse(to_parse)
    if found_see_also_flag==1 and found_index_see_also==1:
        sections.pop(index_see_also)
        content = [str(sec) for sec in sections]
        to_parse = ''
        for elem in content:
            to_parse+=elem
        code = mwparserfromhell.parse(to_parse)
    else: 
        print("no votes found")
    return code


def get_sections_items(code,title,timestamp):
    sentences=[]

    curr_sections = obtain_correct_sections(code,title,timestamp)
    for section in curr_sections:
        sec_sentences = []
        count=0
        item=" "
        for ch in section.strip_code():
            if ch=='\n':
                if count>0:
                    if item.lstrip().startswith("Category:"):
                        item=''
                        continue
                    if len(item)>3:
                        sec_sentences.append(item.lstrip())
                    item=" "
                    continue
                else:
                    item=" "
                    count+=1
                    continue
            item+=ch
        if item.lstrip().startswith("Category:"):
            item=''
            continue
        sec_sentences.append(item.lstrip())
        sentences.append(sec_sentences)
        
    if len(sentences)!=len(curr_sections):
        print("something wrong with sections items")
    if len(sentences)==0:
        print("ERROR: NO SECTIONS FOUND")

    return curr_sections,sentences
