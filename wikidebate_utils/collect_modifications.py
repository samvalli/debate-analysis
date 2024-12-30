from arguemntative_utils.survivors_utils import *
from global_utils.get_wikidebate_revisions_infos import *
import json
import pandas as pd
import numpy as np
import glob
import ast

def build_modfications_dict(titles):
    date_zero='2018-08-01T00:00:00Z'
    votes_start = '2018-08-01T00:00:00Z'
    votes_end = '2024-10-01T00:00:00Z'
    votes_dates = convert_timestamps([votes_start,votes_end])
    keys = ['revid','parentid','itemsids','addedids','modifiedids','delatedids','items','moditems','delateditems','timedist','author','author_diff','minor','timestamp']
    survivors_dict = {}
    for key in keys:
        survivors_dict[key]=[]

    for num_page,title in enumerate(titles):
        count_skipped_rev=0
        survivors_dict = {}
        for key in keys:
            survivors_dict[key]=[]
        print(f" =========== new debate : {title} =============")
        page_id = 1000+num_page
        page_id = str(page_id)
        first_rev = True
        count_minor_rev=0
        prev_size=0
        page_revisions_data = pd.read_csv(f'data/argumentation/revisions_data/{title}.csv',index_col=0)
        page_revisions_data = page_revisions_data[page_revisions_data['content'].notna()].reset_index(drop=True)
        for k,data in page_revisions_data.iterrows():
            print(f"revision number {k}")
            
            #Initialize necessary arrays
            curr_items_ids = []       
            modids = []
            deletedids=[]
            moditems=[]
            deleteditems=[]
            curr_items=[]
            curr_timestamp = data['timestamp']
            minor_flag=0
            
            compare_timestamps = convert_timestamps([curr_timestamp,date_zero])
            if compare_timestamps[0]<compare_timestamps[1]:
                prev_size=data['size']
                prev_user = data['user']
                print("before date zero")
                continue
            
            curr_revid = data['revid']
            curr_parrentid = data['parentid']
            curr_user = data['user']
            
            
            curr_size = data['size']
            curr_content = data['content']
            curr_comment = data['comment']
            curr_minor_flag = data['minor']
            curr_tags = data['tags']
            curr_anon_flag = data['anon']
            size_diff=(curr_size-prev_size)
            curr_code = mwparserfromhell.parse(curr_content)
            print(curr_timestamp)
            print(f"size: {curr_size}")

            #Check votes presence
            if votes_dates[0]<compare_timestamps[0]<votes_dates[1]:
                curr_code = manage_votes_presence(curr_code)
            
            #Detect vandalism
            if first_rev==False and prev_size!=0 and size_diff/prev_size<=-0.5:
                print("DETECTED POSSIBLE VANDALISM: skip this revision")
                count_skipped_rev+=1
                prev_size=curr_size
                prev_timestamp = curr_timestamp
                prev_items = survivors_dict['items'][-1]
                continue

            #Manage minor revisons
            if curr_minor_flag==1:
                if first_rev==True:
                    print("first rev is a minor rev")
                    continue 
                #prev_items = []
                print("=============MINOR REVISION============")
                count_minor_rev+=1
                minor_flag=1
                #prev_size=curr_size
                #prev_timestamp = curr_timestamp
                
                #curr_sections,prev_items = get_sections_items(curr_code,title,curr_timestamp)
            
                #For those cases where the strucutre of the debate is changed in a minor revision
                # if len(curr_sections)!=len(survivors_dict['itemsids'][-1]):
                #     prev_items = survivors_dict['items'][-1]
                #     print("changed structure: ignore revision")
        
                #continue

            # Manage first revision
            if first_rev==True or len(prev_items)==0:
                
                #Get current sections and current items
                curr_sections,curr_items = get_sections_items(curr_code,title,curr_timestamp)
                #Check if there are any items
                if len(curr_sections)>0:
                    total_n_items=0
                    for sec_items in curr_items:
                        total_n_items+=len(sec_items)
                if total_n_items==0:
                    print("ERROR: NO ITEMS FOUND")

                #Collect all the ids from the sections
                for i,sec in enumerate(curr_items):
                    secid=i
                    sec_ids=[]
                    for j,elem in enumerate(sec):
                        num_comment = str(j).zfill(4)
                        sec_ids.append(page_id+'_'+f'{k}'+'_'+f'{secid}'+'_'+f'{num_comment}')
                    curr_items_ids.append(sec_ids)

                added_ids = [item for sublist in curr_items_ids for item in sublist]
            
                cleaned_timestamps = convert_timestamps([curr_timestamp,curr_timestamp])
                timedist=cleaned_timestamps[1]-cleaned_timestamps[0]
                user_diff=0
                if first_rev==False:
                    if prev_user==curr_user:
                        user_diff=1
                first_rev=False

                curr_dict_items = [curr_revid,curr_parrentid,curr_items_ids,added_ids,modids,deletedids,curr_items,moditems,deleteditems,timedist,curr_user,user_diff,minor_flag,curr_timestamp]
                for (key,item) in zip(keys,curr_dict_items):
                    survivors_dict[key].append(item)


            #Manage other revisions
            else:
                #Initialize addedids array to collect ids of added items
                added_ids=[]
                ##Check if user changed
                user_diff=0
                if curr_user==prev_user:
                    user_diff=1
                #Calculate timedistance
                cleaned_timestamps = convert_timestamps([prev_timestamp,curr_timestamp])
                timedist = cleaned_timestamps[1]-cleaned_timestamps[0]

                #Get current sections and items
                curr_sections,curr_items = get_sections_items(curr_code,title,curr_timestamp)
    
                #Check if there are any items
                if len(curr_sections)>0:
                    total_n_items=0
                    for sec_items in curr_items:
                        total_n_items+=len(sec_items)
                if total_n_items==0:
                    print("ERROR: NO ITEMS FOUND")

                #If the number of sections is mantained equal to the previous revision's
                if len(prev_items)==len(curr_sections):
                    for i,(curr_item,prev_item) in enumerate(zip(curr_items,prev_items)):
                        prev_items_ids=survivors_dict['itemsids'][-1][i]
                        curr_sec_items = curr_item
                        modifications = compare_lists(prev_item,curr_sec_items)
                        #CHIAMATA ALLA FUNZIONE PER COLLEZIONARE GLI IDS
                        new_items_id_sec, added_ids_sec, deleted_ids_sec, modified_ids_sec = obtain_correct_ids(curr_sec_items,prev_item,modifications,prev_items_ids,page_id,k,i)
                        #print(f"{count_added}, count mod: {count_modifications}, count_removed: {count_removed}")
                        added_ids+=added_ids_sec
                        deletedids+=deleted_ids_sec
                        modids += modified_ids_sec
                        curr_items_ids.append(new_items_id_sec)
                        moditems += modifications['modified']
                        deleteditems += modifications['deleted']

                #If the number of sections is different between revisions
                if len(prev_items)!=len(curr_sections):
                    print("number of sections changed")
                    prev_items_ids = survivors_dict['itemsids'][-1]
                    curr_items_ids,curr_items,added_ids,deletedids,modids,modifications = manage_different_section_number(prev_items,prev_items_ids,curr_items,page_id,k)
                    
                curr_dict_items = [curr_revid,curr_parrentid,curr_items_ids,added_ids,modids,deletedids,curr_items,moditems,deleteditems,timedist,curr_user,user_diff,minor_flag,curr_timestamp]
                for (key,item) in zip(keys,curr_dict_items):
                    survivors_dict[key].append(item)
            
            #Sanity check
            print(len(curr_sections))
            print("items lenght")
            for con,sec_items in enumerate(curr_items):
                print(f"section {con} items: {len(sec_items)}")
            if len(curr_items)!=len(curr_sections):
                print("WARNING: number of sections differs from number of items sections")
            if len(curr_items_ids)!=len(curr_sections):
                print("WARNING: number of sections differs from number of ids sections")
            if len(curr_items_ids)!=len(curr_items):
                print("WARNING: number of ids sections differs from number of items sections")
            for l,(sec_it, sec_id) in enumerate(zip(curr_items,curr_items_ids)):
                if len(sec_it)!=len(sec_id):
                    print(f"WARNING: num of elements in section {l} doesn't match")

            prev_size=curr_size
            prev_code = curr_code
            prev_items = curr_items
            prev_user = curr_user
            prev_timestamp = curr_timestamp
        
        json_path=f'data/argumentation/survivors_data/{title}.json'
        with open(json_path, 'w') as file:
            json.dump(survivors_dict, file, indent=4, default=str)
            file.write('\n')
    
    return 

def create_survivors_df(flat_ids,flat_items,flat_num_rev_survived,num_modifications,mod_ratio,num_diff_contributors,authors):
    survivors_dict = {
        'id':flat_ids,
        'item':flat_items,
        'num rev survived': flat_num_rev_survived,
        'num modifications':num_modifications,
        'avg mod ratio': mod_ratio,
        'num unique contributors': num_diff_contributors,
        'authors': authors
    }
    survivors_df = pd.DataFrame(survivors_dict,dtype=object)
    return survivors_df


def get_number_and_avg_ratio_of_modifications(data):
    num_modifications = []
    mod_ratio = []
    num_contributors = []
    diff_authors=[]
    
    flat_ids = [id for sublist in data['itemsids'][-1] for id in sublist]
    for id in flat_ids:
        find_creator = True
        authors = []
        num_contrib_item = 1
        count_mod=0
        mod_ratio_value=0
        mod_ratio_count=0
        
        for (rev,mod,author,add) in zip(data['modifiedids'],data['moditems'],data['author'],data['addedids']):
            if len(rev)!=len(mod):
                continue
            if find_creator==True:
                if id in add:
                    authors.append(author)
            if id in rev:
                index = rev.index(id)
                count_mod+=1
                mod_ratio_count+=1
                mod_ratio_value+=mod[index][2]
                if author not in authors:
                    num_contrib_item+=1
                    authors.append(author)
        num_contributors.append(num_contrib_item)
        diff_authors.append(authors)
        if mod_ratio_count>0:
            mod_ratio.append(mod_ratio_value/mod_ratio_count)
        else:
            mod_ratio.append(0.0)
        num_modifications.append(count_mod)
    return num_modifications,mod_ratio,num_contributors,diff_authors

def extract_rev_number(items_ids):
    first_rev_number = []
    for rev in items_ids:
        sec_list=[]
        for sect in rev:
            items_list=[]
            for item in sect:
                rev_num = item[5:]
                index_ = rev_num.find('_')
                rev_num=int(rev_num[:index_])
                items_list.append(rev_num)
            sec_list.append(items_list)
        first_rev_number.append(sec_list)
    return first_rev_number


def build_csv_mod(dir_files):
    percentage_survived=[]
    number_survived = []
    number_revision = []
    number_modifications=[]
    ratio_modifications = []
    percentage_of_relevant_modifications = []
    perc_of_item_modified = []
    titles=[]
    number_of_different_authors=[]

    for dir_file in dir_files:
        with open(dir_file, 'r') as file:
            data = json.load(file)
            titles.append(dir_file.replace('data/argumentation/survivors_data/','').replace('.json',''))
        
        print(f"new debate: {titles[-1]}")
        num_considered_rev=len(data['revid'])
        rev_df = pd.read_csv('data/argumentation/revisions_data/'+titles[-1]+'.csv',index_col=0)
        df_from_first_rev = rev_df[rev_df['revid']>data['revid'][0]]
        num_skipped_rev = len(rev_df)-len(df_from_first_rev)
        first_rev = df_from_first_rev[0:1]
        starting_timestamp=first_rev['timestamp']
        first_rev = extract_rev_number(data['itemsids'])
        num_survived_rev=[]
        print(num_considered_rev)
        for i in range(len(first_rev[-1])):
            num_survived_rev.append([len(rev_df)-x for x in first_rev[-1][i]])
        
        flat_num_rev_survived = [num for sublist in num_survived_rev for num in sublist]
        flat_items = [item for sublist in data['items'][-1] for item in sublist]
        flat_ids = [id for sublist in data['itemsids'][-1] for id in sublist]
        
        number_of_different_authors.append(len(list(set(data['author']))))
        num_modifications,mod_ratio,num_diff_contributors,authors = get_number_and_avg_ratio_of_modifications(data)
        print((num_diff_contributors))
        data_survivors_df = create_survivors_df(flat_ids,flat_items,flat_num_rev_survived,num_modifications,mod_ratio,num_diff_contributors,authors)
        data_survivors_csv = data_survivors_df.to_csv(f'data/argumentation/items survivors infos/{titles[-1]}.csv')
        avg_rev_survived = data_survivors_df['num rev survived'].mean()
        above_zero_ratio_df = data_survivors_df[data_survivors_df['avg mod ratio']>0.0]
        avg_ratio = above_zero_ratio_df['avg mod ratio'].mean()
        num_modified_item = len(above_zero_ratio_df)
        if len(above_zero_ratio_df)==0:
            avg_ratio=1.00000
        number_revision.append(num_considered_rev)
        number_survived.append(avg_rev_survived)
        number_modifications.append(np.mean(num_modifications))
        ratio_modifications.append(np.mean(avg_ratio))
        perc_of_item_modified.append(num_modified_item/len(data_survivors_df))
        percentage_of_relevant_modifications.append(len(data_survivors_df[data_survivors_df['avg mod ratio'].between(0.1,0.85)])/len(data_survivors_df))
        percentage_survived.append(avg_rev_survived/len(data['revid']))

    return 


def updating_routine(title):
    dir_files=[]
    if title[0] == 'Was 9_11 an inside job?':
        title[0]=title[0].replace('_','/')
    csv,title_new = collect_revisions_full_data(title)
    if title[0] == 'Was 9/11 an inside job?':
        title[0]=title_new
    build_modfications_dict(title)
    for tit in title:
        dir_files.append('data/argumentation/survivors_data/'+tit+'.json')
    build_csv_mod(dir_files=dir_files)


def check_number_connected_items(curr_level,below_leves):
    count=0
    for lev in below_leves:
        if lev>curr_level:
            count+=1
        else:
            return count
    return 0

def find_category(page_data):
    category = page_data['hand_labeled_categories'].item()[:15]
    if any(keyword in category for keyword in ['Science','Medicine','Technology']):
        category_int=4
    elif any(keyword in category for keyword in ['Ethics']):
        category_int=2
    elif any(keyword in category for keyword in ['Politics','History']):
        category_int=5
    elif any(keyword in category for keyword in ['Wik']):
        category_int=6
    elif any(keyword in category for keyword in ['Philosophy']):
        category_int=0
    elif any(keyword in category for keyword in ['Religion']):
        category_int=1
    elif any(keyword in category for keyword in ['Economics']):
        category_int=3
    else:
        print("no cat found")
        category_int=7
    return [category_int]

def build_interactions_csv(updating=False):
    dir_path = 'data/argumentation/items survivors infos'
    all_items = pd.read_csv('data/global/all_items_v4.csv',index_col=0)
    pages_data=pd.read_csv('data/global/pages_data_01_10_2024.csv')
    dir_files=sorted(glob.glob(dir_path+'/*'))
    titles = []
    for dir_file in dir_files:
        lenght=[]
        titles.append(dir_file.replace('data/argumentation/items survivors infos/','').replace('.csv',''))
    
        if updating==True:
            #print(f"FOUND AN ERROR in {titles[-1]}")
            updating_routine([titles[-1]])


        data = pd.read_csv(dir_file,index_col=0)
        aut=data['authors'].apply(ast.literal_eval)
        aut = aut.tolist()
        
        print(f"new debate {titles[-1]}")
        if '_' in titles[-1]:
            title_to_find = titles[-1].replace('_','/')
            all_items_page = all_items[all_items['title']==title_to_find]
            page_data=pages_data[pages_data['title']==title_to_find]
        else:
            all_items_page = all_items[all_items['title']==titles[-1]]
            page_data=pages_data[pages_data['title']==titles[-1]]
        all_types = all_items_page['type'].to_list()
        all_levels = all_items_page['level'].to_list()
        flat_items = data['item'].to_list()
        references = all_items_page['references'].to_list()

        category = find_category(page_data)
        categories=category*len(data)
        page_id = page_data['page_id'].item()
        page_ids=[page_id]*len(data)
        
        for i,row in data.iterrows():
            lenght.append(len(row ['item']))
        
        if len(all_levels)!=len(flat_items):
            print("ERROR FOUND")
            continue

        num_items_in_lower_levels=[]
        threads_interactions_authors = []
        num_threads_authors=[]
        for i,row in data.iterrows():
            curr_level = all_levels[i]
            below_items=check_number_connected_items(curr_level,all_levels[i+1:])
            num_items_in_lower_levels.append(below_items)
            involved_authors=aut[i:i+below_items+1]
            flat_aut = list(set([aut_fl for sub in involved_authors for aut_fl in sub]))
            threads_interactions_authors.append(flat_aut)
            num_threads_authors.append(len(flat_aut))
        
        data=data.assign(page_id = page_ids,lenght = lenght,thred_items = num_items_in_lower_levels,type=all_types,level=all_levels, thred_users_number=num_threads_authors,categories=categories,references=references)
        data.to_csv(dir_file)
    return