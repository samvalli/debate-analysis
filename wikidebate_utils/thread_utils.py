import pandas as pd
import glob
from arguemntative_utils.print_tree_utils import *
import json
import ast

def separate_threads_indexes(data):
    level_one_items = data[data['level']==1]
    indexes=level_one_items.index.tolist()
    return indexes



def search_parent(subthread,level):
    if subthread.empty:
        return 0
    for i,row in subthread.iterrows():
        if row['level']<level:
            parent_id = row['id']
    return parent_id



def assign_parent_id(thread):
    thread=thread.reset_index(drop=True)
    parent_id=[]
    #max_depth=thread['level'].max()
    for i,row in thread.iterrows():
        level = row['level']
        parent_id.append(search_parent(thread.iloc[:i],level))
    return parent_id



def assign_parent_and_thread_id_to_data(data,page_id):    
    page_parent_id=[]
    thread_ids =[]
    indexes = separate_threads_indexes(data)
    for i,indx in enumerate(indexes):
        if i ==len(indexes)-1:
            thread = data.iloc[indexes[i]:]
        else:
            thread = data.iloc[indexes[i]:indexes[i+1]]
        page_parent_id+=assign_parent_id(thread)
        thread_id = str(page_id) +'_'+str(i+1).zfill(4)
        thread_ids += [thread_id]*len(thread)
    data = data.assign(parent_id = page_parent_id,thread_id=thread_ids)
    return data



def find_thread_path(thread,parent_id,comparison_key,level,ids_dict):
    #level_string = '='*level
    #print(level_string)
    #print(parent_id)
    connected_items = thread[thread['parent_id']==parent_id]
    #print(thread[['id','parent_id','item']])
    #print(connected_items[['id','item']])
    if connected_items.empty:
        return 
    baseline = connected_items[comparison_key].mean()
    if level==1 or len(connected_items)==1:
        above_baseline_items = connected_items[connected_items[comparison_key]>=baseline]
    else:
        above_baseline_items = connected_items[connected_items[comparison_key]>baseline]
    
    for i,row in above_baseline_items.iterrows():
        curr_id = row['id']
        #print(f"parent_id: {parent_id}, id: {curr_id}, item {row['item']}, level: {level} ")
        ids_dict['id'].append(curr_id)
        #path.append((parent_id,curr_id,level,row['item']))
        find_thread_path(thread,curr_id,comparison_key,level+1,ids_dict)

def obtain_scores_sub_threads(thread,parent_id,comparison_key,level,ids_dict,scores):
    connected_items = thread[thread['parent_id']==parent_id]
    if connected_items.empty:
        return 
    mean = connected_items[comparison_key].mean()
    std = connected_items[comparison_key].std()
    total = connected_items[comparison_key].sum()
    
    for i,row in connected_items.iterrows():
        curr_id = row['id']
        ids_dict['id'].append(curr_id)
        if std>0.0:
            ids_dict[scores[0]].append((row[comparison_key]-mean)/std)
        else: 
            ids_dict[scores[0]].append(0.0)
        if total>0.0:
            ids_dict[scores[1]].append(row[comparison_key]/total)
        else:
            ids_dict[scores[1]].append(0.0)
        ids_dict[scores[2]].append(len(connected_items))
        obtain_scores_sub_threads(thread,curr_id,comparison_key,level+1,ids_dict,scores)




def get_sourcing_and_modificatioins_thread_infos(thread):
    thread = thread.reset_index()
    thread_modifications=[]
    thread_sourcing=[]
    for i,row in thread.iterrows():
        modifications=0
        sourcing = 0
        curr_level=row['level']
        if i==len(thread)-1:
            thread_modifications.append(row['num modifications'])
            thread_sourcing.append(row['references'])
            continue
        for i,row in thread[i:].iterrows():
            below_levels=row['level']
            if below_levels>curr_level:
                modifications+=row['num modifications']
                sourcing+=row['references']
        thread_modifications.append(modifications)
        thread_sourcing.append(sourcing)
    
    return thread_modifications,thread_sourcing


def assing_sourcing_and_modifications_thread_infos():
    dir_path = 'data/argumentation/items survivors infos'
    dir_files=sorted(glob.glob(dir_path+'/*'))
    for dir_file in dir_files:
        page_modifications_data=[]
        page_sourcing_data=[]
        data=pd.read_csv(dir_file,index_col=0)
        thread_ids = data['thread_id'].unique()
        for thread_id in thread_ids:
            thread = data[data['thread_id']==thread_id]
            thread_modifications_num,thread_sourcing_num = get_sourcing_and_modificatioins_thread_infos(thread=thread)
            page_modifications_data+=thread_modifications_num
            page_sourcing_data+=thread_sourcing_num
        data=data.assign(thread_modifications = page_modifications_data,thread_sourcing=page_sourcing_data)
        data_csv = data.to_csv(dir_file)
    return


def get_above_baseline_sub_threads(all_debates_data):
    dir_path = 'data/argumentation/items survivors infos'
    titles=[]

    dir_files=sorted(glob.glob(dir_path+'/*'))
    keys = ['thred_items','thread_modifications','thread_sourcing']
    page_keys = ['num_objections','num_modifications','num_items_referenced']
    for (dir_file,(i,page_data)) in zip(dir_files[-4:-3],all_debates_data[-4:-3].iterrows()):
        titles.append(page_data['title'])
        page_data['num_modifications']=page_data['avg modifications']*page_data['num_items']
        #print(page_data['num_modifications'])
        print(f"new debate: {titles[-1]}")
        for j,comparison_key in enumerate(keys):
            ids_objects={'id':[],
                        'z_score':[],
                        'proportion':[]}
            data=pd.read_csv(dir_file,index_col=0)
            page_baseline = page_data[page_keys[j]]/page_data['num_arguments']
            #print(page_baseline)
            thread_ids = data['thread_id'].unique()
            for thread_id in thread_ids:
                thread = data[data['thread_id']==thread_id]
                if thread[comparison_key].tolist()[0]>page_baseline:
                    #print(thread[comparison_key].tolist()[0])
                    obtain_scores_sub_threads(thread,'0',comparison_key,1,ids_objects)
            relevant_items = data[data['id'].isin(ids_objects['ids'])]
            if j==0:
                print("full data")
                print_tree(data,'',titles[-1])
                print('\n\n')
            print(f"{comparison_key}")
            print_tree(relevant_items,comparison_key,titles[-1])
            print('\n\n')
    return 

def assign_scores_to_data(data, dict_objects,j):
    # Convert dict to DataFrame and set 'id' as the index
    scores_df = pd.DataFrame(dict_objects)
    if j>0:
       scores_df.pop('width')
    # Set 'id' as the index in the original DataFrame for efficient updating
    # Convert dict to DataFrame and set 'id' as the index
    data = pd.merge(data,scores_df,on='id')
    
    return data


def check_and_remove_doubled(all_debates_data):
    dir_path = 'data/argumentation/items survivors infos'
    titles=[]

    dir_files=sorted(glob.glob(dir_path+'/*'))
    for (dir_file,(i,page_data)) in zip(dir_files,all_debates_data.iterrows()):
        titles.append(page_data['title'])
        print(titles[-1],i)
        page_data['num_modifications']=page_data['avg modifications']*page_data['num_items']
        page_id=page_data['page_id']
        data=pd.read_csv(dir_file,index_col=0)
        list_doubled=data[data.duplicated(subset='id', keep=False)]
        if len(list_doubled)>0:
            print("Found doubled items in the debate")
            count=list_doubled.groupby('id').cumcount()
            for indx,doubles in list_doubled.iterrows():
                prev_id = doubles['id']
                prev_num = prev_id[-4:]
                new_num = int(prev_num) + count[indx]
                new_id = prev_id.replace(prev_num,str(new_num).zfill(4))
                data.loc[indx, "id"]=new_id
                print(data)
            data=assign_parent_and_thread_id_to_data(data,page_id)
            data.to_csv(dir_file)
    return 


def create_scores_dataset(all_debates_data,print_tree=False):
    dir_path = 'data/argumentation/items survivors infos'
    titles=[]

    dir_files=sorted(glob.glob(dir_path+'/*'))
    keys = ['thred_items','thread_modifications','thread_sourcing']
    short_keys = ['num_it','num_mod','num_ref']
    page_keys = ['num_objections','num_modifications','num_items_referenced']
    for (dir_file,(i,page_data)) in zip(dir_files,all_debates_data.iterrows()):
        titles.append(page_data['title'])
        ##ERROR WITH PAGE DATA
        ####page_data['num_modifications']=page_data['avg modifications']*page_data['num_items']
        data=pd.read_csv(dir_file,index_col=0)
        #print(page_data['num_modifications'])
        print(f"new debate: {titles[-1]}")
        for j,comparison_key in enumerate(keys):
            dict_keys = ['id','z_score'+'_'+short_keys[j],'proportion'+'_'+short_keys[j],'width']
            dict_objects={
                'id':[],
                'z_score'+'_'+short_keys[j]:[],
                'proportion'+'_'+short_keys[j]:[],
                'width':[]
            }
            obtain_scores_sub_threads(data,'0',comparison_key,1,dict_objects,dict_keys[1:])
            data = assign_scores_to_data(data,dict_objects,j)
            
        data.to_csv(f'data/argumentation/items_score_df/{titles[-1]}.csv')
        if print_tree==True:
            print(f"{comparison_key}")
            print_tree_score(data,'thred_items','thred_items',titles[-1])
            print('\n\n')
    return 


def assign_page_infos(all_threads_df,all_debates_data):
    total_number_of_users = []
    total_number_of_items=[]
    for i,row in all_threads_df.iterrows():
        page_data = all_debates_data[all_debates_data['page_id']==row['page_id']]
        total_number_of_users.append(page_data['number of users'].item())
        total_number_of_items.append(page_data['num_items'].item())

    page_data_all_items = all_threads_df.assign(number_of_page_users=total_number_of_users,number_of_page_items=total_number_of_items)
    return page_data_all_items

def assign_level_wise_z_scores(all_threads_df):
    mean_it=[]
    std_it=[]
    mean_mod=[]
    std_mod=[]
    z_scores_level_based_it=[]
    z_scores_level_based_mod=[]

    for level in all_threads_df['level'].unique():
        level_df = all_threads_df[all_threads_df['level']==level]
        mean_it.append(level_df['thred_items'].mean())
        std_it.append(level_df['thred_items'].std())
        mean_mod.append(level_df['thread_modifications'].mean())
        std_mod.append(level_df['thread_modifications'].std())

    for i,row in all_threads_df.iterrows():
        #print("================== new item ============")
        level=row['level']
        j=level-1
        # print(level)
        # print(mean_it[j])
        # print(row['thred_items'])
        z_scores_level_based_it.append((row['thred_items']-mean_it[j])/std_it[j])
        z_scores_level_based_mod.append((row['thread_modifications']-mean_mod[j])/std_mod[j])

    level_wise_z_scores=all_threads_df.assign(z_scores_level_it=z_scores_level_based_it,z_scores_level_mod=z_scores_level_based_mod)
    return level_wise_z_scores

def assign_page_wise_z_scores(all_threads_df):
    z_scores_page_based_it=[]
    z_scores_page_based_mod=[]

    for id in all_threads_df['page_id'].unique():
        mean_it=[]
        std_it=[]
        mean_mod=[]
        std_mod=[]
        #print("================== new item ============")

        page_items = all_threads_df[all_threads_df['page_id']==id]
        for level in page_items['level'].unique():
            level_df = page_items[all_threads_df['level']==level]
            mean_it.append(level_df['thred_items'].mean())
            std_it.append(level_df['thred_items'].std())
            mean_mod.append(level_df['thread_modifications'].mean())
            std_mod.append(level_df['thread_modifications'].std())

        for i,row in page_items.iterrows():
            level=row['level']
            j=level-1

            z_scores_page_based_it.append((row['thred_items']-mean_it[j])/std_it[j])
            z_scores_page_based_mod.append((row['thread_modifications']-mean_mod[j])/std_mod[j])

    page_wise_z_scores=all_threads_df.assign(z_scores_page_it=z_scores_page_based_it,z_scores_page_mod=z_scores_page_based_mod)
    return page_wise_z_scores


def collect_path_scores(node, path,author_path,lenght,page_score_path,branch_score_path,width,data,node_dict,path_scores,page_id,title):
    # Add current node to the path and add its z_score to the cumulative score
    current_path = path + [node]
    num_mod_score = data.loc[data['id'] == node, 'z_scores_page_it'].values[0]
    num_items_score = data.loc[data['id']==node, 'z_scores_page_mod'].values[0]
    branch_mod_score = data.loc[data['id'] == node, 'z_score_num_it'].values[0]
    branch_items_score = data.loc[data['id']==node, 'z_score_num_mod'].values[0]
    aut=data[data['id']==node]['authors'].apply(ast.literal_eval).values[0]
    if data.loc[data['id']==node, 'level'].values[0] == 1:
        width_level = 0
    else:
        width_level = data.loc[data['id']==node, 'width'].values[0]
    current_author_path = author_path + aut
    curr_page_score_path = page_score_path + [(num_mod_score,num_items_score)]
    curr_branch_score_path = branch_score_path + [(branch_mod_score,branch_items_score)]
    curr_lenght = lenght + 1
    curr_width = width + width_level
    # Check if the node has children
    if node in node_dict:
        # Recurse for each child node
        for child in node_dict[node]:
            collect_path_scores(child, current_path,current_author_path,curr_lenght, curr_page_score_path, curr_branch_score_path,curr_width,data,node_dict,path_scores,page_id,title)
    else:
        # If no children, store the final path and cumulative score
        id_path = str(page_id)+'_'+current_path[0][5:]+'_'+current_path[-1][5:]
        num_useres = len(list(set(current_author_path)))
        path_scores[id_path] = (current_path,curr_page_score_path,curr_branch_score_path,current_author_path,curr_lenght,curr_width/curr_lenght,num_useres,page_id,title)

def create_dataset_with_page_scores():
    threads_data = pd.read_csv('data/argumentation/page_wise_score.csv',index_col=0)
    debates_data = pd.read_csv('data/global/relevant_debates_31_10_2024.csv',index_col=0)
    page_ids = threads_data['page_id'].unique()
    titles=[]
    for page_id in page_ids:
        data=threads_data[threads_data['page_id']==page_id]
        title = debates_data[debates_data['page_id']==page_id]['title'].item()
        titles.append(title)
        print(f"========= new debate: {title} ========")
        child_dict = data.groupby('parent_id')['id'].apply(list).to_dict()
        root_nodes = data[data['parent_id']=='0']['id'].values
        path_scores={}
        for root in root_nodes:
            collect_path_scores(root, [],[],lenght=0,page_score_path=[],branch_score_path=[],width=0,data=data,node_dict=child_dict,path_scores=path_scores,page_id=page_id,title=titles[-1])
        json_path=f'data/argumentation/paths_data/page_wise_path/{titles[-1]}.json'
        with open(json_path, 'w') as file:
            json.dump(path_scores, file, indent=4, default=str)
            file.write('\n')
        
    dir_path_paths = 'data/argumentation/paths_data/page_wise_path'
    dir_files=sorted(glob.glob(dir_path_paths+'/*'))
    columns = ['path','page_z_scores_path','branch_z_scores_path','authors','lenght','mean_width','num_unique_users','page_id','title']
    first_data=True
    for dir in dir_files:
        with open(dir,'r') as file:
            data=json.load(file)
        df = pd.DataFrame.from_dict(data,orient='index',columns=columns)
        if first_data==True:
            gloabl_path_data = pd.DataFrame(columns=columns)
            first_data=False
        gloabl_path_data = pd.concat([gloabl_path_data,df],ignore_index=True)
    gloabl_path_data.to_csv('data/argumentation/global_page_paths_data.csv')
    
    return 

def assign_degree(data):
    degree_list=[]
    for i,row in data.iterrows():
        num_child = len(data[data['parent_id']==row['id']])
        degree_list.append(num_child)
    return degree_list