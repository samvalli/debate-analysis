import mwparserfromhell
import pandas as pd
import requests
import math
import os

API_URL = 'https://en.wikiversity.org/w/api.php'

def return_page_rev_dict(rev_data,keys,page_revision_dict):
    for key in keys:
        if key == 'content':
            if '*' not in rev_data['slots']['main']:
                print("no conetnt found for this revision")
                page_revision_dict[key].append('')
            else:
                page_revision_dict[key].append(rev_data['slots']['main']['*'])
        else:
            if key not in rev_data:
                page_revision_dict[key].append(0)
            elif key=='minor' or key=='anon':
                page_revision_dict[key].append(1)
            else:
                page_revision_dict[key].append(rev_data[key])
    return page_revision_dict

def collect_revisions_full_data(titles):
    users_dict={}
    users_page_id_dict={}
    users_m_flags_dict={}
    users_anon_flag={}
    users_page_id_dict_repeted={}
    keys=['revid','parentid','user','timestamp','size','content','comment','minor','tags','anon']
    path = 'data/argumentation/revisions_data/'
    for title in titles:
        if '/' in title:
            print("found unacceptable characther in title")
            file_name=title.replace('/','_')
        else:
            file_name=title
        _,_,_,_,page_dict = request_page_revisions_infos(title,users_dict,users_page_id_dict,users_m_flags_dict,users_anon_flag,users_page_id_dict_repeted)
        test_len= len(page_dict['revid'])
        for key in keys:
            if len(page_dict[key])!=test_len:
                print(f"error in {title}")
                continue
        page_df = pd.DataFrame.from_dict(page_dict)
        page_csv = page_df.to_csv(path+file_name+'.csv')
    return page_csv,file_name

def request_page_revisions_infos(TITLE,users,users_page_id,users_m_flags,users_anon_flag, user_page_id_repeted,LANG='en'):
    global N_RESPONSE_EXCEPTIONS
    
    session = requests.Session()

        
    PARAMS = {
        "action": "query",
        "prop": "revisions",
        "format": "json",
        "titles": TITLE,
        "rvprop": "ids|timestamp|user|userid|content|flags|tags|size|contentmodel|comment",
        "rvslots": "main",
        "rvdir": "newer", 
        "rvlimit":"max",
        # "formatversion": "2",
        "continue": ""

    }
    
    N_RV=0
    N_users=0
    hiddenusers = 0
    N_minor=0
    page_users={}
    keys=['revid','parentid','user','timestamp','size','content','comment','minor','tags','anon']
    page_revisions_dict = {
        'revid':[],
        'parentid':[],
        'user':[],
        'timestamp':[],
        'size':[],
        'content':[],
        'comment':[],
        'minor':[],
        'tags':[],
        'anon':[],
    }

    while True:
        
        while True:
            _exception_counter=0
            try:
                _response = session.get(url=API_URL, params=PARAMS, timeout=60*10)
                break
            except:
                _exception_counter+=1
                N_RESPONSE_EXCEPTIONS+=1
                if _exception_counter==10:
                    return 0
                sleep(10)
        
        response_data = _response.json()
        response_pages = response_data["query"]["pages"]
        
        for page_id, page_data in response_pages.items():
            if not 'revisions' in page_data:
                print(f'revisions not found in response data for page {TITLE}({LANG}); response data:\n\n')
                print(response_data)
                return 0
            revisions = page_data['revisions']         
            N_RV+=len(revisions)
            print(f"page {TITLE} found {N_RV} revisions", end=' '*15+'\r')
            for rev_data in revisions:
                if not 'user' in rev_data:
                    hiddenusers+=1
                    continue
                if rev_data["user"] not in page_users:
                    #users[rev_data["user"]]= 1
                    page_users[rev_data["user"]]=1
                    if rev_data["user"] in users:
                        users[rev_data["user"]] += 1
                        users_page_id[rev_data["user"]].append(int(page_id))
                        user_page_id_repeted[rev_data['user']].append(int(page_id))
                    else:
                        users[rev_data["user"]] = 1
                        users_page_id[rev_data["user"]] = [int(page_id)]
                        user_page_id_repeted[rev_data['user']] = [int(page_id)]
                    N_users+=1
                else:
                    users[rev_data["user"]] += 1
                    user_page_id_repeted[rev_data['user']].append((int(page_id)))
                if 'minor' in rev_data:
                    N_minor+=1
                    if rev_data['user'] in users_m_flags:
                        users_m_flags[rev_data['user']]+=1
                    else: 
                        users_m_flags[rev_data['user']]=1
                if rev_data['user'] not in users_anon_flag:
                    if 'anon' in rev_data:
                        users_anon_flag[rev_data['user']]=True
                    else:
                        users_anon_flag[rev_data['user']]=False
                
                page_revisions_dict = return_page_rev_dict(rev_data,keys,page_revisions_dict)

        if 'continue' in response_data:
            PARAMS['continue'] = response_data['continue']['continue']
            PARAMS['rvcontinue'] = response_data['continue']['rvcontinue']
        
        else:
            return N_RV, N_users, N_minor, hiddenusers, page_revisions_dict

def get_all_wikidebates_revisions_infos(titles):
    num_revisions=[]
    num_users=[]
    num_minor_revisions=[]
    num_hiddenusers=[]
    users_dict={}
    users_page_id_dict={}
    users_m_flags_dict={}
    users_anon_flag={}
    users_page_id_dict_repeted={}
    for title in titles:
        N_RV, N_users, N_minor, hiddenusers = request_page_revisions_infos_mod(title,users_dict,users_page_id_dict,users_m_flags_dict,users_anon_flag, users_page_id_dict_repeted)
        num_revisions.append(N_RV)
        num_users.append(N_users)
        num_minor_revisions.append(N_minor)
        num_hiddenusers.append(hiddenusers)
    return num_revisions,num_users,num_minor_revisions,num_hiddenusers,users_dict,users_page_id_dict,users_m_flags_dict,users_anon_flag,users_page_id_dict_repeted