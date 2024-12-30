import numpy as np
import glob
from mdutils.mdutils import MdUtils
from sentence_transformers import util

def get_cluster_arguments(pairs, threshold=0.75):
    clusters = []
    N = 0
    
    for pair in pairs:
        S1=pair[0]
        S2=pair[1]
        similarity=np.float64(pair[2])
        if similarity >= threshold:
            # Check if S1 and S2 are already in any existing clusters
            S1_cluster = None
            S2_cluster = None
            
            for cluster in clusters:
                if S1 in cluster:
                    S1_cluster = cluster
                if S2 in cluster:
                    S2_cluster = cluster
            
            # Neither S1 nor S2 are in any clusters, create a new cluster
            if S1_cluster is None and S2_cluster is None:
                new_cluster = {S1, S2}
                clusters.append(new_cluster)
                N += 1
            
            # S1 is in a cluster but S2 is not, add S2 to S1's cluster
            elif S1_cluster is not None and S2_cluster is None:
                S1_cluster.add(S2)
            
            # S2 is in a cluster but S1 is not, add S1 to S2's cluster
            elif S2_cluster is not None and S1_cluster is None:
                S2_cluster.add(S1)
            
            # Both S1 and S2 are in different clusters, merge the clusters
            elif S1_cluster is not S2_cluster:
                S1_cluster.update(S2_cluster)
                clusters.remove(S2_cluster)
    
    return N, clusters


def get_clusters(dir_path,folder,treshold=0.75,verbose=False):
    num_clusters=[]
    clustered_arguments=[]
    dir_files=sorted(glob.glob(dir_path+'/'+folder+'/*'))
    #Attualmente sto escludendno l'ultimo dibattito perchè troppo grosso
    for file in dir_files[:10]:
        print("============================ new topic ===============================")
        pairs=np.load(file)
        N,clusters=get_cluster_arguments(pairs,treshold)
        num_clusters.append(N)
        clustered_arguments.append(clusters)
        if verbose:
            for cluster in clusters:
                print("=========== new cluster =============")
                for elem in cluster:
                    print(f"{elem[:50]}...")
    return clustered_arguments, num_clusters


def create_clusters_list_and_write_md(full_corpus,wiki_titles,wiki_debates,model,model_type,write_md,treshold=0.7):
    embeddings=[]
    clusters_list=[]
    flags=[]
    for j,debate in enumerate(full_corpus):
        debate_flags=[]
        print(f"=============== new debate {wiki_titles[j]} ===============")
        if write_md:
            md_title=f'wikidebate:  {wiki_titles[j]}' +'\n'
            mdFile=MdUtils(file_name="clusters " + model_type+' '+ wiki_titles[j], title= md_title)
            mdFile.new_header(level=1, title=f"Obtained clusters with model {model_type[1]}")
        embedding=model.encode(debate)
        clusters=util.community_detection(embedding,min_community_size=2,threshold=treshold)
        for i, cluster in enumerate(clusters):
            cluster_flags=[]
            if write_md:
                mdFile.new_header(level=2, title='Cluster {}, #{} Elements '.format(i+1, len(cluster)))
                mdFile.write('\n')
                mdFile.write('\n')
            #print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
            for sentence_id in cluster:   
                if write_md:
                    if sentence_id<len(wiki_debates[j]):
                        mdFile.write('**'+debate[sentence_id].strip()+'**')
                        cluster_flags.append("wiki")
                    else:
                        mdFile.write(debate[sentence_id])
                        cluster_flags.append("cmv")
                    mdFile.write('\n')
                    mdFile.write('================================================================')
                    mdFile.write('\n')
                    mdFile.write('\n')
            debate_flags.append(cluster_flags)
        flags.append(debate_flags)
        embeddings.append(embedding)
        clusters_list.append(clusters)
        if write_md:
            mdFile.create_md_file()
    return clusters_list,flags

def get_clusters_content(cluster_list,full_content):
    clusters=[]
    for ids in cluster_list:
        sub_cluster=[]
        for id in ids:
            sub_cluster.append(full_content[id])
        clusters.append(sub_cluster)
    return clusters

def obtain_cluster_content(clusters_list,full_corpus):
    clusters_content=[]
    for (clusters_ids,debate) in zip(clusters_list,full_corpus):
        clusters_content.append(get_clusters_content(clusters_ids,debate))
    return clusters_content

def convert_cluster_to_string(cluster):
    usable_query='\n'
    for i,elem in enumerate(cluster):
        elem = elem.replace('\n',' ')
        usable_query+=str(i)+'. '+elem+'\n'
    return usable_query

def clean_output(string):
    #string=summary.choices[0].message.content
    to_be_removed=['\n','\t','{','}','"specific topic":','"sub_topic":','"subtopic":','"sub topic":','"specific_topic":','"topic":','"0":','"']
    for char in to_be_removed:
        string=string.casefold().replace(char,' ')
    return string.strip()


def get_openai_cluster_identification(cluster,topic,client):
  cluster_to_string=convert_cluster_to_string(cluster)
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      response_format={ "type": "json_object" },
      temperature=0.3,
      messages=[
        {"role": "system", "content": f"You are a helpful assistant designed to output very short summary, the starting question is: {topic}, you should be very specific about the sub topic discussed in each cluster of comments JSON."},
        {"role": "user", "content": f"The main topic is given by the starting question:{topic}"+
         """I divided some comments debating about the main topic based on the sub-topic they are debating about. 
         I wish you to identify that one specific sub-topic. 
         The sub-topic should not be the same as the main topic.
         Give me only you best guess using a maximum of 6 words
         These are the comments which share the same sub-topic:"""
         +cluster_to_string+
         'If you encounter comments like "Confirmed: 1 delta awarded to /u" followed by a username, use "reddit reward method" as the sub-topic. \n'+
         'Reaply in JSON format using as standard format "sub_topic": "your best guess"'},
      ]
    )
  return clean_output(response.choices[0].message.content)


def open_ai_cluster_sub_topic_summarization(clusters_content,wiki_titles,client):
    clusters_each_topic_titles=[]
    lenght_each_topic_clusters=[]
    for i,topic in enumerate(clusters_content):
        topic_title=wiki_titles[i]
        print(f"=========== new topic : {topic_title} ==========")
        clusters_titles=[]
        lenght_clusters=[]
        for j,cluster in enumerate(topic):
            clusters_titles.append(get_openai_cluster_identification(cluster,topic_title,client))
            lenght_clusters.append(len(cluster))
        clusters_each_topic_titles.append(clusters_titles)
        lenght_each_topic_clusters.append(lenght_clusters)
    return clusters_each_topic_titles,lenght_each_topic_clusters
