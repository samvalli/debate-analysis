from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# def get_FastText_raw_embeddings(merged_data,platforms,word_vectors,pages,normalization=True,plot_keywords=True,plot_distances=False):    
#     threads_words_dict={}
#     embeddings_matrices=[]
#     platform_matrices=[]
#     documents_matrices=[]
#     representation_2d=[]
#     clusters=[]
#     num_claims=[]
#     for i,page_id in enumerate(merged_data['page_id'].unique().tolist()):
#         threads_words_dict[str(page_id)] = []
#         page_data = merged_data[merged_data['page_id']==page_id]
#         print(f"NEW PAGE {i+1}. {page_data['title'].tolist()[0]}")
#         title=page_data['title'].tolist()[0]
#         page=pages[i]
#         documents=[]
#         level_one = page_data[page_data['level']==1]
#         threads_platforms=[]
#         num_claims_threads=[]
#         for j,platform in enumerate(platforms):
#             plat_data = page_data[page_data['platform']==platform]
#             thread_ids = plat_data['thread_id'].unique().tolist()
#             for id in thread_ids:
#                 threads_platforms.append(platform)
#                 items = plat_data[plat_data['thread_id']==id]['item'].tolist()
#                 num_claims_threads.append(len(items))
#                 text = create_topic_document(items)
#                 counts,preprocessed_text = tokenize_preprocess_and_count(text,preprocess=True)
#                 documents.append(preprocessed_text)
#         num_claims.append(num_claims_threads)
#         tfidf_vectorizer = TfidfVectorizer(norm='l1')
#         X_tfidf = tfidf_vectorizer.fit_transform(documents)
#         vocaboulary = tfidf_vectorizer.get_feature_names_out()
#         matrix=X_tfidf.toarray()
        
#         threads_embeddings = get_FastText_embedding(vocaboulary,word_vectors,matrix,normalization)
#         embeddings_matrices.append(threads_embeddings)
#         platform_matrices.append(threads_platforms)
#         documents_matrices.append(documents)
#         word_importance = matrix.sum(axis=0)


#         for i in range(len(matrix)):
#             words=[]
#             values=matrix[i].tolist()
#             top_10 = heapq.nlargest(5, enumerate(values), key=lambda x: x[1])
#             for ind in top_10:
#                 words.append(vocaboulary[ind[0]])
#             threads_words_dict[str(page_id)].append(words)


#     return threads_words_dict,embeddings_matrices,platform_matrices,documents,representation_2d,num_claims


# def get_FastText_raw_embeddings(merged_data,platforms,word_vectors,normalization=True):    
#     threads_words_dict={}
#     embeddings_matrices=[]
#     debate_ids_matrices=[]
#     documents_matrices=[]
#     num_claims=[]

#     for i,page_id in enumerate(merged_data['page_id'].unique().tolist()):
#         threads_words_dict[str(page_id)] = []
#         page_data = merged_data[merged_data['page_id']==page_id]
#         print(f"NEW PAGE {i+1}. {page_data['title'].tolist()[0]}")
        
#         documents=[]
#         threads_debate_id=[]
#         num_claims_threads=[]
#         for j,platform in enumerate(platforms):
#             plat_data = page_data[page_data['platform']==platform]
#             thread_ids = plat_data['thread_id'].unique().tolist()
#             for id in thread_ids:
#                 thread_data = plat_data[plat_data['thread_id']==id]
#                 items = thread_data['item'].tolist()
#                 threads_debate_id.append(thread_data['debate_id'].tolist()[0])
#                 num_claims_threads.append(len(items))
#                 text = create_topic_document(items)
#                 counts,preprocessed_text = tokenize_preprocess_and_count(text,preprocess=True)
#                 documents.append(preprocessed_text)
#         num_claims.append(num_claims_threads)
#         tfidf_vectorizer = TfidfVectorizer(norm='l1')
#         X_tfidf = tfidf_vectorizer.fit_transform(documents)
#         vocaboulary = tfidf_vectorizer.get_feature_names_out()
#         matrix=X_tfidf.toarray()
        
#         threads_embeddings = get_FastText_embedding(vocaboulary,word_vectors,matrix,normalization)
#         embeddings_matrices.append(threads_embeddings)
#         debate_ids_matrices.append(threads_debate_id)
#         documents_matrices.append(documents)


#         for i in range(len(matrix)):
#             words=[]
#             values=matrix[i].tolist()
#             top_10 = heapq.nlargest(5, enumerate(values), key=lambda x: x[1])
#             for ind in top_10:
#                 words.append(vocaboulary[ind[0]])
#             threads_words_dict[str(page_id)].append(words)


#     return threads_words_dict,embeddings_matrices,debate_ids_matrices,documents,num_claims





def mean_semantic_similarity_sentence_transformer(embeddings, platform_labels,page_id,platform_similarities,distance_distribution,model):
    platform_labels = np.array(platform_labels) 
    unique_platforms = np.unique(platform_labels)  # Get unique platforms
    similarity_matrix = model.similarity(embeddings,embeddings)
    #similarity_matrix = np.maximum(similarity_matrix, 0.0)
    similarity_matrix = np.array(1-similarity_matrix)
    platform_similarities[page_id] = {}
    distance_distribution[page_id]= {}
    for platform in unique_platforms:
        indices = np.where(platform_labels == platform)[0] 
        if len(indices) > 1:  # Skip if there's only one embedding (no pairs)
            upper_triangle = similarity_matrix[np.ix_(indices, indices)] 
            distance_paltform_distribution=upper_triangle[np.triu_indices(len(indices), k=1)]
            mean_similarity = np.mean(distance_paltform_distribution)  # Exclude diagonal
            platform_similarities[page_id][platform] = mean_similarity
            distance_distribution[page_id][platform] = np.array(distance_paltform_distribution)
        else:
            platform_similarities[page_id][platform] = None  # No meaningful similarity for single elements

    return platform_similarities

def mean_semantic_similarity_old(embeddings, platform_labels,data,page_id,platform_similarities):
    platform_labels = np.array(platform_labels) 
    unique_platforms = np.unique(platform_labels)  # Get unique platforms
    similarity_matrix = cosine_similarity(embeddings)  # Compute cosine similarity matrix
    #platform_similarities[page_id] = {}
    similarity_matrix = np.array(1-similarity_matrix)
    for platform in unique_platforms:
        plat_data = data[(data['platform']==platform) & (data['page_id']==page_id)]
        num_debate = len(plat_data['debate_id'].unique())
        last_index=0
        for i,debate_id in enumerate(plat_data['debate_id'].unique()):
            platform_similarities[debate_id]={}
            debate_data=plat_data[plat_data['debate_id']==debate_id]
            num_threads_debate=len(debate_data['thread_id'].unique())
            indices = np.where(platform_labels == platform)[0]
            if len(indices) > 1:  # Skip if there's only one embedding (no pairs)
                indices=indices[last_index:(last_index+num_threads_debate-1)]
                upper_triangle = similarity_matrix[np.ix_(indices, indices)]  # Extract submatrix
                
                mean_similarity = np.mean(upper_triangle[np.triu_indices(len(indices), k=1)])  # Exclude diagonal
                platform_similarities[debate_id][platform] = mean_similarity
            else:
                platform_similarities[debate_id][platform] = None  # No meaningful similarity for single elements
            last_index=last_index+num_threads_debate
    return platform_similarities

def chunk_text(text, tokenizer, max_length=256, stride=128):
    """Splits text into overlapping chunks to avoid truncation loss."""
    tokens = tokenizer.tokenize(text)
    chunks = []

    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
        if i + max_length >= len(tokens):  
            break  # Stop when reaching the end

    return chunks

def get_sentence_transformers_claim_embeddings(merged_data, platforms, model, tokenizer, batch_size=32, max_length=256, stride=128):    
    embeddings_matrices = []
    debate_matrices = []
    documents_matrices = []

    for i, page_id in enumerate(merged_data['page_id'].unique().tolist()):
        page_data = merged_data[merged_data['page_id'] == page_id]
        print(f"NEW PAGE {i+1}. {page_data['title'].tolist()[0]}")
        documents = []
        claims_debate = []
        for platform in platforms:
            plat_data = page_data[page_data['platform'] == platform]
            claims_debate += plat_data['debate_id'].tolist()
            documents += plat_data['item'].tolist()

        topic_embedding = []
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing in batches"):
            batch = documents[i : i + batch_size]
            processed_batch = []
            # Handle long claims before encoding
            for claim in batch:
                if len(tokenizer.tokenize(claim)) > max_length:
                    claim_chunks = chunk_text(claim, tokenizer, max_length, stride)
                    processed_batch.append(claim_chunks)  # Store as list of chunks
                else:
                    processed_batch.append([claim])  # Store as single-element list
            # Flatten batch for encoding
            flat_batch = [chunk for sublist in processed_batch for chunk in sublist]
            batch_embeddings = model.encode(flat_batch)  

            # Aggregate embeddings (mean of chunk embeddings per original claim)
            claim_embeddings = []
            index = 0
            for sublist in processed_batch:
                num_chunks = len(sublist)
                claim_embeddings.append(np.mean(batch_embeddings[index:index + num_chunks], axis=0))
                index += num_chunks

            topic_embedding.extend(claim_embeddings)
        print(len(topic_embedding))
        np.save('data/global/st_embeddings/'+f'{page_id}_embeddings.npy',topic_embedding)
        np.save('data/global/st_debates/'+f'{page_id}_debate_ids.npy',np.array(claims_debate))
        
        embeddings_matrices.append(topic_embedding)
        debate_matrices.append(claims_debate)
        documents_matrices.append(documents)

    return embeddings_matrices, debate_matrices, documents_matrices

def convex_hull_metric(embeddings, platform_labels,page_id,dim,perplexity,convex_hull):
    platform_labels = np.array(platform_labels) 
    convex_hull[page_id] = {}
    unique_platforms = ['wiki','kialo','cmv'] #np.unique(platform_labels)
    tsnse = TSNE(n_components=dim,learning_rate='auto',init='pca',perplexity=perplexity,random_state=42, method='exact')
    points=tsnse.fit_transform(embeddings)
    for platform in unique_platforms:
        indices = np.where(platform_labels == platform)[0]
        plat_points = points[indices]
        hull = ConvexHull(plat_points)
        convex_hull[page_id][platform]=np.log(hull.volume)
    return points,convex_hull

def mean_semantic_similarity_fixed_threads(embeddings, platform_labels, page_id, platform_similarities,platform_convex_hull, num_threads):
    platform_labels = np.array(platform_labels)
    unique_platforms = ['wiki','kialo','cmv']  # Get unique platforms
    reduced_embeddings = []
    reduced_labels = []
    
    for i,platform in enumerate(unique_platforms):
        indices = np.where(platform_labels == platform)[0]  # Get indices of the platform
        print(f"{platform}: {len(indices)}")
        selected_indices = indices[:num_threads[i]]  # Keep only the first num_threads indices
        print(len(selected_indices))
        reduced_embeddings.extend(embeddings[selected_indices])
        reduced_labels.extend(platform_labels[selected_indices])
    
    reduced_embeddings = np.array(reduced_embeddings)
    reduced_labels = np.array(reduced_labels)
    
    similarity_matrix = 1-cosine_similarity(reduced_embeddings)  # Compute cosine similarity matrix
    platform_similarities[page_id] = {}
    #return reduced_embeddings, reduced_labels, similarity_matrix
    for platform in unique_platforms:
        print(platform)
        indices = np.where(reduced_labels == platform)[0]  # Get indices in reduced set
        if len(indices) > 1:  # Skip if there's only one embedding (no pairs)
            upper_triangle = similarity_matrix[np.ix_(indices, indices)]  # Extract submatrix
            
            mean_similarity = np.mean(upper_triangle[np.triu_indices(len(indices), k=1)])  # Exclude diagonal
            platform_similarities[page_id][platform] = mean_similarity
        else:
            platform_similarities[page_id][platform] = None  # No meaningful similarity for single elements
    
    # dim = 2
    # perplexity = 15
    # points,platform_convex_hull=convex_hull_metric(reduced_embeddings,reduced_labels,page_id,dim,perplexity,platform_convex_hull)

    return platform_similarities #,platform_convex_hull,points


def get_FastText_claim_embeddings(merged_data,platforms,word_vectors,pages,normalization=True,plot_keywords=True,plot_distances=False):    
    claims_words_dict={}
    embeddings_matrices=[]
    platform_matrices=[]
    documents_matrices=[]
    num_claims=[]
    for i,page_id in enumerate(merged_data['page_id'].unique().tolist()):
        #create the dictionary key
        claims_words_dict[str(page_id)] = []
        page_data = merged_data[merged_data['page_id']==page_id]
        print(f"NEW PAGE {i+1}. {page_data['title'].tolist()[0]}")
        title=page_data['title'].tolist()[0]
        page=pages[i]
        documents=[]
        num_claims_threads=[]
        claims_platform=[]
        #Discard thread made by that have less than 3 items
        #print(min_num_threads)
        for j,platform in enumerate(platforms):
            plat_data = page_data[page_data['platform']==platform]
            items = plat_data['item'].tolist()
            claims_platform+=[platform]*len(items)
            num_claims_threads.append(len(items))
                #text = create_topic_document(items)
            for item in items:
                counts,preprocessed_text = tokenize_preprocess_and_count(item,preprocess=True)
                documents.append(preprocessed_text)
        num_claims.append(num_claims_threads)
        tfidf_vectorizer = TfidfVectorizer(norm='l1')
        X_tfidf = tfidf_vectorizer.fit_transform(documents)
        vocaboulary = tfidf_vectorizer.get_feature_names_out()
        matrix=X_tfidf.toarray()
        
        claims_embeddings = get_FastText_embedding(vocaboulary,word_vectors,matrix,normalization)
        embeddings_matrices.append(claims_embeddings)
        platform_matrices.append(claims_platform)
        documents_matrices.append(documents)
        
    return embeddings_matrices,platform_matrices,documents,num_claims


def plot_matrix_distances_2D(distances,threads_platform, platforms, page,keyword=False,dir=''):
    # Step 1: Reduce dimensionality (PCA or t-SNE)
    if not keyword:
        distances=distances[1:,1:]
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    points_2d = mds.fit_transform(distances)

    

    # Plot 1: Scatterplot
    plt.figure(figsize=(8, 6))
    colors = ['orange', 'blue', 'red']
    if keyword==False:
        #num_threads = int((len(distances)) / 3)
        wiki_threads = threads_platform.count('wiki')
        kialo_threads = threads_platform.count('kialo')
        #cmv_threads = threads_platform.count('cmv')
        platform_emb = [
            points_2d[:wiki_threads],
            points_2d[wiki_threads:wiki_threads+kialo_threads],
            points_2d[wiki_threads+kialo_threads:]
        ]
    
    if keyword==True:
        num_threads = int((len(distances)-1) / 3)
        keyword_point=points_2d[0]
        points_2d=points_2d[1:]
        platform_emb = [
            points_2d[:num_threads],
            points_2d[num_threads:num_threads * 2],
            points_2d[num_threads * 2:]
        ]
    

    count = 0
    for i, platform in enumerate(platforms):
        plat_flag=False
        emb = platform_emb[i]
        x = [elem[0] for elem in emb]
        y = [elem[1] for elem in emb]
        for coord_x, coord_y in zip(x, y):
            if not plat_flag:
                plt.scatter(coord_x, coord_y, label=f'{platform}', color=colors[i])
                plat_flag=True
            else:
                plt.scatter(coord_x, coord_y, color=colors[i])
            #plt.text(coord_x + 0.005, coord_y + 0.005, f'c_{count}', fontsize=9)
            count += 1
    
    if keyword==True:
        plt.scatter(keyword_point[0],keyword_point[1], label=f'keywords', color='black',marker='*')
    
    plt.title(f"2D Scatterplot of {page} threads using cosine dist")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.savefig(f'{dir}/{page}_FastText_three_tables.png')
    plt.close()
    return points_2d



def get_distance_and_plot(k_embeddings,t_embeddings,threads_platform,page,platforms,keyword=True,dir=''):
    stack_emb=np.vstack((k_embeddings,t_embeddings))
    similarities_ft_idf = cosine_similarity(stack_emb)

    # Convert similarity to distance
    distance_matrix = 1 - similarities_ft_idf

    # Clamp any negative distances to 0 (unlikely if similarity is in [0, 1])
    distance_matrix = np.maximum(distance_matrix, 0)

    points_2d=plot_matrix_distances_2D(distance_matrix,threads_platform,platforms,page=page,keyword=keyword,dir=dir)
    plot_heatmap_cosine_similarity(similarities_ft_idf,page,dir=dir)
    return points_2d


def get_FastText_raw_embeddings(merged_data,platforms,word_vectors,pages,normalization=True,plot_keywords=True,plot_distances=False):    
    threads_words_dict={}
    embeddings_matrices=[]
    platform_matrices=[]
    documents_matrices=[]
    representation_2d=[]
    clusters=[]
    num_claims=[]
    for i,page_id in enumerate(merged_data['page_id'].unique().tolist()):
        #create the dictionary key
        threads_words_dict[str(page_id)] = []
        page_data = merged_data[merged_data['page_id']==page_id]
        print(f"NEW PAGE {i+1}. {page_data['title'].tolist()[0]}")
        title=page_data['title'].tolist()[0]
        page=pages[i]
        documents=[]

        #Discard thread made by that have less than 3 items
        level_one = page_data[page_data['level']==1]
        #num_threads_pl= [len(level_one[level_one['platform']==pl]['thread_id'].unique()) for pl in platforms]
        #num_threads_pl= [3,3,3]
        #min_num_threads=min(num_threads_pl)
        #print(min_num_threads)
        threads_platforms=[]
        num_claims_threads=[]
        for j,platform in enumerate(platforms):
            plat_data = page_data[page_data['platform']==platform]
            #level_1 = plat_data[plat_data['level']==1].sort_values(by='thread_items',ascending=False)
            #thread_ids = level_1['thread_id'].unique().tolist()[:num_threads_pl[j]]
            thread_ids = plat_data['thread_id'].unique().tolist()
            for id in thread_ids:
                threads_platforms.append(platform)
                items = plat_data[plat_data['thread_id']==id]['item'].tolist()
                num_claims_threads.append(len(items))
                text = create_topic_document(items)
                counts,preprocessed_text = tokenize_preprocess_and_count(text,preprocess=True)
                documents.append(preprocessed_text)
        num_claims.append(num_claims_threads)
        tfidf_vectorizer = TfidfVectorizer(norm='l1')
        X_tfidf = tfidf_vectorizer.fit_transform(documents)
        vocaboulary = tfidf_vectorizer.get_feature_names_out()
        matrix=X_tfidf.toarray()
        
        threads_embeddings = get_FastText_embedding(vocaboulary,word_vectors,matrix,normalization)
        embeddings_matrices.append(threads_embeddings)
        platform_matrices.append(threads_platforms)
        documents_matrices.append(documents)
        #keyword_embedding = get_keywords_embedding(word_tokenize(keywords[i]),word_vectors)
        word_importance = matrix.sum(axis=0)
        #embeddings = keyword_embedding+threads_embeddings

        for i in range(len(matrix)):
            words=[]
            values=matrix[i].tolist()
            #print(max(values))
            top_10 = heapq.nlargest(5, enumerate(values), key=lambda x: x[1])
            for ind in top_10:
                words.append(vocaboulary[ind[0]])
            threads_words_dict[str(page_id)].append(words)

        #Find global keywords
        top_keywords = heapq.nlargest(10, enumerate(word_importance), key=lambda x: x[1])
        keywords=vocaboulary[[elem[0] for elem in top_keywords]]
        keywords_embedding = get_keywords_embedding(keywords,word_vectors,normalization)

        if plot_distances:
            points_2d=get_distance_and_plot(keywords_embedding,threads_embeddings,threads_platforms,page,platforms,
                                  keyword=plot_keywords,dir='plots/topic_distance/new_clustering_method')
            #plot_tables(threads_words_dict[str(page_id)],len(threads_embeddings),title
                        #,dir='plots/topic_distance/new_clustering_method')
        representation_2d.append(points_2d)
    return threads_words_dict,embeddings_matrices,platform_matrices,documents,representation_2d,num_claims




def get_sentence_transformers_claim_embeddings(merged_data,platforms,model):    
    embeddings_matrices=[]
    debate_matrices=[]
    documents_matrices=[]
    for i,page_id in enumerate(merged_data['page_id'].unique().tolist()[:1]):
        page_data = merged_data[merged_data['page_id']==page_id]
        print(f"NEW PAGE {i+1}. {page_data['title'].tolist()[0]}")
        documents=[]
        claims_debate=[]
        for j,platform in enumerate(platforms):
            plat_data = page_data[page_data['platform']==platform]
            items = plat_data['item'].tolist()
            claims_debate+=plat_data['debate_id'].tolist()
            documents+=items

        batch_size = 32  # Adjust based on available memory
        topic_embedding = []

        for i in tqdm(range(0, len(documents), batch_size), desc="Processing in batches"):
            batch = documents[i : i + batch_size]
            batch_embeddings = model.encode(batch)  # Encode entire batch
            topic_embedding.extend(batch_embeddings) 

        # topic_embedding=[]
        # for claim in tqdm(documents, desc="Processing documents"):
        #     #sentences = tokenize.sent_tokenize(claim)
        #     embeddings = model.encode(tokenize.sent_tokenize(claim)) #,normalize_embeddings=True
        #     #cumulative_embedding = 
        #     topic_embedding.append(np.mean(embeddings, axis=0))

        embeddings_matrices.append(topic_embedding)
        debate_matrices.append(claims_debate)
        documents_matrices.append(documents)
        
    return embeddings_matrices,debate_matrices,documents 