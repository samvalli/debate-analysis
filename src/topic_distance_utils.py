from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


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

def chunk_text(text, tokenizer, max_length=256, stride=128):

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


