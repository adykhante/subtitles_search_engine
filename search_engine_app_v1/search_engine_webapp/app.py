from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load the MiniLM-L6-v2 model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define input folder path
input_folder_path = r"C:\Users\adykh\Desktop\subs_db\subtitles\subtitle_demo\overlapping_chunks"

def encode_chunks(chunks):
    # Encode each chunk using the Sentence Transformer model
    chunk_embeddings = model.encode(chunks)
    return chunk_embeddings

def cosine_similarity_search(query_embedding, chunk_embeddings, chunk_texts):
    # Calculate cosine similarity between query embedding and chunk embeddings
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    # Sort the similarities in descending order
    sorted_indices = np.argsort(similarities)[::-1]
    # Return sorted chunk texts, similarities, and corresponding file names
    sorted_chunk_texts = [chunk_texts[i] for i in sorted_indices]
    sorted_similarities = [similarities[i] for i in sorted_indices]
    return sorted_chunk_texts, sorted_similarities, sorted_indices

def search_engine(query, num_chunks_per_file, top_k=5):
    # Initialize a list to store the similarity scores for each file
    file_similarities = []

    # Process each file in the input folder
    for filename in os.listdir(input_folder_path):
        input_file_path = os.path.join(input_folder_path, filename)
        if os.path.isfile(input_file_path):
            # Read file content
            with open(input_file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Find the starting and ending index of the chunks
            start_index = content.find("[")
            end_index = content.rfind("]")

            # Extract the chunk content
            chunk_content = content[start_index + 1:end_index]

            # Split the chunk content into individual chunks
            chunks = [chunk.strip()[1:-1] for chunk in chunk_content.split(",")]

            # Encode the query
            query_embedding = model.encode([query])

            # Calculate cosine similarity for each chunk
            chunk_similarities = []
            for chunk_text in chunks[:num_chunks_per_file]:
                # Encode the chunk
                chunk_embedding = model.encode([chunk_text])
                # Calculate cosine similarity with the query embedding
                similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]
                chunk_similarities.append(similarity)

            # Average similarity for the specified number of chunks per file
            average_similarity = np.mean(chunk_similarities)

            # Append file name and average similarity to the list
            file_similarities.append((filename, average_similarity))

    # Sort the file similarities based on average similarity in descending order
    sorted_file_similarities = sorted(file_similarities, key=lambda x: x[1], reverse=True)

    # Return top-k results
    return sorted_file_similarities[:top_k]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get query from form submission
        query = request.form.get('query')
        
        if request.form.get('clear') == 'true':
            # If clear button is clicked, return to the index page without any results
            return render_template('index.html')
        
        # Check if query is provided
        if not query:
            return render_template('index.html', error='Query is required.')

        # Perform search
        results = search_engine(query, num_chunks_per_file=5, top_k=10)

        # Format results
        formatted_results = [{'file': result[0], 'average_similarity': result[1]} for result in results]

        return render_template('index.html', query=query, results=formatted_results)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
