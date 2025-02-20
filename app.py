import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import nltk
import os
import io
import pickle
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from transformers import GPT2TokenizerFast

# Page config
st.set_page_config(page_title="IFRM Search & Chat", layout="wide")

# Initialize OpenAI client and tokenizer
client = OpenAI()
tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/gpt-4o")
MAX_CONTEXT_WINDOW = 128000  # GPT-4o context window size

# Determine device - will use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize session state for chat history if not exists
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'search_results' not in st.session_state:
    st.session_state.search_results = None

def init_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

init_nltk_resources()

# Cache the embedding model
@st.cache_resource
def get_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    return model

@st.cache_data
def load_default_dataset(default_dataset_path):
    # Get absolute path
    try:
        abs_path = os.path.abspath(default_dataset_path)
        st.sidebar.info(f"Looking for dataset at: {abs_path}")
        
        if not os.path.exists(abs_path):
            st.error(f"Dataset not found at: {abs_path}")
            return None
            
        # Try to load the dataset
        try:
            df = pd.read_excel(abs_path)
            st.sidebar.success(f"Successfully loaded dataset with {len(df)} rows")
            return df
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error resolving file path: {str(e)}")
        return None

@st.cache_data
def load_uploaded_dataset(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df

def generate_embeddings(texts, model):
    with st.spinner('Calculating embeddings...'):
        embeddings = model.encode(texts, show_progress_bar=True, device=device)
    return embeddings

def load_or_compute_embeddings(df, using_default_dataset, uploaded_file_name=None, text_columns=None):
    if text_columns is None or len(text_columns) == 0:
        return None, None

    # Get the absolute path of the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create 'embeddings' subdirectory if it doesn't exist
    embeddings_dir = os.path.join(current_dir, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Create a consistent filename
    cols_key = "_".join(sorted(text_columns))
    
    if using_default_dataset:
        embeddings_file = os.path.join(embeddings_dir, f'PRMS_2022_2023_QAed_{cols_key}.pkl')
        st.sidebar.info(f"Embeddings file path: {embeddings_file}")  # Debug info
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(uploaded_file_name)[0] if uploaded_file_name else "custom_dataset"
        embeddings_file = os.path.join(embeddings_dir, f"{base_name}_{cols_key}_{timestamp_str}.pkl")

    # First check if we have valid embeddings in session state
    if ('embeddings' in st.session_state and 
        'last_text_columns' in st.session_state and 
        st.session_state['last_text_columns'] == text_columns and 
        len(st.session_state['embeddings']) == len(df)):
        return st.session_state['embeddings'], embeddings_file

    # If not in session state, try to load from file
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
                if len(embeddings) == len(df):  # Verify the embeddings match the dataset size
                    st.write(f"Loading pre-calculated embeddings from: {embeddings_file}")
                    # Store in session state for future use
                    st.session_state['embeddings'] = embeddings
                    st.session_state['last_text_columns'] = text_columns
                    return embeddings, embeddings_file
        except Exception as e:
            st.error(f"Error loading embeddings file: {str(e)}")

    # If we reach here, we need to compute new embeddings
    st.write(f"Generating new embeddings, will save to: {embeddings_file}")
    df_fill = df.fillna("")
    texts = df_fill[text_columns].astype(str).agg(' '.join, axis=1).tolist()
    model = get_embedding_model()
    embeddings = generate_embeddings(texts, model)
    
    # Save to file and session state
    try:
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        st.success(f"Saved embeddings to: {embeddings_file}")
    except Exception as e:
        st.error(f"Error saving embeddings file: {str(e)}")
    
    st.session_state['embeddings'] = embeddings
    st.session_state['last_text_columns'] = text_columns
    
    return embeddings, embeddings_file

def reset_filters():
    st.session_state['selected_regions'] = []
    st.session_state['selected_countries'] = []
    st.session_state['selected_centers'] = []
    st.session_state['selected_impact_area'] = []
    st.session_state['selected_sdg_targets'] = []
    st.session_state['additional_filters'] = {}
    st.session_state['selected_additional_filters'] = {}

def get_chat_response(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error querying OpenAI: {e}")
        return None

# Main app layout
st.title("IFRM Search & Chat")

# Create two main columns for the layout
col1, col2 = st.columns([2, 1])

with col1:
    # Dataset selection in sidebar
    st.sidebar.title("Data Selection")
    dataset_option = st.sidebar.selectbox('Select Dataset', ('PRMS 2022+2023 QAed', 'Upload my dataset'))

    if dataset_option == 'PRMS 2022+2023 QAed':
        # Use a simpler path resolution
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_dataset_path = os.path.join(current_dir, 'input', 'export_data_table_results_20240312_160222CET.xlsx')
        df = load_default_dataset(default_dataset_path)
        if df is not None:
            st.session_state['df'] = df.copy()
            st.session_state['using_default_dataset'] = True
    else:
        uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])
        if uploaded_file is not None:
            df = load_uploaded_dataset(uploaded_file)
            if df is not None:
                st.session_state['df'] = df.copy()
                st.session_state['using_default_dataset'] = False
                st.session_state['uploaded_file_name'] = uploaded_file.name

    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Text columns selection
        st.write("**Select Text Columns for Search**")
        text_columns_selected = st.multiselect(
            "Text Columns:",
            df.columns.tolist(),
            default=['Title', 'Description'] if 'Title' in df.columns and 'Description' in df.columns else []
        )
        st.session_state['text_columns'] = text_columns_selected

        # Dynamic Filtering Interface
        st.write("**Filter Options**")
        # Exclude text columns and similarity score from filter options
        filter_candidates = [col for col in df.columns if col not in text_columns_selected and col != 'similarity_score']
        selected_filter_cols = st.multiselect(
            "Select columns to filter by:",
            filter_candidates,
            default=st.session_state.get('filter_columns_selected', [])
        )
        st.session_state['filter_columns_selected'] = selected_filter_cols

        # Store filter selections
        current_filters = {}
        
        # Create filter interface for each selected column
        for col_name in selected_filter_cols:
            if f'filter_{col_name}' not in st.session_state:
                st.session_state[f'filter_{col_name}'] = []
            
            unique_vals = df[col_name].dropna().unique().tolist()
            unique_vals = sorted(unique_vals)
            
            selected_vals = st.multiselect(
                f"Filter by {col_name}:",
                options=unique_vals,
                default=st.session_state[f'filter_{col_name}']
            )
            st.session_state[f'filter_{col_name}'] = selected_vals
            current_filters[col_name] = selected_vals

        # Semantic Search Interface
        st.header("Semantic Search")
        
        query = st.text_input("Enter your search query:")
        include_keywords = st.text_input("Filter by keywords (comma-separated):", help="Enter keywords to filter results. Only shows results containing ALL keywords (case-insensitive).")
        
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.35)

        # Function to apply filters
        def apply_filters(df_to_filter):
            filtered_df = df_to_filter.copy()
            for col, values in current_filters.items():
                if values:  # Only apply filter if values are selected
                    filtered_df = filtered_df[filtered_df[col].isin(values)]
            return filtered_df

        # Add search button
        if st.button("Apply Filters and Search"):
            # Apply filters first
            filtered_results = apply_filters(df)

            # Then apply semantic search if query exists
            if query.strip() and text_columns_selected:
                with st.spinner("Performing Semantic Search..."):
                    # Get embeddings for the full dataset
                    embeddings, _ = load_or_compute_embeddings(
                        df,  # Use full dataset instead of filtered_results
                        st.session_state.get('using_default_dataset', False),
                        st.session_state.get('uploaded_file_name', None),
                        text_columns_selected
                    )

                    if embeddings is not None:
                        model = get_embedding_model()
                        df_fill = df.fillna("")  # Use full dataset
                        search_texts = df_fill[text_columns_selected].agg(' '.join, axis=1).tolist()
                        query_embedding = model.encode([query], device=device)
                        similarities = cosine_similarity(query_embedding, embeddings)
                        
                        # Get indices that meet similarity threshold
                        above_threshold_indices = np.where(similarities[0] > similarity_threshold)[0]

                        if len(above_threshold_indices) == 0:
                            st.warning("No results found above the similarity threshold.")
                            st.session_state.search_results = None
                        else:
                            # Apply filters to the original dataframe indices first
                            filtered_indices = filtered_results.index
                            
                            # Find intersection of filtered indices and similarity threshold indices
                            valid_indices = np.intersect1d(filtered_indices, above_threshold_indices)
                            
                            if len(valid_indices) == 0:
                                st.warning("No results found after applying filters.")
                                st.session_state.search_results = None
                            else:
                                # Get the results and their similarity scores
                                results = df.iloc[valid_indices].copy()
                                results['similarity_score'] = similarities[0][valid_indices]
                                results = results.sort_values(by='similarity_score', ascending=False)

                                # Apply include keywords filter
                                if include_keywords.strip():
                                    inc_words = [w.strip().lower() for w in include_keywords.split(',') if w.strip()]
                                    if inc_words:
                                        # Only search in selected text columns
                                        text_content = results[text_columns_selected].astype(str).agg(' '.join, axis=1).str.lower()
                                        results = results[text_content.apply(lambda x: all(w in x for w in inc_words))]

                                if results.empty:
                                    st.warning("No results found after applying filters.")
                                    st.session_state.search_results = None
                                else:
                                    st.session_state.search_results = results
            else:
                # If no search query, just use filtered results and apply keyword filtering
                filtered_results = apply_filters(df)
                
                # Apply include keywords filter even without semantic search
                if include_keywords.strip():
                    inc_words = [w.strip().lower() for w in include_keywords.split(',') if w.strip()]
                    if inc_words:
                        # Only search in selected text columns
                        text_content = filtered_results[text_columns_selected].astype(str).agg(' '.join, axis=1).str.lower()
                        filtered_results = filtered_results[text_content.apply(lambda x: all(w in x for w in inc_words))]
                
                if filtered_results.empty:
                    st.warning("No results found after applying filters.")
                    st.session_state.search_results = None
                else:
                    st.session_state.search_results = filtered_results

        # Always display results if they exist (outside the search button condition)
        if st.session_state.search_results is not None:
            st.write(f"Found {len(st.session_state.search_results)} results:")
            display_df = st.session_state.search_results.reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True)

with col2:
    st.header("Chat with Results")
    
    if st.session_state.search_results is not None:
        # Add clear message about result limitation
        if len(st.session_state.search_results) > 210:
            st.info("ℹ️ For optimal performance, the chat will only analyze the first 210 results.")
            
        # Initialize system message
        system_msg = {
            "role": "system",
            "content": "You are a specialized assistant analyzing search results from a research database. "
                      "Your role is to:\n"
                      "1. Provide clear, concise answers based on the search results provided\n"
                      "2. Highlight relevant information from specific results when answering\n"
                      "3. When referencing specific results, ALWAYS use their Result code (e.g., 'Result Code 12345')\n"
                      "4. Clearly state if information is not available in the results\n"
                      "5. Maintain a professional and analytical tone\n\n"
                      "The search results are provided in a structured format where:\n"
                      "- Each result is separated by a blank line\n"
                      "- Each result starts with its Result code\n"
                      "- Within each result, information is organized as 'Column name: Value' pairs\n"
                      "- All available fields are included for comprehensive analysis\n\n"
                      "Example result format:\n"
                      "12:\n"
                      "Result code: 12\n"
                      "Year: 2022\n"
                      "PDF link: https://reporting.cgiar.org/reports/result-details/12?phase=1\n"
                      "Title: Duroc breed of pig: Pig breed factsheet for Uganda\n"
                      "... (additional fields)\n\n"
                      "Note: Always refer to results by their Result code, NOT by any other numbering system."
        }

        # Calculate and display token usage
        display_results = st.session_state.search_results.head(210) if len(st.session_state.search_results) > 210 else st.session_state.search_results
        results_text = "Search Results:\n"
        for _, row in display_results.iterrows():
            results_text += f"\n{row['Result code']}:\n"
            # Include all columns except similarity_score
            for col in row.index:
                if col != 'similarity_score' and not pd.isna(row[col]) and str(row[col]).strip():  # Skip empty/null values
                    results_text += f"{col}: {row[col]}\n"
        
        if len(st.session_state.search_results) > 210:
            results_text += f"\nNote: Showing first 210 results out of {len(st.session_state.search_results)} total results."
        
        # Count tokens in system message and results
        system_tokens = len(tokenizer(system_msg["content"])["input_ids"])
        results_tokens = len(tokenizer(results_text)["input_ids"])
        total_tokens = system_tokens + results_tokens
        context_usage_percent = (total_tokens / MAX_CONTEXT_WINDOW) * 100
        
        # Display token usage information
        st.subheader("Context Window Usage")
        st.write(f"System Message: {system_tokens:,} tokens")
        st.write(f"Search Results: {results_tokens:,} tokens")
        st.write(f"Total: {total_tokens:,} tokens ({context_usage_percent:.1f}% of available context)")
        if context_usage_percent > 90:
            st.warning("⚠️ High context usage! Consider reducing the number of results or filtering further.")
        elif context_usage_percent > 75:
            st.info("ℹ️ Moderate context usage. Still room for your question, but consider reducing results if asking a long question.")

        # Add download button for chat context
        chat_context = f"""System Message:
{system_msg['content']}

{results_text}"""
        st.download_button(
            label="📥 Download Chat Context",
            data=chat_context,
            file_name="chat_context.txt",
            mime="text/plain",
            help="Download the exact context that the chatbot receives, including system message and all results"
        )

        # Display chat interface
        col_chat1, col_chat2 = st.columns([3, 1])
        with col_chat1:
            user_input = st.text_area("Ask a question about the search results:", key="chat_input")
        with col_chat2:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        if st.button("Send", key="send_button"):
            if user_input:
                with st.spinner("Processing results and getting response..."):
                    # Format results only when sending a question
                    results_text = "Search Results:\n"
                    
                    # Get the first 210 results if there are more (to avoid context window issues)
                    display_results = st.session_state.search_results.head(210) if len(st.session_state.search_results) > 210 else st.session_state.search_results
                    
                    for _, row in display_results.iterrows():
                        results_text += f"\n{row['Result code']}:\n"
                        # Include all columns except similarity_score
                        for col in row.index:
                            if col != 'similarity_score' and not pd.isna(row[col]) and str(row[col]).strip():  # Skip empty/null values
                                results_text += f"{col}: {row[col]}\n"
                    
                    # Add note if results were truncated
                    if len(st.session_state.search_results) > 210:
                        results_text += f"\nNote: Showing first 210 results out of {len(st.session_state.search_results)} total results."
                    
                    # Add user's question to chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    
                    # Prepare messages for API call
                    messages = [system_msg]
                    messages.append({"role": "user", "content": f"Here are the search results to reference:\n\n{results_text}\n\nUser question: {user_input}"})
                    
                    # Get response from OpenAI
                    response = get_chat_response(messages)
                    
                    if response:
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display chat history
        st.subheader("Chat History")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write("You:", message["content"])
            else:
                st.write("Assistant:", message["content"])
    else:
        st.info("Please perform a search first to enable chat functionality.")

# Log device being used
if device == 'cuda':
    st.sidebar.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.info("Using CPU")
