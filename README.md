# Information Retrieval Engine 
## A search engine built on Wikipedia 
The search engine searches relevant wikipedia pages, for a given query.

## Key Component of the Engine
The project contains several logical units

### Search Front-End
* search - the main search method of the engine, combine results from number of sub-searches.
* search_body - search function over the body index.
* search_title - search function over the title index.
* search_anchor - search function over the anchor index.
* get_pagerank - return the pagerank of a given wiki-id page.
* get_pageviews - return the page views count of a given wiki article id


### Search Engine
* set_indices - Initialize the Indices.
* set_dicts - Initialize the Dictionaries.
* page_rank - Given a list of wiki_ids, returns the page rank of each id.
* page_views -  Given a list of wiki_ids, returns the page views of each id.

And a variety of other search functions, like search_body and search_body_tfidf which search using different indices
and different techniques to retrieve relevant documents.

### Tokenizer 
* _get_stopwords - returns the Corpus stopwords.
* tokenize - returns a list of tokenized words.


### Stemmer
A stemmer using Snowball stemmer, the stemmer is based on the Porter
stemmer algorithm but also includes additional features such as support for multiple languages and the ability to perform stemming using various algorithms.


### Ranker
The Ranker class is a collection of functions for ranking documents based on different similarity measures.

* cosine_similarity - calculates the cosine similarity between a given query and a set of documents.
* top_N_documents - sorts and filters the top N documents (by score) for each query.
* binary_ranking - returns a list of documents that contain at least one term from the query.
* page_rank - returns a list of page ranks for given a list of wiki_ids.
* page_views - returns a list of views for given a list of wiki_ids.
* get_candidate_documents_and_scores - generate a dictionary representing a pool of candidate documents for a given query.


### Evaluator
The Evaluator class evaluates the performance of the engine.
* evaluate - returns a dictionary of evaluation metrics for the engine, using true_rank as the true relevance scores, 
  predicted_rank as the predicted relevance scores, and k as the number of top-k documents to consider.
  the metrices calculated are : intersect, recall_at_k, precision_at_k, r_precision, reciprocal_rank_at_k, fallout_rate and f_score.


### Pickle Handler
allows to interact with pickle files, as well as Google Cloud Storage (GCS) bucket.
* download_from_gcp - Downloads a file from the GCS bucket and saves it to the specified destination path.
* get_index - Downloads the index file from the GCS bucket and returns it as an InvertedIndex object.
* write_pickle_file - Writes a file to the specified path in pickle format.
* read_pickle_file - Reads a file from the specified path in pickle format.
* read_pickle_from_gcp - Downloads a file from the GCS bucket and reads it in pickle format.
* exists - Checks if a file exists at the specified path.
* get_page_rank - Downloads a csv.gz file containing page rank values, reads it, and returns it as a pandas dataframe.

In order to query our search engine, please use this path:
`http://34.28.7.246:8080/`

For example:
import requests

requests.get('http://34.28.7.246:8080' + '/search', {'query': q}, timeout=35)

In order to query our ultimate search - use 'search' suffix.
In order to query our body search - use 'search_body' suffix.
In order to query our title search - use 'search_title' suffix.
In order to query our anchor search - use 'search_anchor' suffix.
In order to query our page rank search - use 'get_pagerank' suffix.
In order to query our page rank search - use 'get_pageview' suffix.
