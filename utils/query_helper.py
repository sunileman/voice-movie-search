import json


from utils.openai_embedder import get_embedding

from utils.openai_helper import get_openai_guidance_no_context, get_openai_large_guidance, get_chat_guidance, \
    get_chat_guidance_rag
from variables import vector_embedding_field, model, elser_embedding_field, elser_model, byom_index_name
import streamlit as st

def build_bm25_query(user_query):
    # Constructing the match query for 'organic' search using 'body_content'
    bm25_query = {
        "bool": {
            "should": [
                {
                    "query_string": {
                        "default_field": "body_content",
                        "query": user_query
                    }
                }
            ]
        }
    }

    full_query = {
        "size": 5,  # Specify the number of results to return
        "query": bm25_query
    }

    # Debug: Dump the assembled query for inspection
    print(json.dumps(full_query, indent=4))

    return full_query


def build_openai_hybrid_query(embeddings, user_query, BM25_Boost, KNN_Boost):
    """
    Builds a hybrid Elasticsearch query based on the provided parameters.

    Returns:
    - A dictionary representing the Elasticsearch query.
    """

    knn_query = {
        "field": vector_embedding_field,  # Field containing the OpenAI embeddings
        "k": 10,
        "num_candidates": 100,
        "query_vector": embeddings,
        "boost": KNN_Boost
    }

    main_filters = []

    # Copy the main_filters for the knn part
    knn_filters = list(main_filters)

    query = {
        "query": {
            "bool": {
                "must": {
                    "match": {
                        "combined_relevancy": {
                            "query": user_query,
                            "boost": BM25_Boost
                        }
                    }
                },
                "filter": main_filters
            }
        },
        "knn": knn_query
    }

    if knn_filters:
        query["knn"]["filter"] = knn_filters

    print(json.dumps(query, indent=4))

    return query


def build_hybrid_query(user_query, BM25_Boost, KNN_Boost):
    """
    Builds a hybrid Elasticsearch query based on the provided parameters.

    Returns:
    - A dictionary representing the Elasticsearch query.
    """

    # Base structure for knn
    knn_structure = {
        "field": vector_embedding_field,
        "k": 10,
        "num_candidates": 100,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": model,
                "model_text": user_query
            }
        },
        "boost": KNN_Boost
    }

    main_filters = []

    # Copy the main_filters for the knn part
    knn_filters = list(main_filters)

    query = {
        "query": {
            "bool": {
                "must": {
                    "match": {
                        "text": user_query
                    }
                },
                "filter": main_filters
            }
        },
        "knn": knn_structure
    }

    if knn_filters:
        query["knn"]["filter"] = knn_filters

    print(json.dumps(query, indent=4))

    return query


def build_vector(es, text):
    docs = [{"text_field": text}]
    response = es.ml.infer_trained_model(model_id=model, docs=docs)

    predicted_value = response.get('inference_results', [{}])[0].get('predicted_value', [])

    print(predicted_value)
    return predicted_value


def build_elser_hybrid_query(user_query, selected_authors, selected_states, selected_companies, bm25_boost,
                             elser_boost):
    """
    Builds an Elasticsearch hybrid query combining BM25 and Elser.

    Returns:
    - A dictionary representing the Elasticsearch query.
    """

    # Base structure for text_expansion
    text_expansion_structure = {
        "text_expansion": {
            elser_embedding_field: {
                "model_text": user_query,
                "model_id": elser_model
            }
        }
    }

    text_expansion_structure["text_expansion"][elser_embedding_field]["boost"] = elser_boost

    # Base structure for query_string
    query_string_structure = {
        "query_string": {
            "default_field": "text",  # Assuming text is the main field you're querying
            "query": user_query
        }
    }

    query_string_structure["query_string"]["boost"] = bm25_boost

    query = {
        "query": {
            "bool": {
                "should": [
                    text_expansion_structure,
                    query_string_structure
                ],
                "filter": []
            }
        },
        "aggs": {
            "author_facet": {
                "terms": {
                    "field": "metadata.author.keyword",
                    "min_doc_count": 1
                }
            },
            "state_facet": {
                "terms": {
                    "field": "metadata.state.keyword",
                    "min_doc_count": 1
                }
            },
            "company_facet": {
                "terms": {
                    "field": "metadata.company.keyword",
                    "min_doc_count": 1
                }
            }
        }
    }

    filters = []
    if selected_authors:
        filters.append({
            "terms": {
                "metadata.author.keyword": selected_authors
            }
        })

    # If companies are selected, we'll add a filter
    if selected_companies:
        filters.append({
            "terms": {
                "metadata.company.keyword": selected_companies
            }
        })

    if selected_states:
        filters.append({
            "terms": {
                "metadata.state.keyword": selected_states
            }
        })

    if filters:
        query["query"]["bool"]["filter"] = filters

    return query


def build_knn_query(user_query, query_vector):
    """
    Builds an updated Elasticsearch KNN query for nested structures with query vectors.

    Parameters:
    - user_query: The query text input by the user. (Not used in this specific function but kept for compatibility)
    - query_vector: The precomputed vector for the KNN query.

    Returns:
    - A dictionary representing the Elasticsearch KNN nested query.
    """

    # Nested KNN query structure
    nested_knn_query = {
        "query": {
            "nested": {
                "path": "passages",
                "query": {
                    "knn": {
                        "query_vector": query_vector,
                        "field": "passages.vector.predicted_value",
                        "num_candidates": 2
                    }
                },
                "inner_hits": {
                    "_source": [
                        "passages.text"
                    ]
                }
            }
        }
    }

    # Debug: Dump the assembled query for inspection
    print(json.dumps(nested_knn_query, indent=4))

    return nested_knn_query


def build_rrf_query(embeddings, user_query, rrf_rank_constant, rrf_window_size):
    """
    Builds a complex query with sub_searches including match, knn, and text_expansion queries,
    and aggregates results using RRF.

    Parameters:
    - embeddings: The precomputed vector for the KNN query.
    - user_query: The query text input by the user.
    - rrf_rank_constant: The rank constant used in RRF ranking.
    - rrf_window_size: The window size used in RRF ranking.

    Returns:
    - A dictionary representing the complex query.
    """

    # Define the base structure of the query
    query = {
        "sub_searches": [
            {
                "query": {
                    "match": {
                        "body_content": user_query
                    }
                }
            },
            {
                "query": {
                    "nested": {
                        "path": "passages",
                        "query": {
                            "knn": {
                                "query_vector": embeddings,
                                "field": "passages.vector.predicted_value",
                                "num_candidates": 50
                            }
                        }
                    }
                }
            },
            {
                "query": {
                    "nested": {
                        "path": "passages",
                        "query": {
                            "bool": {
                                "should": [
                                    {
                                        "text_expansion": {
                                            "passages.content_embedding.predicted_value": {
                                                "model_id": ".elser_model_2_linux-x86_64",
                                                "model_text": user_query
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        ],
        "rank": {
            "rrf": {
                "window_size": rrf_window_size,
                "rank_constant": rrf_rank_constant
            }
        }
    }
    # Debug: Print the assembled query for inspection
    print(json.dumps(query, indent=4))

    return query


def build_elser_query(user_query):
    # Nested query with text_expansion
    nested_query = {
        "nested": {
            "path": "passages",
            "query": {
                "bool": {
                    "should": [
                        {
                            "text_expansion": {
                                "passages.content_embedding.predicted_value": {
                                    "model_id": ".elser_model_2_linux-x86_64",
                                    "model_text": user_query
                                }
                            }
                        }
                    ]
                }
            }
        }
    }

    query = {
        "size": 5,
        "query": nested_query
    }

    # Debug: Print the assembled query for inspection
    print(json.dumps(query, indent=4))

    return query


def build_openai_query(embeddings, color=None, os=None):
    """
    Builds an Elasticsearch KNN query using OpenAI embeddings and includes aggregations.

    Parameters:
    - embeddings: The vector embeddings.
    - color (optional): Color value for filtering.
    - os (optional): OS value for filtering.
    - selected_states (optional): List of selected states.
    - selected_companies (optional): List of selected companies.
    - optional_state (optional): State value for the should clause.

    Returns:
    - A dictionary representing the Elasticsearch KNN query.
    """

    # Base KNN query structure
    knn_query = {
        "field": vector_embedding_field,  # Field containing the OpenAI embeddings
        "k": 10,
        "num_candidates": 100,
        "query_vector": embeddings
    }

    filters = []
    bool_query = {}

    # Handle color filter
    if color:
        filters.append({
            "term": {
                "color": color
            }
        })

    # Handle OS filter
    if os:
        filters.append({
            "term": {
                "OS": os
            }
        })

    # If there are any filters, add them to the bool query
    if filters:
        bool_query["filter"] = filters

    # If the bool_query is not empty, add it to the knn_query
    if bool_query:
        knn_query["filter"] = {
            "bool": bool_query
        }

    # Final query structure including aggregations
    full_query = {
        "knn": knn_query
    }

    # Dump the assembled query for debugging
    print(json.dumps(full_query, indent=4))

    return full_query


def get_loc_entity(result):
    for doc in result.get('inference_results', []):
        entities = doc.get('entities', [])
        for entity in entities:
            if entity['class_name'] == "LOC":
                return entity['entity']
    return None  # Return None if no LOC entity is found



def find_color_in_text(text):
    # List of common colors
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "black", "white", "pink", "gray", "violet",
              "cyan", "magenta", "gold", "silver", "bronze", "grey"]

    # Split the text into words and check each word against the list of colors
    for word in text.split():
        if word.lower() in colors:
            print("Color found in text: " + word.lower())
            return word.lower()

    return None


def find_os_in_text(text):
    """
    Finds the OS mentioned in the provided text.

    Parameters:
    - text: The input text.

    Returns:
    - A string representing the OS if found, otherwise None.
    """

    os_list = ["chrome", "chromebook", "windows"]

    # Split the text into words and check each word against the list of OS
    for word in text.split():
        if word.lower() in os_list:
            if word.lower() in ["chrome", "chromebook"]:
                print("OS found in text: Chrome OS")
                return "chrome"
            elif word.lower() == "windows":
                print("OS found in text: Windows")
                return "windows"

    return None


def search_products_for_chatbot(es, user_query, searchtype, rrf_rank_constant, rrf_window_size,
                                azureclient, conversation_history):
    # Select the appropriate query building function based on searchtype
    if searchtype == "Vector":
        query = build_knn_query(user_query, build_vector(es, user_query))
    elif searchtype == "BM25":
        query = build_bm25_query(user_query)
    elif searchtype == "Reciprocal Rank Fusion":
        query = build_rrf_query(build_vector(es, user_query), user_query, rrf_rank_constant, rrf_window_size)
    elif searchtype == "Elser":
        query = build_elser_query(user_query)
    elif searchtype == "Vector OpenAI":
        # query = build_openai_query(get_embedding(user_query), run_ner_inference(es, user_query))
        query = build_openai_query(get_embedding(user_query), find_color_in_text(user_query),
                                   find_os_in_text(user_query))

    elif searchtype == "GenAI":
        print("GenAI Search Only")
    else:
        raise ValueError(f"Invalid searchtype: {searchtype}")

    results = es.search(index=byom_index_name, body=query, _source=True)

    # Set a default value for num_results
    num_results = 0

    if results and results.get('hits') and results['hits'].get('hits'):
        # Limit the number of results displayed

        num_results = min(5, len(results['hits']['hits']))

        for i in range(num_results):
            if results['hits']['hits'][i]["_source"].get("passages") and len(
                    results['hits']['hits'][i]["_source"]["passages"]) > 0:
                first_passage_text = results['hits']['hits'][i]["_source"]["passages"][0]["text"]
            else:
                first_passage_text = "No passages text available"

            print(first_passage_text)

    else:
        print("No results found.")

    return get_chat_guidance_rag(user_query, azureclient, results, conversation_history)


def search_products_v2(es, user_query, searchtype, rrf_rank_constant, rrf_window_size):
    # Select the appropriate query building function based on searchtype
    if searchtype == "Vector":
        query = build_knn_query(user_query, build_vector(es, user_query))
    elif searchtype == "BM25":
        query = build_bm25_query(user_query)
    elif searchtype == "Reciprocal Rank Fusion":
        query = build_rrf_query(build_vector(es, user_query), user_query, rrf_rank_constant, rrf_window_size)
    elif searchtype == "Elser":
        query = build_elser_query(user_query)
    elif searchtype == "Vector OpenAI":
        # query = build_openai_query(get_embedding(user_query), run_ner_inference(es, user_query))
        query = build_openai_query(get_embedding(user_query), find_color_in_text(user_query),
                                   find_os_in_text(user_query))

    elif searchtype == "GenAI":
        print("GenAI Search Only")
    else:
        raise ValueError(f"Invalid searchtype: {searchtype}")

    results = es.search(index=byom_index_name, body=query, _source=True)

    # Set a default value for num_results
    num_results = 0

    if results and results.get('hits') and results['hits'].get('hits'):
        # Limit the number of results displayed

        num_results = min(5, len(results['hits']['hits']))

        for i in range(num_results):
            if results['hits']['hits'][i]["_source"].get("passages") and len(
                    results['hits']['hits'][i]["_source"]["passages"]) > 0:
                first_passage_text = results['hits']['hits'][i]["_source"]["passages"][0]["text"]
            else:
                first_passage_text = "No passages text available"

            print(first_passage_text)

    else:
        print("No results found.")

    blog_bodies = []
    urls = []
    titles = []
    passages_texts = []
    scores = []

    # Check if there are any hits
    if results['hits']['hits']:
        # Process up to the first 3 results
        for hit in results['hits']['hits'][:3]:  # Limit to first 3 results
            source = hit["_source"]

            # Extract and accumulate the body content
            blog_body = source.get("body_content", "No body content available")
            blog_bodies.append(blog_body)

            # Extract and accumulate the URL, preferring 'additional_urls' if available and non-empty
            additional_urls = source.get("additional_urls", [])
            url = additional_urls[0] if additional_urls else source.get("url", "No URL available")
            urls.append(url)

            # Extract and accumulate the title
            title = source.get("title", "No title available")

            st.session_state.titles.append(" Movie Title: " + title)

            # Extract and accumulate the first passage text if passages exist and are non-empty
            passages = source.get("passages", [])
            first_passage_text = passages[0].get("text",
                                                 "No passages text available") if passages else "No passages available"
            passages_texts.append(first_passage_text)

            # Extract and accumulate the score
            score = hit["_score"]
            scores.append(score)

    return titles
