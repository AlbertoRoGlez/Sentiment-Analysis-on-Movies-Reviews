from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec


def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """
    # TODO
    corpus_vectors = np.zeros((len(corpus), num_features), dtype="float64")
    
    for i, doc in enumerate(corpus):
        feat_vec = np.zeros((num_features,), dtype="float64")
        nwords = 0
        
        for word in doc:
            if word in model.wv.index_to_key: 
                nwords = nwords + 1
                feat_vec = np.add(feat_vec, model.wv[word])
        
        if nwords:
            feat_vec = np.divide(feat_vec, nwords)
        
        corpus_vectors[i] = feat_vec
    
    return corpus_vectors
