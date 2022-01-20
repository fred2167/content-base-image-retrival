import numpy as np
import streamlit as st
import nn


def manhantanDistance(queryIdx, img_paths, features):

    query = features[queryIdx, :].reshape(1, -1)

    l1 = np.abs(query - features).sum(axis=1)
    smallestDistanceIndex = l1.argsort()

    results = [img_paths[i] for i in smallestDistanceIndex if i != queryIdx]
    return results

def localSensitiveHash(queryIdx, img_paths, features):
    
    lsh = _getHashTables(img_paths, features)
    
    results = lsh.query(queryIdx)

    return results
    
@ st.cache
def _getHashTables(img_paths, features):
    
    input_dim, hash_size, num_tables = 512, 12, 10

    lsh = nn.LSH(input_dim, hash_size, num_tables)
    
    for i in range(len(img_paths)):
        lsh.index(features[i,:], img_paths[i])
    
    return lsh
    

