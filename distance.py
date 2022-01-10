import numpy as np
import streamlit as st

@st.cache
def getDistance_fn(distance_fn_str):
        return manhantanDistance
    

def manhantanDistance(queryIdx, features, topk = 10):

    query = features[queryIdx, :].reshape(1, -1)

    l1 = np.abs(query - features).sum(axis=1)
    smallestDistanceIndex = l1.argsort()[1:topk + 1]

    return smallestDistanceIndex.tolist(), l1[smallestDistanceIndex]
