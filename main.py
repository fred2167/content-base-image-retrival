import streamlit as st
import os
import helper
import histogram
import distance
import collections

def displayResults(queryState, paths, relevance_flag = False, num_columns = 3, num_rows = 3, ):

    img_idx = queryState * num_columns * num_rows
    
    for i in range(num_rows):
        cols = st.columns(num_columns)
        
        for i in range(num_columns):
            
            if img_idx >= len(paths):
                return
            
            with cols[i]:
                st.image(paths[img_idx], caption=f"{helper.getImageSortKey(paths[img_idx])}.jpg")
                
                if relevance_flag:
                    img_relevance_flag = st.checkbox("Relevance", key=img_idx+1)

                    img_feature_idx = helper.getImageSortKey(paths[img_idx]) - 1
                    if img_relevance_flag:
                        st.session_state["relevanceIdx"].add(img_feature_idx)
                    else:
                        if img_feature_idx in st.session_state["relevanceIdx"]:
                            st.session_state["relevanceIdx"].remove(img_feature_idx)

            img_idx += 1

def changeState(idx, step):
    def clearCache():
        for key in st.session_state.keys():
            del st.session_state[key]

    if idx not in st.session_state:
        clearCache()
        st.session_state[idx] = 0
        st.session_state["relevanceIdx"] = set()
        return 0
    else:
        if step < 0 and st.session_state[idx] == 0:
            return 0

        if step > 0 and st.session_state[idx] == 4:
            return 4

        st.session_state[idx] += step
        return st.session_state[idx]


if __name__ == "__main__":
   
    img_folder = os.path.join(os.getcwd(), "retrival-images")
    img_paths = helper.getImagePaths(img_folder)

    feature_fn_str = st.sidebar.radio("Features", ("Intensity", "Color Code", "Intensity + Color Code", "Neural Network"))
    relevance_flag = st.sidebar.checkbox("Relevance")
    # feature_fn_strs = st.sidebar.multiselect("Features", ("Intensity", "Color Code", "Neural Network"), default="Intensity")

    st.title('Content Base Image Retrival')

    queryIdx = st.number_input("image ID (between 1 and 100)", min_value=1, max_value=100, value=1, step=1)
    queryIdx -= 1 # translate from input to features idx and img_path idx
    st.image(img_paths[queryIdx])

    st.header("Results")
    
    features = histogram.getFeatures(feature_fn_str, img_paths)
    
    if feature_fn_str == "Neural Network":
        closest_match_paths = distance.localSensitiveHash(queryIdx, img_paths, features)
    else:
        
        weights = None
        if relevance_flag and len(st.session_state["relevanceIdx"]) > 0:
            relevantFeatures = helper.getRelevantFeatures(features, st.session_state["relevanceIdx"])
            weights = helper.getFeatureWeights(relevantFeatures)
            
        closest_match_paths = distance.manhantanDistance(queryIdx, img_paths, features, weights)

    # closest_match_paths = getClosetPaths(feature_fn_strs, queryIdx, img_paths)

    prev_flag = st.button("Previous")
    next_flag = st.button("Next Page")

    if next_flag:
        queryState = changeState(queryIdx, 1)
    elif prev_flag:
        queryState = changeState(queryIdx, -1)
    else:
        queryState = changeState(queryIdx, 0)
    
    displayResults(queryState, closest_match_paths, relevance_flag, 4, 5)

    st.sidebar.write(str(st.session_state["relevanceIdx"]))