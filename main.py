import streamlit as st
import os
import helper
import histogram
import distance


def displayResults(queryState, paths, num_columns = 3, num_rows = 3):
    img_idx = queryState * num_columns * num_rows

    for i in range(num_rows):
        cols = st.columns(num_columns)
        
        for i in range(num_columns):
            
            if img_idx >= len(paths):
                return
            
            with cols[i]:
                st.image(paths[img_idx][0], caption=f"{helper.getImageSortKey(paths[img_idx][0])}.jpg: {paths[img_idx][1]:.4f}")

            img_idx += 1

def changeState(idx, step):
    def clearCache():
        for key in st.session_state.keys():
            del st.session_state[key]

    if idx not in st.session_state:
        clearCache()
        st.session_state[idx] = 0
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

    feature_fn_str = st.sidebar.radio("Features", ("Intensity", "Color Code"))
    features = histogram.getFeatures(feature_fn_str, img_paths)

    distance_fn_str = st.sidebar.radio("Distance Function", ("Manhantan Distance",))
    distance_fn = distance.getDistance_fn(distance_fn_str)
    st.title('Content Base Image Retrival')

    queryIdx = st.number_input("image ID (between 1 and 100)", min_value=1, max_value=100, value=1, step=1)
    queryIdx -= 1 # translate from input to features idx and img_path idx
    st.image(img_paths[queryIdx])

    st.header("Results")

    closestIdx, dists = distance_fn(queryIdx, features, 100)
    closest_match_paths = [(img_paths[j], dists[i]) for i, j in enumerate(closestIdx)]

    prev_flag = st.button("Previous")
    next_flag = st.button("Next Page")

    if next_flag:
        queryState = changeState(queryIdx, 1)
    elif prev_flag:
        queryState = changeState(queryIdx, -1)
    else:
        queryState = changeState(queryIdx, 0)
    
    displayResults(queryState, closest_match_paths, 4, 5)
