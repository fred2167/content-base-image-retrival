import os
import glob
import streamlit as st

def getImageSortKey(path):
  token = path.split("/")[-1]
  idx = token[:token.find(".")]
  return int(idx)

@st.cache
def getImagePaths(img_folder):
    img_paths = glob.glob(img_folder + "/*.jpg")
    img_paths.sort(key = getImageSortKey)
    return img_paths