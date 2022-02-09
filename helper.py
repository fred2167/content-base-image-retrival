import os
import glob
from turtle import pos
import numpy as np
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

def getRelevantFeatures(features, featureIdx):
    idx = np.array(list(featureIdx))
    return features[idx]

def getFeatureWeights(posFeatures, eps = 1e-6):

    if len(posFeatures) == 0:
      return None

    std = posFeatures.std(axis= 0)
    avg = posFeatures.mean(axis= 0)

    idxToZero = np.logical_and(std == 0, avg == 0)
    idxToAvg = np.logical_and(std == 0, avg != 0)
    minStd = np.min(std[std != 0]) if len(std[std != 0]) > 0 else 0

    weights =  1 / (std + eps)
    weights[idxToZero] = 0
    weights[idxToAvg] = minStd / 2

    if weights.sum() != 0:
      weights /= weights.sum()
    else:
      N = posFeatures.shape[1]
      weights = np.ones((1, N)) / N
      
    return weights


relevance_help = "change query image to reset relevance"