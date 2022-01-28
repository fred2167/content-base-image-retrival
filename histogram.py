import streamlit as st
import numpy as np
import imageio
import nn

@st.cache
def getFeatures(feature_fn_str, img_paths):
  
  if feature_fn_str == "Intensity":
    features = intensityHistogram(img_paths)
  elif feature_fn_str == "ColorCode":
    features = colorCodeHistogram(img_paths)
  elif feature_fn_str == "Neural Network":
    features = neuralNetworkHistogram(img_paths)
  else:
    features = intensityColorCodeHistogram(img_paths)

  return features

def intensityHistogram(img_paths):

  features = []
  rgbWeights = np.array([.299, .587, .114]).reshape(3, 1, 1)
  bins = list(range(0, 250, 10)) + [255]

  for path in img_paths:

    img = imageio.imread(path).transpose(2,0,1) # read image and move channel to the first dimension

    intensity = (img * rgbWeights).sum(axis=0) # combine RGB with weights and sum up as intensity

    hist, _ = np.histogram(intensity, bins=bins) # calculate histagram with predifine bin range (25 bins)

    hist = hist / intensity.size # normalize each bin with the size of the image to get probability distribution
    
    # assert  1 - hist.sum() < 1e-5, f"{hist.sum()}" # checking sum of probability distribution equal to 1

    features.append(hist)
  
  features = np.stack(features)
  return features
  

def colorCodeHistogram(img_paths):
  
  features = []
  weights = np.array([[2**4, 2**2, 1]]).reshape(3, 1, 1)

  for path in img_paths:

    img = imageio.imread(path).transpose(2, 0, 1) # read image and move channel to the first dimension

    mostSignificantBits = img // 2**6 # right shift 6 bits

    colorCode = (mostSignificantBits * weights).sum(axis=0) # translate bits to values

    hist = np.bincount(colorCode.reshape(-1, ), minlength=64) # create histgram with 64 bins

    hist = hist / colorCode.size # normalize with size of image to get probability distribution

    # assert  1 - hist.sum() < 1e-5, f"{hist.sum()}" # checking sum of probability distribution equal to 1

    features.append(hist)
  
  features = np.stack(features)

  return features

def intensityColorCodeHistogram(img_paths):
  intensityFeat = intensityHistogram(img_paths)
  colorCodeFeat = colorCodeHistogram(img_paths)

  features = np.concatenate((intensityFeat, colorCodeFeat), axis= 1)

  # whiten : mean = 0, std = 1
  features = (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, ddof=1, keepdims=True) + 1e-8) 
  return features


def neuralNetworkHistogram(img_paths):
  model = nn.PretrainModel()
  tensors = nn.imgPathsToTensors(img_paths)
  features = model(tensors)
  return features