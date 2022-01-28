import torch
import torch.nn as nn
import torchvision
import imageio
import numpy as np
import collections

class PretrainModel(nn.Module):

  def __init__(self):
    super().__init__()
    pretrain = torchvision.models.resnet34(pretrained=True)
    self.model = nn.Sequential(*list(pretrain.children())[:-1])
    self.model = self.model.to(torch.float32)
  def forward(self, input):
  
    with torch.no_grad():
      feats = self.model(input)
      feats = feats.squeeze().detach()
      return feats.numpy()

class HashInfo:

  def __init__(self):
    self.features = []
    self.paths = []


class LSH:

  def __init__(self,input_dim= 512, hash_size= 10, num_table= 1):
    np.random.seed(1234)
    self.tables = [np.random.randn(hash_size, input_dim) for _ in range(num_table)]
    self.hashToImages = collections.defaultdict(lambda: HashInfo())
    self.features = []
  def _npArrayToHash(self, arr):
      return tuple(map(int, arr.squeeze().tolist()))
    
  def _featureToHashes(self, feature):

    if len(feature.shape) == 1:
      feature = feature.reshape(-1,1)

    hashes = []
    for table in self.tables:
      binary = table.dot(feature)
      binary[binary > 0] = 1
      binary[binary < 0] = 0
      hash = self._npArrayToHash(binary)
      hashes.append(hash)

    return hashes

  def index(self, feature, path):

    self.features.append(feature)
    hashes = self._featureToHashes(feature)
    for hash in hashes:
      self.hashToImages[hash].features.append(feature)
      self.hashToImages[hash].paths.append(path)

  def query(self, queryIdx, topk = None):

    query = self.features[queryIdx]
    hashes = self._featureToHashes(query)
    queyFeatures = []
    queryPaths = []
    for hash in hashes:
      info = self.hashToImages[hash]

      queyFeatures.extend(info.features)
      queryPaths.extend(info.paths)

    queyFeatures = np.stack(queyFeatures)

    # calculate distances
    query = query.reshape(1, -1)
    dists = ((query - queyFeatures)**2).sum(axis=1)** 0.5

    # sort paths by distances in acced order
    sortIdx = dists.argsort()
    sortQueryPaths = dict.fromkeys([queryPaths[i] for i in sortIdx]) # remove duplicate while preserve orders
    results = list(sortQueryPaths.keys())
    return results[1:] if topk is None else results[:topk]


def imgPathsToTensors(image_paths):
  tensors = []
  transformers = torchvision.transforms.Resize((224,224))
  for path in image_paths:
    tensor = torchvision.io.read_image(path)
    tensor = transformers(tensor)
    tensor = tensor / 255
    tensors.append(tensor)

  tensors = torch.stack(tensors)
  return tensors