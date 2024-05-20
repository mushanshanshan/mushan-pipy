
import torch
from torch.nn.functional import normalize
from time import time
import numpy as np
import math
import psutil
import torch

def check_available_ram(device="cpu"):
  """
  Returns available RAM on target device
  args:
    device:     str or torch.device
  """
  if isinstance(device, str):
    device = torch.device(device)
  elif isinstance(device, torch.device):
    device = device
  else:
    raise RuntimeError("`device` must be str or torch.device")
  
  if device.type == "cpu":
    return psutil.virtual_memory().available
  else:
    total = torch.cuda.get_device_properties(device).total_memory
    used = torch.cuda.memory_allocated(device)
    return total - used

def will_it_fit(size, device="cpu", safe_mode=True):
  """
  Returns True if an array of given byte size fits in target device.
  if self.safe_mode = False, this function simply compares the given byte size with the remaining RAM on target device. This option is faster, 
    but it doesn't take memory fragmentation into account. So it will still be possible to run out of memory.
  if self.safe_mode = True, it will try to allocate a tensor with the given size. if allocation fails, return False. 
    This option is recommended when the other option fails because of OOM.
  
  args:
    size:       int
    device:     str or torch.device
    safe_mode:  bool
  returns:
    result:     bool
  """
  if safe_mode:
    try:
      torch.empty(size, device=device, dtype=torch.uint8)
    except:
      return False
    return True
  else:
    return check_available_ram(device) >= size

def find_optimal_splits(n, get_required_memory, device="cpu", safe_mode=True):
  """
  Find an optimal number of split for `n`, such that `get_required_memory(math.ceil(n / n_split))` fits in target device's RAM.
  get_required_memory should be a fucntion that receives `math.ceil(n/n_split)` and returns the required memory in bytes.
  args:
      n:                      int
      get_required_memory:    function
      device:                 str or torch.device
      safe_mode:              bool
  returns:
      n_splits:               int
  """
  splits = 1
  sub_n = n
  break_next = False
  while True:
    if break_next:
      break
    if splits > n:
      splits = n
      break_next = True
    sub_n = math.ceil(n / splits)
    required_memory = get_required_memory(sub_n)
    if will_it_fit(required_memory, device):
      break
    else:
      splits *= 2
      continue
  return splits
def _kpp(data: torch.Tensor, k: int, sample_size: int = -1):
    """ Picks k points in the data based on the kmeans++ method.

    Parameters
    ----------
    data : torch.Tensor
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int
        sample data to avoid memory overflow during calculation

    Returns
    -------
    init : ndarray
        A 'k' by 'N' containing the initial centroids.

    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.
    .. [2] scipy/cluster/vq.py: _kpp
    """
    if sample_size is not None and sample_size > 0:
        data = data[torch.randint(0, int(data.shape[0]),
                                  [min(100000, data.shape[0])], device=data.device)]
    dims = data.shape[1] if len(data.shape) > 1 else 1
    init = torch.zeros((k, dims)).to(data.device)

    r = torch.distributions.uniform.Uniform(0, 1)
    for i in range(k):
        if i == 0:
            init[i, :] = data[torch.randint(data.shape[0], [1])]

        else:
            D2 = torch.cdist(init[:i, :][None, :], data[None, :], p=2)[0].amin(dim=0)
            probs = D2 / torch.sum(D2)
            cumprobs = torch.cumsum(probs, dim=0)
            init[i, :] = data[torch.searchsorted(
                cumprobs, r.sample([1]).to(data.device))]
    return init


def _krandinit(data: torch.Tensor, k: int, sample_size: int = -1):
    """Returns k samples of a random variable whose parameters depend on data.

    More precisely, it returns k observations sampled from a Gaussian random
    variable whose mean and covariances are the ones estimated from the data.

    Parameters
    ----------
    data : torch.Tensor
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int
        sample data to avoid memory overflow during calculation

    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids

    References
    ----------
    .. [1] scipy/cluster/vq.py: _krandinit
    """
    mu = data.mean(axis=0)
    if sample_size is not None and sample_size > 0:
        data = data[torch.randint(0, int(data.shape[0]),
                                  [min(100000, data.shape[0])], device=data.device)]
    if data.ndim == 1:
        cov = torch.cov(data)
        x = torch.randn(k, device=data.device)
        x *= np.sqrt(cov)
    elif data.shape[1] > data.shape[0]:
        # initialize when the covariance matrix is rank deficient
        _, s, vh = data.svd(data - mu, full_matrices=False)
        x = torch.randn(k, s.shape[0])
        sVh = s[:, None] * vh / torch.sqrt(data.shape[0] - 1)
        x = x.dot(sVh)
    else:
        cov = torch.atleast_2d(torch.cov(data.T))

        # k rows, d cols (one row = one obs)
        # Generate k sample of a random variable ~ Gaussian(mu, cov)
        x = torch.randn(k, mu.shape[0], device=data.device)
        x = torch.matmul(x, torch.linalg.cholesky(cov).T)
    x += mu
    return x


def _kpoints(data, k, sample_size=-1):
    """Pick k points at random in data (one row = one observation).

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 are assumed to describe one
        dimensional data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int (not used)
        sample data to avoid memory overflow during calculation

    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids

    """
    return data[torch.randint(0, data.shape[0], size=[k], device=data.device)]


init_methods = {
    "gaussian": _krandinit,
    "kmeans++": _kpp,
    "random": _kpoints,
}

class KMeans:
  '''
  Kmeans clustering algorithm implemented with PyTorch

  Parameters:
    n_clusters: int, 
      Number of clusters

    max_iter: int, default: 100
      Maximum number of iterations

    tol: float, default: 0.0001
      Tolerance
    
    verbose: int, default: 0
      Verbosity

    mode: {'euclidean', 'cosine'}, default: 'euclidean'
      Type of distance measure
      
    init_method: {'random', 'point', '++'}
      Type of initialization

    minibatch: {None, int}, default: None
      Batch size of MinibatchKmeans algorithm
      if None perform full KMeans algorithm
      
  Attributes:
    centroids: torch.Tensor, shape: [n_clusters, n_features]
      cluster centroids
  '''
  def __init__(self, n_clusters, max_iter=100, tol=0.0001, verbose=0, mode="euclidean", init_method="random", minibatch=None):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.init_method = init_method
    self.minibatch = minibatch
    self.lr_rate = 1

    if mode == 'cosine':
      self.sim_func = self.cos_sim
    elif mode == 'euclidean':
      self.sim_func = self.euc_sim
    else:
      raise NotImplementedError()
    
    self.centroids = None

  @staticmethod
  def cos_sim(a, b):
    """
      Compute cosine similarity of 2 sets of vectors

      Parameters:
      a: torch.Tensor, shape: [m, n_features]

      b: torch.Tensor, shape: [n, n_features]
    """
    return normalize(a, dim=-1) @ normalize(b, dim=-1).transpose(-2, -1)

  @staticmethod
  def euc_sim(a, b):
    """
      Compute euclidean similarity of 2 sets of vectors

      Parameters:
      a: torch.Tensor, shape: [m, n_features]

      b: torch.Tensor, shape: [n, n_features]
    """
    return 2 * a @ b.transpose(-2, -1) - (a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]

  def max_sim(self, a, b):
    """
      Compute maximum similarity (or minimum distance) of each vector
      in `a` with all of the vectors in `b`

      Parameters:
      a: torch.Tensor, shape: [n_samples, n_features]

      b: torch.Tensor, shape: [n_clusters, n_features]
    """
    device = a.device
    n_samples = a.shape[0]

    if device.type == 'cpu':
      sim = self.sim_func(a, b)
      max_sim_v, max_sim_i = sim.max(dim=-1)
      return max_sim_v, max_sim_i
    else:
      max_sim_v = torch.empty(n_samples, device=a.device, dtype=a.dtype)
      max_sim_i = torch.empty(n_samples, device=a.device, dtype=torch.int64)

      def get_required_memory(chunk_size):
        return chunk_size * a.shape[1] * b.shape[0] * a.element_size() + n_samples * 2 * 4
      
      splits = find_optimal_splits(n_samples, get_required_memory, device=a.device, safe_mode=True)
      chunk_size = math.ceil(n_samples / splits)

      for i in range(splits):
        if i*chunk_size >= n_samples:
          continue
        start, end = i * chunk_size, min((i + 1) * chunk_size, n_samples)
        sub_x = a[start: end]
        sub_sim = self.sim_func(sub_x, b)
        sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
        del sub_sim
        max_sim_v[start: end] = sub_max_sim_v
        max_sim_i[start: end] = sub_max_sim_i

      return max_sim_v, max_sim_i

  def fit_predict(self, X, centroids=None):
    """
      Combination of fit() and predict() methods.
      This is faster than calling fit() and predict() seperately.

      Parameters:
      X: torch.Tensor, shape: [n_samples, n_features]

      centroids: {torch.Tensor, None}, default: None
        if given, centroids will be initialized with given tensor
        if None, centroids will be randomly chosen from X

      Return:
      labels: torch.Tensor, shape: [n_samples]
    """
    assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
    assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
    assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "

    batch_size, emb_dim = X.shape
    device = X.device
    start_time = time()
    if centroids is None:
      self.centroids = init_methods[self.init_method](X, self.n_clusters, self.minibatch)
    else:
      self.centroids = centroids
    if self.minibatch is not None:
      num_points_in_clusters = torch.ones(self.n_clusters, device=device, dtype=X.dtype)

    closest = None
    arranged_mask = torch.arange(self.n_clusters, device=device)[:, None]
    for i in range(self.max_iter):
      iter_time = time()
      if self.minibatch is not None:
        x = X[np.random.choice(batch_size, size=[self.minibatch], replace=False)]
        closest = self.max_sim(a=x, b=self.centroids)[1]
        matched_clusters, counts = closest.unique(return_counts=True)
      else:
        x = X
        closest = self.max_sim(a=x, b=self.centroids)[1]

      expanded_closest = closest[None].expand(self.n_clusters, -1)
      mask = (expanded_closest==arranged_mask).to(X.dtype)
      c_grad = mask @ x / mask.sum(-1)[..., :, None]
      torch.nan_to_num_(c_grad)

      error = (c_grad - self.centroids).pow(2).sum()
      if self.minibatch is not None:
        lr = 1/num_points_in_clusters[:,None] * 0.9 + 0.1
        num_points_in_clusters[matched_clusters] += counts
      else:
        lr = 1
        
      lr *= self.lr_rate

      self.centroids = self.centroids * (1-lr) + c_grad * lr
      if self.verbose >= 2:
        print('iter:', i, 'error:', error.item(), 'time spent:', round(time()-iter_time, 4))
      if error <= self.tol:
        break

    if self.verbose >= 1:
      print(f'used {i+1} iterations ({round(time()-start_time, 4)}s) to cluster {batch_size} items into {self.n_clusters} clusters, error: {error.item()}')
    return closest

  def predict(self, X):
    """
      Predict the closest cluster each sample in X belongs to

      Parameters:
      X: torch.Tensor, shape: [n_samples, n_features]

      Return:
      labels: torch.Tensor, shape: [n_samples]
    """
    assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
    assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
    assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "

    return self.max_sim(a=X, b=self.centroids)[1]

  def fit(self, X, centroids=None):
    """
      Perform kmeans clustering

      Parameters:
      X: torch.Tensor, shape: [n_samples, n_features]
    """
    assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
    assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
    assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "

    self.fit_predict(X, centroids)