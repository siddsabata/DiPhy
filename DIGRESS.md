# DiGress: Discrete Denoising Diffusion for Graph Generation

## Paper Overview

**Title:** DiGress: Discrete Denoising Diffusion for Graph Generation  
**Authors:** Clément Vignac*, Igor Krawczuk*, Antoine Siraudin, Bohan Wang, Volkan Cevher, Pascal Frossard (EPFL)  
**Conference:** ICLR 2023  
**GitHub:** https://github.com/cvignac/DiGress  

### Key Contribution
DiGress introduces the first discrete denoising diffusion model for generating graphs with categorical node and edge attributes. Unlike previous continuous diffusion approaches that destroy graph structure, DiGress operates directly in discrete space, preserving sparsity and structural properties during the diffusion process.

## Background & Motivation

### Problem with Continuous Diffusion for Graphs
Previous graph diffusion models (GDSS, GraphGDP) embed graphs in continuous space and add Gaussian noise, which:
- Destroys graph sparsity (creates dense noisy graphs)
- Eliminates structural information (connectivity, cycles)
- Makes it difficult for networks to capture graph properties

### Discrete Diffusion Advantages
- Preserves graph sparsity during noise process
- Maintains structural properties in intermediate states
- Allows computation of graph-theoretic features at each step
- More natural for categorical node/edge attributes

## Technical Methodology

### 1. Graph Representation
- **Nodes:** Categorical attributes `X ∈ R^(n×a)` where `a` is number of node types
- **Edges:** Categorical attributes `E ∈ R^(n×n×b)` where `b` is number of edge types
- **Encoding:** One-hot encoding for both nodes and edges
- **Undirected graphs:** Apply noise only to upper triangular part, then symmetrize

### 2. Discrete Diffusion Process

#### Forward Process (Adding Noise)
```
q(G^t|G^(t-1)) = (X^(t-1)Q^t_X, E^(t-1)Q^t_E)
q(G^t|G) = (XQ̄^t_X, EQ̄^t_E)
```

Where:
- `Q^t_X`, `Q^t_E` are transition matrices for nodes and edges
- `Q̄^t = Q^1...Q^t` (cumulative transition matrix)
- Each node and edge diffuses independently

#### Two Noise Models:

**Uniform Transition (Baseline):**
```
Q^t = α_t I + (1 - α_t)(1_d 1'_d)/d
```

**Marginal Transition (Proposed):**
```
Q^t_X = α_t I + β_t 1_a m'_X
Q^t_E = α_t I + β_t 1_b m'_E
```
Where `m_X`, `m_E` are marginal distributions from training data.

### 3. Reverse Process (Denoising)

#### Network Architecture
- **Base:** Graph Transformer with self-attention
- **Input:** Noisy graph G^t, timestep t, auxiliary features
- **Output:** Predicted clean graph probabilities

#### Reverse Sampling
```python
p_θ(G^(t-1)|G^t) = ∏_i p_θ(x^(t-1)_i|G^t) ∏_(i,j) p_θ(e^(t-1)_ij|G^t)

p_θ(x^(t-1)_i|G^t) = Σ_x q(x^(t-1)_i|x_i=x, x^t_i) p̂^X_i(x)
```

### 4. Structural Feature Augmentation

DiGress augments the denoising network with auxiliary features:

**Graph-Theoretic Features:**
- Cycle counts (k=3,4,5,6) using matrix powers
- Node-level: cycles containing each node
- Graph-level: total cycle counts

**Spectral Features:**
- Graph Laplacian eigenvalues/eigenvectors
- Connected components count
- Eigenvector-based node features

**Molecular Features (for chemistry):**
- Atom valency
- Molecular weight

### 5. Conditional Generation

DiGress supports conditioning on graph-level properties through **discrete regressor guidance**:

1. Train regressor `g_η(G^t) → ŷ` to predict properties from noisy graphs
2. During sampling, modulate probabilities:
   ```
   p_η(ŷ|G^(t-1)) ∝ exp(-λ⟨∇_G ||ŷ - y||^2, G^(t-1)⟩)
   ```
3. Sample from: `p_θ(G^(t-1)|G^t) p_η(ŷ|G^(t-1))`

## Key Theoretical Properties

### Permutation Equivariance
**Lemma 3.1:** DiGress is permutation equivariant
- Architecture: Graph transformer with permutation-equivariant layers
- Loss: Decomposes as sum over individual nodes/edges
- No explicit graph matching required

### Exchangeability
**Lemma 3.3:** Generated distributions are exchangeable
- All permutations of generated graphs are equally likely
- Enables tractable likelihood computation

### Optimal Prior (Marginal Transitions)
**Theorem 4.1:** Marginal distributions provide optimal factorized prior
- Minimizes L2 distance to true data distribution
- Justifies the marginal transition model

## Implementation Details

### Training Algorithm
```python
def train_step(G):
    t = uniform_sample(1, T)
    G_t = add_noise(G, t)  # Sample noisy graph
    z = compute_features(G_t, t)  # Structural features
    p_pred = model(G_t, z, t)  # Forward pass
    loss = cross_entropy(p_pred, G)  # Classification loss
    return loss
```

### Sampling Algorithm  
```python
def sample(n_nodes):
    G_T = sample_prior(n_nodes)  # Random graph
    for t in range(T, 1, -1):
        z = compute_features(G_t, t)
        p_pred = model(G_t, z, t)
        
        # Compute posterior probabilities
        p_posterior = compute_posterior(p_pred, G_t, t)
        G_t_minus_1 = sample_categorical(p_posterior)
    
    return G_0
```

### Network Architecture
- **Layers:** Multiple graph transformer layers
- **Attention:** Node self-attention incorporating edge features
- **Features:** Node (X), edge (E), graph-level (y), time (t)
- **Updates:** Residual connections, layer normalization
- **Complexity:** O(n²) per layer due to attention and edge predictions

## Experimental Results

### Datasets Evaluated
1. **Abstract Graphs:** Stochastic Block Model, Planar Graphs
2. **Small Molecules:** QM9 (≤9 heavy atoms)
3. **Large Molecules:** MOSES (1M molecules), GuacaMol (1.3M molecules)

### Key Performance Metrics
- **Validity:** Proportion of chemically valid molecules
- **Uniqueness:** Proportion of non-duplicate molecules  
- **Novelty:** Proportion not in training set
- **Graph Statistics:** Degree distribution, clustering, orbit counts
- **Chemical Properties:** FCD, scaffold similarity, molecular filters

### Major Results

**QM9 Results:**
- Validity: 99.0% (vs 95.7% GDSS)
- Training Speed: 7× faster than continuous model
- Near-perfect uniqueness (96.2%)

**Large-Scale Results:**
- **First one-shot model** to scale to MOSES/GuacaMol
- Matches autoregressive model performance (GraphINVENT)
- Significantly outperforms other one-shot methods

**Abstract Graphs:**  
- 3× improvement in validity on planar graphs
- Strong performance on stochastic block model

## Working with the Codebase

### Repository Structure
```
DiGress/
├── digress/
│   ├── models/          # Model architectures
│   ├── diffusion/       # Diffusion process implementation  
│   ├── datasets/        # Dataset loading and preprocessing
│   ├── utils/          # Utilities and metrics
│   └── main.py         # Training/sampling scripts
├── configs/            # Configuration files
├── data/              # Dataset storage
└── outputs/           # Model checkpoints and results
```

### Key Classes to Understand

**1. Diffusion Process (`diffusion/`)**
- `DiscreteDiffusion`: Main diffusion class
- `NoiseSchedule`: Implements different noise models
- `utils.py`: Posterior computation, sampling utilities

**2. Models (`models/`)**
- `GraphTransformer`: Core denoising network
- `layers.py`: Graph transformer layers with attention
- `features.py`: Structural feature computation

**3. Datasets (`datasets/`)**
- `QM9Dataset`, `MOSESDataset`: Molecular data handling
- `AbstractDataset`: General graph datasets
- Data preprocessing and graph conversion utilities

### Configuration System
DiGress uses Hydra for configuration management:
```yaml
# config/qm9.yaml
model:
  type: 'digress'
  n_layers: 4
  hidden_dim: 256
  
diffusion:
  noise_schedule: 'cosine'
  timesteps: 1000
  noise_type: 'marginal'  # or 'uniform'
  
dataset:
  name: 'qm9'
  batch_size: 32
```

### Training a Model
```bash
# Train on QM9
python main.py dataset=qm9 model=digress

# Train with custom config
python main.py dataset=moses model=digress diffusion.timesteps=500

# Resume from checkpoint
python main.py dataset=qm9 resume=path/to/checkpoint.pt
```

### Key Implementation Notes

**1. Feature Computation:**
- Cycle counts computed via matrix powers (expensive for large graphs)
- Spectral features require eigendecomposition O(n³)
- Consider computational cost vs. performance trade-off

**2. Memory Requirements:**  
- O(n²) memory per layer due to attention mechanism
- Edge predictions require O(n²) outputs
- Batch size limited by graph size

**3. Noise Schedule:**
- Cosine schedule works well: `α_t = cos²(π(t/T + s)/(2(1+s)))`
- Small offset `s` prevents complete noise at t=T

**4. Loss Weighting:**
- Balance node vs edge loss with hyperparameter `λ`
- Typically need higher weight on edges due to sparsity

### Extending DiGress

**Adding New Datasets:**
1. Implement dataset class inheriting from base
2. Define graph conversion (molecules ↔ graphs)
3. Add dataset config file
4. Register in dataset factory

**New Conditional Properties:**
1. Train property regressor on your targets
2. Implement guidance computation in sampling loop
3. Add property prediction during data preprocessing

**Architecture Modifications:**
1. Modify `GraphTransformer` for new attention mechanisms
2. Add feature types in `features.py`
3. Update config schema for new hyperparameters

### Common Issues & Solutions

**Memory Issues:**
- Reduce batch size or max graph size
- Disable structural features for large graphs
- Use gradient checkpointing

**Training Instability:**
- Check loss weighting between nodes/edges
- Verify noise schedule parameters
- Monitor gradient norms

**Poor Sample Quality:**
- Try marginal transitions vs uniform
- Add structural features if computationally feasible
- Increase model capacity (layers, hidden dim)

**Invalid Molecules:**
- Check RDKit sanitization settings
- Verify bond valency constraints in data
- Consider post-processing validity filters

This comprehensive guide should enable ML researchers to understand, implement, and extend DiGress for their graph generation tasks.