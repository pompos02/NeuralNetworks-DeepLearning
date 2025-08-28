# Neural Networks & Deep Learning Repository

This repository contains comprehensive implementations of machine learning and deep learning algorithms for classification and dimensionality reduction tasks using CIFAR-10 and MNIST datasets.

## Project Structure

### `src2/` - Classical & Modern Machine Learning
**CIFAR-10 Binary Classification (Cats vs Dogs)**

#### SVM Implementations
- **Linear/Polynomial/RBF SVMs** - Custom gradient descent with CuPy GPU acceleration
- **SMO Algorithm** - Sequential Minimal Optimization for all kernel types
- **Hyperparameter Tuning** - Grid search optimization for all SVM variants

#### Neural Networks
- **Multi-Layer Perceptron** - PyTorch implementation with custom hinge loss
- **K-Nearest Neighbors** - Multiple k values with Euclidean distance
- **Nearest Centroid Classifier** - Centroid-based classification

### `src3/` - Deep Learning & Dimensionality Reduction
**MNIST Digit Recognition & Reconstruction**

#### Autoencoder Architectures
- **Basic Autoencoder** - 784→128→32→128→784 architecture
- **Deep Autoencoder** - 784→512→256→64→256→512→784 architecture
- **PCA Comparison** - Classical dimensionality reduction baseline

#### Advanced Models
- **CNN Classifier** - Convolutional network for digit classification
- **t-SNE Visualization** - Latent space analysis and visualization
- **Reconstruction Analysis** - Quality assessment and per-digit performance

## Technical Features

### High-Performance Computing
- **GPU Acceleration** - CuPy integration for SVM training
- **Efficient Data Processing** - StandardScaler normalization, batch processing
- **Learning Rate Scheduling** - Adaptive learning rate strategies

### Advanced Analysis
- **Comparative Evaluation** - Autoencoder vs PCA reconstruction quality
- **Latent Space Visualization** - t-SNE embedding of learned representations
- **Cross-Architecture Testing** - Classification on reconstructed images
- **Statistical Analysis** - Per-class and per-digit performance metrics

## Performance Results

### CIFAR-10 Classification (Cats vs Dogs)
| Algorithm | Test Accuracy |
|-----------|---------------|
| MLP (PyTorch) | 65.50% |
| Linear SVM | 63.40% |
| KNN/NCC | ~58% |

### MNIST Results
| Model | Metric | Performance |
|-------|--------|-------------|
| CNN Classification | Accuracy | 98.89% |
| Deep Autoencoder | MSE | 0.0057 |
| Basic Autoencoder | MSE | 0.0063 |
| PCA (64 components) | MSE | 0.0090 |
| Classification on AE Reconstruction | Accuracy | 98.27% |
| Classification on PCA Reconstruction | Accuracy | 98.22% |

## Dependencies
```
torch, torchvision          # Deep learning framework
scikit-learn, numpy         # Classical ML algorithms
cupy-cuda                   # GPU acceleration
matplotlib, seaborn         # Visualization
tensorflow                  # Dataset loading
tqdm                        # Progress tracking
```

## Key Implementations

### SVM with Custom Optimization
- Hand-coded gradient descent with CuPy acceleration
- Multiple kernel implementations (Linear, Polynomial, RBF)
- SMO algorithm for efficient quadratic programming

### Autoencoder Architecture Comparison
- Systematic evaluation of shallow vs deep architectures
- Reconstruction quality analysis and latent space visualization
- Performance comparison with classical PCA dimensionality reduction

### Cross-Modal Evaluation
- Classification accuracy on autoencoder-reconstructed images
- Robustness analysis of learned representations
- Comparative study of neural vs classical dimensionality reduction

## Author
**Γιάννης Καραβέλλας** (Student ID: 4228)

## Documentation
- **Report2.pdf** - Classical ML implementations and SVM theory
- **Report3.pdf** - Deep learning architectures and autoencoder analysis

---
*Comprehensive implementation of classical and modern machine learning techniques with performance analysis and theoretical documentation.*