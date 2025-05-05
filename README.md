# ANN-Hindi-Character-Classification-in-C-from-scratch
A neural network built from scratch in C++ to classify handwritten Hindi characters using a custom dataset.
# 🧠 Hindi Character Classification using ANN from Scratch in C++

This project implements a **4-layer artificial neural network (ANN)** completely from scratch in **C++** to classify **handwritten Hindi characters**. From collecting the dataset to deriving backpropagation gradients by hand, this project is focused on understanding the inner mechanics of neural networks — not just using them.

---

## 📌 Project Overview

- 🖋️ Created a custom dataset of **handwritten Hindi characters**
- 🧮 Implemented a **fully connected ANN** with:
  - Input Layer: 225 (15×15 grayscale image)
  - Hidden Layers: 2 × 15 neurons
  - Output Layer: 10 classes (one-hot)
- 💻 Entire model written in **C++**, without any ML frameworks
- 🔁 Implemented forward propagation, sigmoid/leaky ReLU, softmax, MSE and cross-entropy loss
- 📉 Used **gradient descent with momentum** and performed extensive hyperparameter tuning

---

## 📐 ANN Architecture

<p align="center">
  <img src="images/Screenshot 2025-05-04 202549.png" width="600" alt="Architecture Diagram">
</p>

---

## 🧮 Training & Optimization Highlights

- Initially trained with **MSE loss** + **sigmoid**, later switched to:
  - **Cross-entropy loss** + **softmax**
  - **Leaky ReLU** to overcome vanishing gradients
- Manually derived backpropagation gradients for all layers
- Implemented **momentum**-based gradient updates
- Explored hyperparameters: learning rate, momentum, epochs

---

## 📊 Results

### 🔁 Epochs vs Training Error and Testing Accuracy

<p align="center">
  <img src="images/Screenshot 2025-05-01 000352.png" width="600" alt="Epoch vs Error/Accuracy">
</p>

---

### 📈 3D Plot: Epochs vs Learning Rate vs Error/Accuracy

<p align="center">
  <img src="images/Screenshot 2025-05-01 000411.png" width="500" alt="3D Hyperparameter Plot">
</p>

---

### 📉 Confusion Matrix on Validation Set

<p align="center">
  <img src="images/Screenshot 2025-05-05 121605.png" width="300" alt="Confusion Matrix">
</p>

- **Validation Accuracy:** `60.07%`
- Errors mostly occurred in visually similar characters
- Demonstrates meaningful pattern learning

---

## 📂 Folder Structure

📁 training/
├── main_training.cpp
└── weights.bin # Binary file after training

📁 testing/
├── main_testing.cpp
└── weights.bin # Used for inference

📁 notes/
└── handwritten_derivations.png

📁 results/
├── loss_accuracy_plot.png
├── 3d_plot.png
└── confusion_matrix.png

---

## 💡 Key Learnings

- Backpropagation is easier to understand once you derive and code it yourself.
- MSE isn't suited for multi-class classification — switching to cross-entropy was essential.
- Sigmoid is vulnerable to vanishing gradients; leaky ReLU solved it for deeper networks.
- Momentum accelerated convergence and reduced the noise in weight updates.
- Generalization is bounded by data quantity and quality; this project hit ~60% validation accuracy with ~3500 images.

---

## 📽️ Demo & Notes

- 🎥 *A screen recording video (training + testing)* will be pinned on LinkedIn & is also available in the result folder.
- ✍️ *Handwritten derivations and architecture diagrams* are available in the `notes/` folder.

---

## 📌 TODO / Future Work

- Expand dataset to include more characters
- Try regularization (dropout/L2) & will try to build a cnn from scratch
- Port the model to Python/CUDA for speedup
- Compare with a PyTorch equivalent implementation

---

## 🙋‍♂️ Author

**Sarthak patial**  
`Computer Science Undergraduate`  
xarthak@proton.me , connect me over linkeden

