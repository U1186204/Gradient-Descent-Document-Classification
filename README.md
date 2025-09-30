[![CI/CD Pipeline](https://img.shields.io/github/actions/workflow/status/U1186204/Gradient-Descent-Document-Classification/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI%2FCD%20Pipeline)](https://github.com/U1186204/Gradient-Descent-Document-Classification/actions/workflows/ci.yml)

# Gradient-Descent-Document-Classification

### Abstract
This project implements and evaluates a unigram language model trained with gradient descent. The script analyzes the model's performance by adjusting the number of iterations and learning rate, and visualizes the training loss in comparison to the theoretical minimum loss. The following charts compare the model's learned token probabilities against the optimal (MLE) probabilities and plot the model's training loss over time 

### Repository Structure
```txt
Gradient-Descent-Document-Classification
├── .github
│   └── workflows
│       └── ci.yml
├── image
│   └── Figure_1.png
├── instructions
│   └── gradient_descent.pdf
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
└── unigram_pytorch.py
```

**What this neural network does? What are the inputs and outputs of the learned function?**
A neural Netowrk purpose is to determine the probability of any single character appearing in a text, assuming each character's appearance is independent of any other. The network begins by guessing that all characters are equally likely and then uses gradient descent to adjust its internal parameters. It improves its guesses by comparing them against the actual character frequencies in the provided text until its learned probabilities closely match the real distribution. The input to the learned function is a single tensor containing the total count of each character in the vocabulary (including a combined count for all out-of-vocabulary characters) found in the entire document. The function's output is a single number representing the total log-probability of the entire text based on the model's learned probabilities; the training process works to make this number as large as possible.

**Describe how you could modify/augment this code to perform document classification**
To adapt this code for document classification there would need to be fundamentally a change to the model's structure and data processing. Instead of processing a single corpus, the input would become individual documents, typically represented as a vector which could need to be adjusted for word counts. The model's architecture would be augmented with at least one hidden layer to learn relationships between words. The final output layer would be replaced with a softmax activation function to produce a probability distribution across the desired document clusters (e.g., 'Legal Documents' vs. 'System Files' vs. 'Financial Statements'). The training process would use labeled data (documents paired with their correct categories) and a different loss function, such as cross-entropy, to measure the accuracy of its predictions and guide the learning process.
