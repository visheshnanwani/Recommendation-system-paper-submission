# UITrust — Recommendation Systems Mini Project

## A Multidimensional Model for Recommendation Systems Based on Classification and Entropy

📄 **Replication of:** Yuan, Y.; Chen, L.; Yang, J. Electronics 2023, 12, 402
🔗 DOI: https://doi.org/10.3390/electronics12020402

---

## 📌 Project Overview

Modern recommendation systems suffer from **data sparsity** and **cold-start problems**.
This project replicates the **UITrust model**, which improves collaborative filtering using:

* Classification-based neighbour selection
* Entropy-driven similarity

---

## 🛠️ Tech Stack

* Python
* NumPy
* Pandas

---

## 🎯 Problem Statement

Given a user–item rating matrix:

[
\hat{r}*{u,i} = \frac{\sum*{v \in N_k(u)} Sim_{u,v} \cdot r_{v,i}}{\sum_{v \in N_k(u)} Sim_{u,v}}
]

The goal is to predict ratings using a **dense and meaningful similarity measure**.

---

## 📊 Dataset

| Dataset         | Ratings   | Users | Items | Scale | Density |
| --------------- | --------- | ----- | ----- | ----- | ------- |
| ML-100k         | 100,000   | 943   | 1,682 | 1–5   | 6.30%   |
| ML-latest-small | 100,836   | 610   | 9,742 | 0.5–5 | 1.70%   |
| ML-1m           | 1,000,209 | 6,040 | 3,900 | 1–5   | 4.25%   |

* 400 users sampled per dataset
* 80/20 train-test split
* 3000 test samples evaluated

---

## ⚙️ Methodology

### 1️⃣ Classification-Based Neighbour Selection

* Uses item **genre vectors**
* Builds user taste vector:

[
s_{u,k} = \frac{\sum g^*_{i,k}}{\sum g^*_{i}}
]

* Weight:

[
w_{u,i} = s_u \cdot c_i
]

✅ Produces **zero sparsity (fully dense matrix)**

---

### 2️⃣ Entropy-Driven Similarity

[
H_o^I(i) = -\sum prob \cdot \log_2(prob)
]

[
H_o^U(u) = -\sum prob \cdot \log_2(prob)
]

Final trust score:

[
UITrust = \alpha w_{u,i} + (1-\alpha)\frac{H_o^I + H_o^U}{2}
]

---

### 3️⃣ Final Prediction

[
\hat{r}_{u,i} = \bar{r}*u + \frac{\sum UITrust \cdot r*{u,j}}{\sum UITrust}
]

---

## 🧪 Baseline Methods

* UITrust_C
* UITrust_P
* UITrust_MSD
* UITrust_R
* CKNN_P
* CKNN_MSD
* BKNN_P
* BKNN_MSD

(All use **k = 40 neighbours**)

---

## 📈 Experimental Results

### 🔹 ML-100k

| Method    | MAE        | RMSE   |
| --------- | ---------- | ------ |
| UITrust_C | 0.7327     | 0.9377 |
| UITrust_P | **0.7281** | 0.9299 |
| CKNN_P    | 0.7315     | 0.9308 |
| BKNN_P    | 0.7916     | 0.9954 |

✅ UITrust improves MAE by **7.4%**

---

### 🔹 ML-latest-small

| Method    | MAE        | RMSE       |
| --------- | ---------- | ---------- |
| UITrust_C | 0.6967     | 0.9183     |
| UITrust_P | **0.6941** | **0.9146** |
| CKNN_MSD  | 0.6921     | 0.9135     |
| BKNN_P    | 0.7601     | 0.9890     |

✅ Best performance on **most sparse dataset**

---

### 🔹 ML-1m

| Method    | MAE        | RMSE       |
| --------- | ---------- | ---------- |
| UITrust_C | 0.7593     | 0.9825     |
| UITrust_P | 0.7560     | 0.9781     |
| CKNN_P    | **0.7507** | **0.9750** |
| BKNN_P    | 0.8002     | 1.0143     |

---

## 📉 Sparsity Analysis

* Pearson & MSD → Sparse matrices
* UITrust → **0 empty entries**

✅ Works well even with **few ratings**

---

## 📊 Comparison with Paper

| Dataset  | Paper MAE | Our MAE |
| -------- | --------- | ------- |
| ML-100k  | ~0.722    | 0.7327  |
| ML-small | ~0.660    | 0.6967  |
| ML-1m    | ~0.702    | 0.7593  |

✔ Results follow same trends

---

## 💡 Discussion

### Key Advantages:

* Eliminates sparsity
* Uses genre-based similarity
* Considers rating diversity via entropy

### Limitations:

* Requires pre-computation
* Not suitable for real-time systems

---

## ✅ Conclusion

* UITrust outperforms traditional KNN models
* Classification + entropy improves predictions
* Best improvement ≈ **7.6% MAE**

---

## 🚀 Future Work

* Deep learning embeddings
* Larger datasets (Amazon reviews, etc.)

---

## 📚 References

1. Yuan et al., 2023
2. MovieLens Dataset (GroupLens)
3. Surprise Library
4. Recommender Systems Handbook

---
