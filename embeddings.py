import numpy as np

sentence = "I like to code"

words = set(sentence.split())

word2idx = {}
idx2word = {}

for i, word in enumerate(words):
    word2idx[word] = i
    idx2word[i] = word

embedding_dim = 4

vector_embeddings = np.random.randn(len(words), embedding_dim)

def convert_word(word: str):
    index = word2idx[word]
    return vector_embeddings[index]

print(convert_word("like"))

sentence_list = []

for word in sentence.split():
    sentence_list.append(convert_word(word))

sentence_embeddings = np.array(sentence_list)

position_embeddings = np.random.randn(len(sentence.split()), embedding_dim)

for i in range(len(sentence.split())):
    sentence_embeddings[i] += position_embeddings[i]
    
num_heads = 2
head_dim = embedding_dim // num_heads

head_outputs = []

for i in range(num_heads):
    W_q = np.random.randn(embedding_dim, head_dim)
    W_k = np.random.randn(embedding_dim, head_dim)
    W_v = np.random.randn(embedding_dim, head_dim)

    Q = sentence_embeddings @ W_q
    K = sentence_embeddings @ W_k
    V = sentence_embeddings @ W_v

    scores = (Q @ K.T) / np.sqrt(head_dim)

    exp_scores = np.exp(scores)
    row_sums = np.sum(exp_scores, axis=1, keepdims=True)
    attention_weights = exp_scores / row_sums

    output = attention_weights @ V

    head_outputs.append(output)

multi_head_output = np.concatenate(head_outputs, axis=1)

W_o = np.random.randn(embedding_dim, embedding_dim)
final_output = multi_head_output @ W_o

print(final_output)

def layer_norm(x, gamma, beta):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + 1e-5)
    return gamma * normalized + beta

gamma1 = np.ones(embedding_dim)
beta1 = np.zeros(embedding_dim)
gamma2 = np.ones(embedding_dim)
beta2 = np.zeros(embedding_dim)

x1 = layer_norm(sentence_embeddings + final_output, gamma1, beta1)

hidden_dim = embedding_dim * 4
W1 = np.random.randn(embedding_dim, hidden_dim)
b1 = np.random.randn(hidden_dim)

hidden = np.maximum(0, x1 @ W1 + b1)

W2 = np.random.randn(hidden_dim, embedding_dim)
b2 = np.random.randn(embedding_dim)
ffn_output = hidden @ W2 + b2

x2 = layer_norm(x1 + ffn_output, gamma2, beta2)

print(x2)
