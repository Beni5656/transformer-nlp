import torch

sentences = [
    "I like to code",
    "I like to read",
    "I like to run",
    "the cat likes to sleep",
    "the dog likes to run",
    "she likes to code",
    "he likes to read",
    "they like to sleep",
]

all_words = set()
for sentence in sentences:
    for word in sentence.split():
        all_words.add(word)

word2idx = {}
idx2word = {}

for i, word in enumerate(all_words):
    word2idx[word] = i
    idx2word[i] = word

training_pairs = []
for sentence in sentences:
    words = sentence.split()
    for i in range(1, len(words)):
        input_words = words[:i]
        target_word = words[i]
        training_pairs.append((input_words, target_word))

vocab_size = len(word2idx)
embedding_dim = 8

max_sequence_length = 0

for sentence in sentences:
    words = sentence.split()
    if len(words) > max_sequence_length:
        max_sequence_length = len(words)

word_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
position_embedding = torch.nn.Embedding(max_sequence_length, embedding_dim)

class TransformerBlock(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.W_q_layers = torch.nn.ModuleList([])
        self.W_k_layers = torch.nn.ModuleList([])
        self.W_v_layers = torch.nn.ModuleList([])

        for i in range(self.num_heads):
            self.W_q_layers.append(torch.nn.Linear(embedding_dim, self.head_dim, bias=False))
            self.W_k_layers.append(torch.nn.Linear(embedding_dim, self.head_dim, bias=False))
            self.W_v_layers.append(torch.nn.Linear(embedding_dim, self.head_dim, bias=False))

        self.W_o = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.W1 = torch.nn.Linear(embedding_dim, embedding_dim * 4, bias=True)
        self.W2 = torch.nn.Linear(embedding_dim * 4, embedding_dim, bias=True)

        self.layerNorm1 = torch.nn.LayerNorm(embedding_dim)
        self.layerNorm2 = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x):
        head_outputs = []
        for i in range(self.num_heads):
            Q = self.W_q_layers[i](x)
            K = self.W_k_layers[i](x)
            V = self.W_v_layers[i](x)

            scores = (Q @ K.T) / (self.head_dim ** 0.5)
            attention_weights = torch.softmax(scores, dim=-1)
            output = attention_weights @ V
            head_outputs.append(output)

        multiple_head_embedding = torch.cat(head_outputs, dim=1)

        attn_output = self.W_o(multiple_head_embedding)

        x1 = self.layerNorm1(x + attn_output)
        hidden = torch.relu(self.W1(x1))
        ffn_output = self.W2(hidden)
        x2 = self.layerNorm2(x1 + ffn_output)
        return x2


output_layer = torch.nn.Linear(embedding_dim, vocab_size)
block = TransformerBlock(embedding_dim, num_heads=2)

def predict_next_word(input_words):
    input_indicies = torch.tensor([word2idx[word] for word in input_words])
    word_vectors = word_embedding(input_indicies)

    positions = torch.arange(len(input_words))
    position_vectors = position_embedding(positions)

    x = word_vectors + position_vectors

    block_output = block(x)
    last_vector = block_output[-1]

    logits = output_layer(last_vector)
    return logits 

loss_fn = torch.nn.CrossEntropyLoss()

all_params = (
    list(word_embedding.parameters())
    + list(position_embedding.parameters()) 
    + list(block.parameters())
    + list(output_layer.parameters())
)

optimizer = torch.optim.Adam(all_params, lr=0.01)

num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0
    for input_words, target_word in training_pairs:
        optimizer.zero_grad()

        logits = predict_next_word(input_words)
        target_idx = torch.tensor(word2idx[target_word])
        loss = loss_fn(logits, target_idx)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"epoch {epoch}, loss: {total_loss:.4f}")

