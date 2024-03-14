
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train = pd.read_parquet("train-00000-of-00001.parquet")
START_TOKEN = ''
PADDING_TOKEN = ''
END_TOKEN = ''
english_french_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@', 
                        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                        'Y', 'Z',
                        '[', '\\', ']', '^', '_', '`', 
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                        'y', 'z', ';', '-',
                        '{', '|', '}', '~', 'à', 'â', 'ä', 'æ',
                        'ç', 'é', 'è', 'ê', 'ë', 'î', 'ï', 
                        'ô', 'ö', 'ù', 'û', 'ü', 'œ', PADDING_TOKEN, END_TOKEN]

itos = {k:v for k,v in enumerate(english_french_vocabulary)}
stoi = {v:k for k,v in enumerate(english_french_vocabulary)}
train = train.iloc[1:200_000]

train['english'] = train['translation'].apply(lambda x: x.get('en', ''))
train['french'] = train['translation'].apply(lambda x: x.get('fr', ''))
train = train.drop('translation', axis=1)

def is_valid_sentence(sentence, vocabulary):
    return all(char in vocabulary for char in sentence)

train = train[(train['english'].str.len() <= 200) & (train['french'].str.len() <= 200)]
train = train[
    train['english'].apply(lambda x: is_valid_sentence(x, english_french_vocabulary)) &
    train['french'].apply(lambda x: is_valid_sentence(x, english_french_vocabulary))
]

class TextDataset(Dataset):
    def __init__(self, english_sentences, french_sentences):
        self.english_sentences = english_sentences
        self.french_sentences = french_sentences

    def __len__(self):
        return len(self.english_sentences)
    
    def __getitem__(self, idx):
        return self.english_sentences[idx], self.french_sentences[idx]

dataset = TextDataset( train['english'].tolist(), train['french'].tolist())

d_model = 512 # embedding dimension
max_length = 200 # maximum number of words for one translation
batch_size = 32 # number of "sentence" per batch
num_heads = 8 # number of heads during the self attention
drop_prob = 0.1 # probability of dropout for a better generalization
ffn_hidden = 2048 # expend 512 to 2048 during feed forward step
num_layers = 1 # number of sequential encoder
fr_vocab_size = len(english_french_vocabulary) # number of characters

train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)

for batch_num, batch in enumerate(iterator):
    print(batch)
    if batch_num > 3:
        break

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_length):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_length)
                          .reshape(self.max_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class SentenceEmbedding(nn.Module):
    def __init__(self, max_length, d_model, stoi, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(stoi)
        self.max_length = max_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.stoi = stoi
        self.position_encoder = PositionalEncoding(d_model, max_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token, end_token):
        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.stoi[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.stoi[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.stoi[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_length):
                sentence_word_indicies.append(self.stoi[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self, x, start_token, end_token): # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape # [d_model]
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs): #  batch_size * max_length * d_model
        dims = [-(i+1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean)**2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # 3 * d_model to simulate three independant matrix, we can consider these three matrices as concatenate together
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y,  mask=None):
        batch_size, sequence_length, d_model = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        # We create dimension for the heads to parallelize the process.
        # The last dimension contains the matrix q, k and v
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        # We move the head dimension to the second position and the sequence length dimension to the third place.
        # This allows us to parallelize the calculations of the dot products K and Q for each word and then for each head.
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        # We retrieve independent q, k and v matrices by chuking the qkv matrix on the last dimension
        k, v = kv.chunk(2, dim=-1)
        attention = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention = attention.permute(1, 0, 2, 3) + mask
            attention = attention.permute(1, 0, 2, 3)
        attention = F.softmax(attention, dim=-1)
        values = attention @ v
        # Concatenation of all the different head, strictly equivalent to (batch_size, sequence_length, d_model)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # 3 * d_model to simulate three independant matrix, we can consider these three matrices as concatenate together
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, _ = x.size()
        qkv = self.qkv_layer(x)
        # We create dimension for the heads to parallelize the process.
        # The last dimension contains the matrix q, k and v
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        # We move the head dimension to the second position and the sequence length dimension to the third place.
        # This allows us to parallelize the calculations of the dot products K and Q for each word and then for each head.
        qkv = qkv.permute(0, 2, 1, 3)
        # We retrieve independent q, k and v matrices by chuking the qkv matrix on the last dimension
        q, k, v = qkv.chunk(3, dim=-1)
        attention = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention = attention.permute(1, 0, 2, 3) + mask
            attention = attention.permute(1, 0, 2, 3)
        attention = F.softmax(attention, dim=-1)
        values = attention @ v
        # Concatenation of all the different head, strictly equivalent to (batch_size, sequence_length, d_model)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads*self.head_dim)
        out = self.linear_layer(values)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.ffn = FeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])


    def forward(self, x, self_attention_mask):
        residual_x = x
        x= self.attention(x, mask=self_attention_mask) # The encoder has to be able to look at any other word in the sentence
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_length, 
                 stoi, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_length, d_model, stoi, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*(EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)))

    def forward(self,x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.cross_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.ffn = FeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        residual_y = y
        y = self.attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.norm1(y + residual_y)
        residual_y = y
        y = self.cross_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.norm2(y)
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm3(y + residual_y)
        return y

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_length, 
                    stoi, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_length, d_model, stoi, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*(DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)))

    def forward(self, x, y , self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y

class Transformer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_length, fr_vocab_size, stoi, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        print(f"vocab_size = {fr_vocab_size} et stoi = {len(stoi)}")
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_length, stoi, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_length, stoi, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, fr_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x, y, encoder_self_attention_mask=None, decoder_self_attention_mask=None, decoder_cross_attention_mask=None, 
                enc_start_token=False, enc_end_token=False, dec_start_token=False, dec_end_token=False):
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out

NEG_INFTY = -1e9

def create_masks(eng_batch, fr_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_length, max_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_length, max_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_length, max_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_length, max_length] , False)

    for idx in range(num_sentences):
      eng_sentence_length, fr_sentence_length = len(eng_batch[idx]), len(fr_batch[idx])
      eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_length)
      fr_chars_to_padding_mask = np.arange(fr_sentence_length + 1, max_length)
      encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
      encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, fr_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, fr_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, fr_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

transformer = Transformer(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_length, fr_vocab_size, 
                          stoi, START_TOKEN, END_TOKEN, PADDING_TOKEN)

criterian = nn.CrossEntropyLoss(ignore_index=stoi[PADDING_TOKEN],
                                reduction='none')
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)
optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        eng_batch, fr_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, fr_batch)
        optim.zero_grad()
        fr_predictions = transformer(eng_batch,
                                     fr_batch,
                                     encoder_self_attention_mask.to(device), 
                                     decoder_self_attention_mask.to(device), 
                                     decoder_cross_attention_mask.to(device),
                                     enc_start_token=False,
                                     enc_end_token=False,
                                     dec_start_token=True,
                                     dec_end_token=True)
        labels = transformer.decoder.sentence_embedding.batch_tokenize(fr_batch, start_token=False, end_token=True)
        loss = criterian(
            fr_predictions.view(-1, fr_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == stoi[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        #train_losses.append(loss.item())
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"English: {eng_batch[0]}")
            print(f"French Translation: {fr_batch[0]}")
            fr_sentence_predicted = torch.argmax(fr_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in fr_sentence_predicted:
              if idx == stoi[END_TOKEN]:
                break
              predicted_sentence += itos[idx.item()]
            print(f"French Prediction: {predicted_sentence}")


            transformer.eval()
            fr_sentence = ("",)
            eng_sentence = ("should we go to the mall?",)
            for word_counter in range(max_length):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, fr_sentence)
                predictions = transformer(eng_sentence,
                                          fr_sentence,
                                          encoder_self_attention_mask.to(device), 
                                          decoder_self_attention_mask.to(device), 
                                          decoder_cross_attention_mask.to(device),
                                          enc_start_token=False,
                                          enc_end_token=False,
                                          dec_start_token=True,
                                          dec_end_token=False)
                next_token_prob_distribution = predictions[0][word_counter] # not actual probs
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = itos[next_token_index]
                fr_sentence = (fr_sentence[0] + next_token, )
                if next_token == END_TOKEN:
                  break
            
            print(f"Evaluation translation (should we go to the mall?) : {fr_sentence}")
            print("-------------------------------------------")