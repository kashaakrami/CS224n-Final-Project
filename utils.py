import pandas as pd
import re
import numpy as np
import nltk
import torch
from torch import nn
from torch.nn import functional as F
from nltk.corpus import stopwords
from collections import Counter
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn import metrics
from sklearn import preprocessing
nltk.download('stopwords')
import numpy as np
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.special import xlogy


class Model(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(1000, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = self.dropout(x)
        return x


class LSTM(nn.Module):

    def __init__(self, vocab_len, dimension=256):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_len, 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.0)

        self.fc = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)

        return text_fea


def preprocess_text(text):
    text = BeautifulSoup(text)
    text = text.get_text()
    text = re.sub(r'\@.+? ', '', text)
    text = text.replace('“', '')
    text = text.replace('”', '')
    text = text.lower()
    text = re.sub(r"[^a-zA-Z']", ' ', text)
    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in text.split() if w not in stops]
    text = ' '.join(meaningful_words)
    return text


def get_dataset_df(tensor=True):
    df = pd.read_csv('IMDB_Dataset.csv')
    df['text_clean'] = df['review'].apply(preprocess_text)
    vectorizer = CountVectorizer(max_features=1000)
    bag = vectorizer.fit_transform(df['text_clean'])
    X = bag.toarray()
    y = np.array(df['sentiment'] == 'positive').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if tensor == False:
        return X_train_scaled, X_test_scaled, y_train, y_test
    return torch.Tensor(X_train_scaled), torch.Tensor(X_test_scaled), y_train, y_test


def calculate_train_accuracy(X_train_scaled, y_train, net):
    # Train accuracy
    with torch.no_grad():
        X_train_sample = X_train_scaled[0:5000]
        outputs = net(X_train_sample)
        outputs_sigmoid = torch.sigmoid(outputs)
        outputs_binary = outputs_sigmoid.cpu().numpy().flatten() > 0.5
    accuracy = metrics.accuracy_score(y_train[0:5000], outputs_binary)
    return accuracy


def calculate_dev_accuracy(X_test_scaled, y_test, net, lstm=False):
    # Test accuracy
    with torch.no_grad():
        outputs = net(X_test_scaled)
        outputs_sigmoid = torch.sigmoid(outputs)
        outputs_binary = outputs_sigmoid.cpu().numpy().flatten() > 0.5
    accuracy = metrics.accuracy_score(y_test, outputs_binary)
    return accuracy


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def tokenize(x_train,y_train,x_val,y_val):
    word_list = []

    stop_words = set(stopwords.words('english')) 
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)
  
    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
    
    # tokenize
    final_list_train,final_list_test = [],[]
    for sent in x_train:
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                     if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                    if preprocess_string(word) in onehot_dict.keys()])
            
    encoded_train = [1 if label =='positive' else 0 for label in y_train]  
    encoded_test = [1 if label =='positive' else 0 for label in y_val] 
    return np.array(final_list_train), np.array(encoded_train),np.array(final_list_test), np.array(encoded_test),onehot_dict


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def compute_score(predictions):
        """
        Compute the score according to the heuristic.

        Args:
            predictions (ndarray): Array of predictions

        Returns:
            Array of scores.
        """
        assert predictions.ndim >= 3
        # [n_sample, n_class, ..., n_iterations]

        expected_entropy = -np.mean(
            np.sum(xlogy(predictions, predictions), axis=1), axis=-1
        )  # [batch size, ...]
        expected_p = np.mean(predictions, axis=-1)  # [batch_size, n_classes, ...]
        entropy_expected_p = -np.sum(xlogy(expected_p, expected_p), axis=1)  # [batch size, ...]
        bald_acq = entropy_expected_p - expected_entropy
        return bald_acq


class SentimentRNN(nn.Module):
    def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5):
        super(SentimentRNN,self).__init__()
 
        self.output_dim = 1
        self.hidden_dim = hidden_dim
 
        self.no_layers = no_layers
        self.vocab_size = vocab_size
    
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True)
        
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
    
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()
        
    def forward(self,x):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        lstm_out, _ = self.lstm(embeds)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out


# function to predict accuracy
def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()