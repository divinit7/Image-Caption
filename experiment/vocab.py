import os
import pickle
from collections import Counter
import nltk
from PIL import Image
from pycocotools.coco import COCO

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx +=1
            
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)
    
    
def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
    
    # omit non-frequent words
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab



def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
    
    # omit non-frequent words
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(caption_path, vocab_path, threshold):
    vocab = build_vocab(json = caption_path, threshold = threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
        

caption_path = './input/annotations/captions_train2014.json'    
vocab_path = './input/vocab.pkl'                 
threshold = 5

# main(caption_path ,vocab_path,threshold)