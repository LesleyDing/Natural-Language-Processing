import math, collections

class SmoothBigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # set up dictionaries
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.preCounts = collections.defaultdict(lambda: 0)
    
    # Global var
    self.V = 0
    
    self.train(corpus)
    

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    # Tip: To get words from the corpus, try
    for sentence in corpus.corpus:
      for i in xrange(0, len(sentence.data)-1):
        token = (sentence.data[i].word, sentence.data[i + 1].word)
        self.bigramCounts[token] = self.bigramCounts[token] + 1
        self.preCounts[(sentence.data[i].word)] = self.preCounts[(sentence.data[i].word)] + 1
    

  def score(self, sentence):
    # TODO your code here
    score = 0.0
    for i in xrange(0, len(sentence)-1):
      if (sentence[i], sentence[i+1]) not in self.bigramCounts:
        self.V += 1
      token = (sentence[i], sentence[i + 1])
      count = self.bigramCounts[token]
      countPre = self.preCounts[(sentence[i])]
      score += math.log(count + 1)
      score -= math.log(countPre + self.V)
      #Ignore unseen words
    return score
