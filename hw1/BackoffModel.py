import math, collections

class BackoffModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # set up dictionaries
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.preCounts = collections.defaultdict(lambda: 0)
    # Global var
    self.total = 0
    self.V = 0
    
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    for sentence in corpus.corpus:
      for i in xrange(1, len(sentence.data)):
        # tokenUni = (sentence.data[i].word)
        self.unigramCounts[(sentence.data[i].word)] = self.unigramCounts[(sentence.data[i].word)] + 1
        # tokenBi = (sentence.data[i-1].word, sentence.data[i].word)
        self.bigramCounts[(sentence.data[i-1].word, sentence.data[i].word)] = self.bigramCounts[(sentence.data[i-1].word, sentence.data[i].word)] + 1
        self.preCounts[sentence.data[i-1].word] = self.preCounts[sentence.data[i-1].word] + 1
        self.total += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    for i in xrange(1, len(sentence)):
      if (sentence[i]) not in self.unigramCounts:
        self.V += 1
      countUni = self.unigramCounts[(sentence[i])]
      countBi = self.bigramCounts[(sentence[i-1],sentence[i])]
      countPre = self.preCounts[(sentence[i-1])]

      if countBi > 0 and countPre > 0:
        score += math.log(countBi)
        score -= math.log(countPre)
      else:
        score += math.log(countUni + 1)
        score -= math.log(self.total + self.V)
        score += math.log(0.6)
    return score
