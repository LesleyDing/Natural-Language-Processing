import math, collections
class CustomModel:

  def __init__(self, corpus):
    """Initial custom language model and structures needed by this mode"""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.trigramCounts = collections.defaultdict(lambda: 0)
    
    self.biPreCounts = collections.defaultdict(lambda: 0)
    self.triPreCounts = collections.defaultdict(lambda: 0)
    
    self.total = 0
    self.V = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
    """  
    # TODO your code here
    for sentence in corpus.corpus:
      for i in xrange(2, len(sentence.data)):
        self.unigramCounts[(sentence.data[i].word)] += 1
        self.bigramCounts[(sentence.data[i-1].word, sentence.data[i].word)] += 1
        self.biPreCounts[sentence.data[i-1].word] += 1
        self.trigramCounts[(sentence.data[i-2].word, sentence.data[i-1].word, sentence.data[i].word)] += 1
        self.triPreCounts[(sentence.data[i-2].word, sentence.data[i-1].word)] += 1
        self.total += 1 

  def score(self, sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """
    # TODO your code here
    score = 0.0
    for i in xrange(2, len(sentence)):
      if (sentence[i]) not in self.unigramCounts:
        self.V += 1

      countUni = self.unigramCounts[(sentence[i])]
      countBi = self.bigramCounts[(sentence[i-1],sentence[i])]
      countBiPre = self.biPreCounts[(sentence[i-1])]
      countTri = self.trigramCounts[(sentence[i-2], sentence[i-1],sentence[i])]
      countTriPre = self.triPreCounts[(sentence[i-2],sentence[i-1])]
      
      if countTri > 0 and countTriPre > 0:
        score += math.log(countTri)
        score -= math.log(countTriPre)
      elif countBi > 0 and countBiPre > 0:
        score += math.log(countBi)
        score -= math.log(countBiPre)
      else:
        score += math.log(countUni + 1)
        score -= math.log(self.total + self.V)
        score += math.log(0.9)

    return score
