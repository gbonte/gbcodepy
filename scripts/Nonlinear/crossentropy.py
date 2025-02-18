## # "Statistical foundations of machine learning" software
# crossentropy.py
# Author: G. Bontempi
# Computation of cross_entropy losses by using torch

import torch


F=torch.nn.functional

K=5 ## number classes
N=3 ## number observations

logits = torch.randn(N, K, requires_grad=True)
## output of a classification model returning a quantitative score associated to conditional probability
## matrix [N,K]

targets = torch.randint(K, (N,), dtype=torch.int64)
## target classes: for each observation there is a class out of K
## vector of size N


print('logits=', logits, '\n classes=', targets)

loss = F.cross_entropy(logits, targets)

perplexity=torch.exp(loss)
## Perplexity is often considered more interpretable than the raw loss value because 
#it sis comparable to the number of classes (K)

print('loss=',loss, ' perplexity=', perplexity)



# Example of target with class probabilities

ptargets = torch.randn(N, K).softmax(dim=1)
## tensor of conditional probabilities
## matrix [N,K]
## Note that the sum over matrix rows is equal to 1


loss = F.cross_entropy(logits, ptargets)
perplexity=torch.exp(loss)
print('logits=', logits, '\n prob classes=', ptargets)
print('loss=',loss, ' perplexity=', perplexity)
