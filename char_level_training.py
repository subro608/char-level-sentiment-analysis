import unicodedata
import string

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

with open("EnglishSentiWordNet_3.0.0_20130122.txt", "r") as f:
    lines = f.readlines()



im_lines = lines[26:]
with open("yourfile.txt", "w") as f:
    for line in im_lines:
      f.write(line)

with open("yourfile.txt", "r") as f:
    lines = f.readlines()


def unicodetoascii(s):
  return ''.join(
  c for c in unicodedata.normalize('NFD',s)
  if unicodedata.category(c) != 'Mn'
  and c in all_letters    
  )

def create_array(category):
  l = [0,0,0]
  arr = 1-(category[0]+category[1])
  
  if arr > category[0] and arr > category[1]:
    l = [0,0,1]
  elif category[0] > arr and category[0] > category[1]:
    l = [1,0,0]
  elif category[1] > arr and category[1] > category[0]:
    l = [0,1,0]
  elif category[0] == category[1] :
    l = [0,0,0]
  elif category[0] == arr and arr > category[1]:
    l = [1,0,0]
  elif category[1] == arr and arr > category[0]:
    l = [0,1,0]
  return l  
  
  

  

categories = {}

with open("yourfile.txt", "r") as f:
    lines = f.readlines()
    for f in lines[1:]:
      l = f.split()
      im_list = list(map(float,l[2:4] ))
      try:
        a = create_array(im_list)
        unicoded_text = unicodetoascii(l[4].rstrip('#1'))
        categories[unicoded_text] = a
     
      except:
        print("Exception has occured")
     
      


import torch

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, 3)


input = lineToTensor('able')
hidden = torch.zeros(1, n_hidden)

output123, next_hidden = rnn(input[0], hidden)
print(output)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    l = [0,0,0]
    l[category_i] = l[category_i]+1
    return l

print(categoryFromOutput(output123))

criterion = nn.MSELoss()
learning_rate = 0.005 
def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    output_tensor = torch.tensor(categoryFromOutput(output), dtype=torch.float)    

    loss = criterion(output, category_tensor.view(1,3))
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

lines = list(categories.keys())


import time
import math

n_iters = 10000
print_every = 10
plot_every = 1



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
  
  category = categories[lines[iter]]
  category_tensor = torch.tensor(category, dtype=torch.float)
  line_tensor = lineToTensor(line)
  output, loss = train(category_tensor, line_tensor)
  current_loss += loss

  # Print iter number, loss, name and guess
  if iter % print_every == 0:
    
    guess_i = categoryFromOutput(output)
    correct = '✓' if guess_i == category else '✗ (%s)' % category
    print('%d %d%% (%s) %.4f  %s' % (iter, iter / n_iters * 100, timeSince(start), loss, correct))

    # Add current loss avg to list of losses
  if iter % plot_every == 0:
    all_losses.append(current_loss / plot_every)
    current_loss = 0
  
    
    
    
