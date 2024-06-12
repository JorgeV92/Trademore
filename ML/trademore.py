import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    sequence_length: int = None # length of the input sequences of stock data features
    num_features: int = None # the number of input features in your stock data
    hidden_size: int = 64  # hidden state size
    num_layers: int = 2  # number of RNN/GRU layers
    output_size: int = 1  # output size, for instance 1 for a single value prediction 
    block_size = None
    vocab_size = None
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4

# -----------------------------------------------------------------------------

"""
Recurrent Neural Net language model: either a vanilla RNN recurrence or a GRU.
Did not implement an LSTM because its API is a bit more annoying as it has
both a hidden state and a cell state, but it's very similar to GRU and in
practice works just as well.
"""

class RNNCell(nn.Module):
    """
    the job of a 'Cell' is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state
    h_{t} at the current timestep
    """
    def __init__(self, config):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        ht = F.tanh(self.xh_to_h(xh))
        return ht

class GRUCell(nn.Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.
    """
    def __init__(self, config):
        super().__init__()
        # input, forget, output, gate
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        # first use the reset gate to wipe some channels of the hidden state to zero
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev
        # calculate the candidate new hidden state hbar
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr))
        # calculate the switch gate that determines if each channel should be updated at all
        z = F.sigmoid(self.xh_to_z(xh))
        # blend the previous hidden state and the new candidate hidden state
        ht = (1 - z) * hprev + z * hbar
        return ht

class RNN(nn.Module):

    def __init__(self, config, cell_type):
        super().__init__()
        self.input_size = config.num_features  # input feature size
        self.hidden_size = config.hidden_size  # hidden state size
        self.num_layers = config.num_layers  # number of layers
        self.output_size = config.output_size  # output feature size

        if cell_type == 'rnn':
            self.cell = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, batch_first=True)
        elif cell_type == 'gru':
            self.cell = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  
        out, _ = self.cell(x, h0)  
        out = self.fc(out[:, -1, :])  # decode the hidden state of the last time step
        return out


# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

# -----------------------------------------------------------------------------
# helper functions for creating the training and test Datasets 

class StockDataset(Dataset):

    def __init__(self, stock_data, sequence_length):
        self.stock_data = stock_data
        self.sequence_length = sequence_length
        self.num_features = len(stock_data[0])

    def __len__(self):
        # Subtract sequence_length to avoid going off the end of the array
        return len(self.stock_data) - self.sequence_length

    def __getitem__(self, idx):
        # Get sequence_length worth of data starting from idx
        sequence_data = self.stock_data[idx:idx+self.sequence_length]
        # Get next day's adjusted close price
        next_day_close = self.stock_data[idx+self.sequence_length][3]  # 'Close' is at index 3
        return torch.FloatTensor(sequence_data), torch.tensor(next_day_close)
    
    def get_num_features(self):
        return self.num_features

def create_datasets(input_file, sequence_length):
    stock_data = pd.read_csv(input_file)
    stock_data = stock_data.fillna(0)
    features = stock_data.columns.drop('Date').tolist()
    stock_data = stock_data[features].values.tolist()

    test_set_size = min(1000, int(len(stock_data) * 0.1))
    train_size = len(stock_data) - test_set_size
    train_data = stock_data[:train_size]
    test_data = stock_data[train_size:]


    train_dataset = StockDataset(train_data, sequence_length)
    test_dataset = StockDataset(test_data, sequence_length)

    return train_dataset, test_dataset

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default='stock_data.csv', help="input file with things one per line")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=-1, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--device', type=str, default='cpu', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, default=3407, help="seed")
    # sampling
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels elsewhere in the model")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    parser.add_argument('--num-epochs', '-e', type=int, default=500, help="Number of epochs to train")
    args = parser.parse_args()
    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    sequence_length = 5
    train_dataset, test_dataset = create_datasets(args.input_file, sequence_length)
    # sequence_length = train_dataset.get_sequence_length()
    num_features = train_dataset.get_num_features()
    print(f"dataset determined that: {sequence_length=}, {num_features=}")
    
    # Display the first few items in the train_dataset
    # for i in range(5):
    #     sequence, next_day_close = train_dataset[i]
    #     print(f"Sequence {i+1}:")
    #     print(sequence)
    #     print(f"Next day's close price: {next_day_close}\n")

    # init model
    config = ModelConfig(sequence_length=sequence_length, num_features=num_features,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)
    if args.type == 'rnn':
        model = RNN(config, cell_type='rnn')
    elif args.type == 'gru':
        model = RNN(config, cell_type='gru')
    else:
        raise ValueError(f'model type {args.type} is not recognized')
    model.to(args.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))

    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

    # loss function
    criterion = nn.MSELoss()

    # Pytorch DataLoader:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Training loop
    for epoch in range(args.num_epochs):  # Assuming you have num_epochs argument to specify the number of epochs
        model.train()  # Set the model to training mode
        for i, (sequence, target) in enumerate(train_loader):
            # print(type(sequence), sequence)
            # print(type(target), target)
            sequence = sequence.to(args.device)
            target = target.to(args.device)
            
            # Forward pass
            output = model(sequence)
            loss = criterion(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {loss.item()}")

    # Testing loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_loss = 0
        for i, (sequence, target) in enumerate(test_loader):

            sequence = sequence.to(args.device)
            target = target.to(args.device)

            # Forward pass
            output = model(sequence)
            loss = criterion(output, target)
            total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss}")

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(args.work_dir, 'model.pt'))

