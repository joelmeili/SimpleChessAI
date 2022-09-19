import torch
import chess
import random
import numpy
from create_data import random_board, stockfish, split_dims
from model import NeuralNet

model = NeuralNet()
model.load_state_dict(torch.load("best_model.h5"))
model.eval()
