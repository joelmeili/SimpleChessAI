from create_data import random_board, split_dims, stockfish
import numpy
import random

random.seed(2022)

positions = []
scores = []

iter = 10000
count = 0

while (count < iter):
    board = random_board()
    position = split_dims(board)
    score = stockfish(board, depth = 5)

    if score is not None:
        positions.append(position)
        scores.append(score)
        count += 1

scores = numpy.asarray(scores, dtype=numpy.int32)

numpy.savez("data.npz", positions, scores)