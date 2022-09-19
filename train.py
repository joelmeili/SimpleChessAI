import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model import NeuralNet

model = NeuralNet()

data = numpy.load("data.npz")
positions, scores = [data[key] for key in data]

scores = numpy.asarray(scores / abs(scores).max() / 2 + 0.5, dtype = numpy.float32)
tensor_x = torch.Tensor(positions)
tensor_y = torch.Tensor(scores)

data = TensorDataset(tensor_x, tensor_y)

train_set, val_set = random_split(data, lengths = [int(len(scores) * 0.7), len(scores) - int(len(scores) * 0.7)], 
    generator = torch.Generator().manual_seed(2022))

train_loader = DataLoader(train_set, batch_size = 256, shuffle = True, num_workers = 2)
val_loader = DataLoader(val_set, batch_size = 256, shuffle = False, num_workers = 2)

loss_fn = nn.MSELoss()
optimizer = Adam(model.parameters(), lr = 1e-3)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs.squeeze(), labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print("batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.

    return last_loss

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
epoch_number = 0

EPOCHS = 50

best_vloss = 1_000_000

for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(val_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs.squeeze(), vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    writer.add_scalars("Training vs. Validation Loss",
                    { "Training' : avg_loss, 'Validation" : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = "model_{}_{}".format(timestamp, epoch_number)
        torch.save(model.state_dict(), "best_model.h5")

    epoch_number += 1