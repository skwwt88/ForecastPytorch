import torch

from tqdm import tqdm
from models import GRUModel
from datasets import train_data
from config import F, T, LATENT_DIM
from torch.optim import lr_scheduler


device = torch.device("cuda:0")
model = GRUModel(1, F, LATENT_DIM, 1).to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True, cooldown=1, min_lr=0.1e-8, eps=1e-05)

best_score = 0
X_train, X_test, y_train, y_test = train_data()
X_train = torch.tensor(X_train).float().cuda()
X_test = torch.tensor(X_test).float().cuda()
y_train = torch.tensor(y_train).float().cuda()
y_test = torch.tensor(y_test).float().cuda()

pbar = tqdm(range(0, 20000))
for epoch in pbar:
    optimizer.zero_grad()

    # forward + backward + optimize
    train_outputs = model(X_train)
    train_loss = criterion(train_outputs, y_train)
    train_loss.backward()
    optimizer.step()

    with torch.no_grad():
      outputs = model(X_test)
      validate_loss = criterion(outputs, y_test)

    pbar.set_description("{0:.6f}, {1:.6f}".format(train_loss, validate_loss))
    scheduler.step(validate_loss)
    