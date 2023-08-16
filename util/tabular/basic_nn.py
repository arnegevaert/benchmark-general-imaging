from torch import nn
from tqdm import tqdm
import torch
from sklearn.metrics import r2_score


class BasicNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        pred_type="classification",
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.pred_type = pred_type

    def forward(self, x):
        result = self.layers(x)
        if self.pred_type == "classification":
            return nn.functional.log_softmax(result, dim=1)
        return result

    def train(self, train_dl, num_epochs):
        criterion = (
            nn.CrossEntropyLoss()
            if self.pred_type == "classification"
            else nn.MSELoss()
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        prog = tqdm(range(num_epochs))
        for epoch in range(num_epochs):
            epoch_loss = 0
            for x, y in train_dl:
                optimizer.zero_grad()
                y_hat = self(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            prog.set_postfix({"Avg train loss": epoch_loss / len(train_dl)})
            prog.update()
        prog.close()
        print()

    def test(self, test_dl):
        with torch.no_grad():
            if self.pred_type == "classification":
                # compute accuracy
                num_correct = 0
                num_total = 0
                for x, y in test_dl:
                    y_hat = self(x)
                    num_correct += (y_hat.argmax(dim=1) == y).sum().item()
                    num_total += len(y)
                return num_correct / num_total
            else:
                y_true = []
                y_pred = []
                for x, y in test_dl:
                    y_hat = self(x)
                    y_true.append(y)
                    y_pred.append(y_hat)
                return r2_score(torch.cat(y_true), torch.cat(y_pred))

