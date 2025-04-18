import torch
import torch.nn as nn

'''
参照したページはPytorchの公式doc
https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
'''

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0,
                 batch_first=True, biderctional=False, device='cpu', dtype=torch.float32):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidrectional = biderctional
        self.device = device
        self.dtype = dtype
        
        # LSTM本体
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=self.batch_first,
            bidirectional=self.bidrectional
        ).to(device=self.device, dtype=self.dtype)
        
        # 線形結合
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size).to(device=self.device, dtype=self.dtype)
        # 重みの初期化
        self._init_weights()
        
    def _init_weights(self):
        # LSTMの重みを初期化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # 入力から隠れ層への重み
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:  # 隠れ層から隠れ層への重み
                nn.init.orthogonal_(param)
            elif 'bias' in name:  # バイアス
                nn.init.zeros_(param)
            
            # 全結合層の重みを初期化
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)       
        
    def forward(self, x):
        # バッチサイズを取得
        batch_size = x.size(0)
        
        # 初期隠れ状態とセル状態をゼロで初期化
        h_0 = torch.zeros(self.num_layers * (2 if self.bidrectional else 1), batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        c_0 = torch.zeros(self.num_layers * (2 if self.bidrectional else 1), batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        
        
        # LSTMに入力を通す
        lstm_out, (h_n, c_n) = self.lstm(x,(h_0,c_0))
        
        # LSTMの最終出力を取得
        if self.batch_first:
            # バッチサイズが最初の場合、最後のタイムステップを取得
            lstm_out = lstm_out[:, -1, :]
        else:
            # バッチサイズが2番目の場合、最後のタイムステップを取得
            lstm_out = lstm_out[-1, :, :]
        
        # 全結合層に通す
        output = self.fc(lstm_out)
        return output

# テスト
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    # ----- 1. データの準備 -----
    # 振幅2のcos関数 (ノイズ付)
    np.random.seed(42)
    x = np.linspace(0, 20, 100)
    noise = np.random.normal(0, 0.5, size=x.shape)
    y = 2 * np.cos(x) + noise

    # 入力とターゲットを作成 (sequence_length=10)
    sequence_length = 10
    X = []
    Y = []

    for i in range(len(y) - sequence_length):
        X.append(y[i:i+sequence_length])
        Y.append(y[i+sequence_length])

    X = np.array(X)
    Y = np.array(Y)

    # PyTorchテンソルへ変換
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (batch, seq_len, input_size=1)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)
    
    # ----- 2. LSTMモデルの定義(単層) -----
    model = MyLSTM(
        input_size=1,
        hidden_size=32,
        output_size=1,
        num_layers=1,
        batch_first=True,
    )
    criterion = nn.MSELoss() # 損失関数はMSELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # ----- 3. 学習 -----
    epochs = 100
    for epoch in range(epochs):
        model.train()
        output = model(X)
        loss = criterion(output, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # ----- 4. 予測と可視化 -----
    model.eval()
    predicted = model(X).detach().numpy()
    true = Y.numpy()

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot()
    ax.plot(true, label='True')
    ax.plot(predicted, label='Predicted')
    ax.set_title("LSTM Prediction of 2*cos(x) + Noise")
    ax.legend()
    plt.tight_layout()
    fig.savefig('./Predicted.png')
    plt.show()
    plt.close()
