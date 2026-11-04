"""
合成データで小型CNNを学習し、ONNXに書き出す。
入力: 1x40x28 グレースケール (0-1), 出力: 20クラス (0-9=静止, 10+d=d→d+1の回転中)
"""
import glob, random, os
import numpy as np
import torch, torch.nn as nn
from PIL import Image

DATA = "data/synth"
W, H = 28, 40
EPOCHS = int(os.environ.get("EPOCHS", 12))

class DigitNet(nn.Module):
    def __init__(self, n=20):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64))   # 40x28 -> 5x3
        self.c = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 5 * 3, 128), nn.ReLU(), nn.Linear(128, n))
    def forward(self, x):
        return self.c(self.f(x))

def load():
    xs, ys = [], []
    for label in range(20):
        for p in glob.glob(f"{DATA}/{label}/*.png"):
            xs.append(np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0)
            ys.append(label)
    x = torch.tensor(np.stack(xs))[:, None]
    y = torch.tensor(ys)
    idx = torch.randperm(len(y))
    x, y = x[idx], y[idx]
    n = int(len(y) * 0.9)
    return (x[:n], y[:n]), (x[n:], y[n:])

def main():
    (xtr, ytr), (xva, yva) = load()
    net = DigitNet()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    lossf = nn.CrossEntropyLoss()
    bs = 128
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), bs):
            b = perm[i:i + bs]
            xb = xtr[b]
            if random.random() < 0.5:   # 明暗反転は学習時にも
                xb = 1 - xb
            opt.zero_grad()
            loss = lossf(net(xb), ytr[b])
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            acc = (net(xva).argmax(1) == yva).float().mean().item()
        print(f"epoch {ep+1} loss {loss.item():.3f} val_acc {acc:.4f}")
    torch.save(net.state_dict(), "digit_cnn.pt")
    torch.onnx.export(net, torch.zeros(1, 1, H, W), "digit_cnn.onnx",
                      input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, dynamo=False)
    print("saved digit_cnn.onnx", os.path.getsize("digit_cnn.onnx") // 1024, "KB")

if __name__ == "__main__":
    main()
