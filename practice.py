import torch, math
w = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
for step in range(50):
    loss = (w - 3) ** 2
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
    w.grad.zero_()

w.item()
