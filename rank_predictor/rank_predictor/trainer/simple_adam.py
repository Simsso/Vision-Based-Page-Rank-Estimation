from rank_predictor.model.screenshot_feature_extractor import ScreenshotFeatureExtractor
import torch.optim as optim


net = ScreenshotFeatureExtractor()

optimizer = optim.SGD(net.parameters(), lr=0.01)

for _ in range(100):
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(input)
    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()
