import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def model_fit(epochs, model, training_set, validation_set, opt_func=torch.optim.Adam):
    history_training = []
    history_testing = []
    optimizer = opt_func(model.parameters())
    for epoch in range(epochs):
        # Training Phase
        for batch in training_set:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # monitor training phase
        result_train = evaluate(model,training_set)
        history_training.append(result_train)

        # Validation phase
        result_test = evaluate(model, validation_set)
        history_testing.append(result_test)

        model.epoch_end(epoch, result_train, result_test)
    return history_training,history_testing

def check_cuda():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    """Move tensor to cuda if have"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def plot_losses(history_training):
    losses = [x['val_loss'] for x in history_training]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss of testing set during training')
    plt.show()

def plot_accuracies(history_training,history_testing):
    accuracies_train = [x['val_acc'] for x in history_testing]
    accuracies_test = [x['val_acc'] for x in history_training]
    plt.plot(accuracies_train, label = 'accuracies_train')
    plt.plot(accuracies_test, label='accuracies_test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy of training and testing')
    plt.show()

def accuracy_in_each_class(model, test_dl):
    # 0 to 9: correct classify in dict_result
    # 10 to 19: incorrect classify in dict_result
    dict_result = {i: 0 for i in range(20)}
    for batch_data in test_dl:

        images, labels = batch_data
        output = model(images)
        _, preds = torch.max(output, dim=1)
        for target,predict in zip(labels,preds):
            if target.item() == predict.item(): dict_result[target.item()] +=1
            else:
                index = target.item() + 10
                dict_result[index] +=1

    per_class_acc = []
    per_class_num = []
    per_class_num_true = []
    for id in range(10):
        per_class_acc.append(dict_result[id] / (dict_result[id] + dict_result[id + 10]))
        per_class_num.append(dict_result[id] + dict_result[id + 10])
        per_class_num_true.append(dict_result[id])
    print("number data in each class")
    print(per_class_num)
    print("true data classify in each class")
    print(per_class_num_true)
    print("acc of each class")
    print(per_class_acc)

# Plot all image in the first batch size of training set
def plot_grid_images(data_dl):
    for images, _ in data_dl:
        print('images.shape:', images.shape)
        plt.figure(figsize=(16, 8))
        plt.axis('off')
        plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
        plt.show()
        break