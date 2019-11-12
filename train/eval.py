from torch import nn
import torch

def eval_model(model, eval_data, device, epoch, filename):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    data_num = 0
    for inputs, labels in eval_data:
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            data_num += inputs.size(0)
    epoch_loss = running_loss / len(eval_data.dataset)
    epoch_acc = running_corrects / len(eval_data.dataset)
    log = 'Eval: Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc)
    print(log)
    with open(filename, 'a') as f: 
        f.write(log + '\n')
    return epoch_acc