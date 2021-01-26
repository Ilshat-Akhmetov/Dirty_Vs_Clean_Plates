import torch
from os import path
from torch.utils.tensorboard import SummaryWriter


def train_algorithm(model, loss, optimizer, scheduler, num_epochs, train_dataloader,val_dataloader,device):
    writer = SummaryWriter("TbGraph")
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            else:
                dataloader = val_dataloader
                model.eval()   # Set model to evaluate mode

            running_loss = 0.
            running_acc = 0.

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()

                #del inputs # clear memory from unnecesarry objects

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)
            if phase == 'train':
                scheduler.step()
                writer.add_scalar("Loss train: ", epoch_loss, epoch)
                writer.add_scalar("Accuracy train: ", epoch_acc, epoch)
            else:
                writer.add_scalar("Loss val: ", epoch_loss, epoch)
                writer.add_scalar("Accuracy val: ", epoch_acc, epoch)
                PATH = path.join("Weights",("model_{}th_epoch.pth").format(epoch))
                torch.save(model.state_dict(), PATH)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)
    writer.close()