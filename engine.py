import torch
from tqdm import tqdm


def mixup_criterion(criterion, pred, targets_a, targets_b, lam):
    return (lam * criterion(pred, targets_a) + (1.0 - lam) * criterion(pred, targets_b)).mean()


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=10):
    best_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(dataloaders[phase], leave=True)
            progress_bar.set_description(f"Epoch {epoch}/{num_epochs - 1} [{phase.capitalize()}]")

            for batch in progress_bar:
                if phase == 'train':
                    images, targets_a, targets_b, lam = batch
                    images = images.to(device)
                    targets_a = targets_a.to(device)
                    targets_b = targets_b.to(device)
                    lam = lam.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == targets_a).item()
                else:
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels).item()

                running_loss += loss.item() * images.size(0)
                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

            if phase == 'val':
                if scheduler is not None:
                    scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'Best Val Acc: {best_acc:.4f}')

    return model