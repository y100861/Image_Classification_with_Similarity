n_epochs = 100
n_classes = train_data.n_label

# model instance
model = testModel(n_classes)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.03, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
criterion = nn.CrossEntropyLoss()


# Train
def train(model, loader, epoch, optimizer, criterion, losses, accs, scheduler=None):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for i, (input, label) in enumerate(loader):
        input, label = input.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        train_loss += loss.item()
        acc = (output.argmax(dim=1) == label).float().sum().detach().cpu().item()
        train_acc += acc

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
    
        # print(f"Epoch: {epoch+1} | Step: {i+1}/{len(loader)} | Train Loss: {loss.item():.4f}")
    train_loss = train_loss / len(loader)
    train_acc = train_acc / len(loader.dataset)
    losses.append(train_loss)
    accs.append(train_acc)
    print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")


# Validation
def val(model, loader, epoch, criterion, losses, accs):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for i, (input, label) in enumerate(loader):
            input, label = input.to(device), label.to(device)
            output = model(input)
            loss = criterion(output, label)
            val_loss += loss.item()
            acc = (output.argmax(dim=1) == label).float().sum().cpu().item()
            val_acc += acc
    
    val_loss = val_loss / len(loader)
    val_acc = val_acc / len(loader.dataset)
    losses.append(val_loss)
    accs.append(val_acc)
    print(f"Epoch: {epoch+1} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")


# Start
train_losses, train_accs = [], []
val_losses, val_accs = [], []
max_val_acc = 0.0
for epoch in range(n_epochs):
    train(model, trainloader, epoch, optimizer, criterion, train_losses, train_accs)
    val(model, valloader, epoch, criterion, val_losses, val_accs)

    # Save best model
    if val_accs[-1] > max_val_acc:
        max_val_acc = val_accs[-1]
        torch.save(model.state_dict(), "best_simplecnn_model.pth")
