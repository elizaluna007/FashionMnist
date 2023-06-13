import csv
import matplotlib.pyplot as plt

epochs = []
loss = []
accuracy = []
precision = []
recall = []

with open('./test.csv', 'r') as file:
    csv_reader = csv.reader(file, delimiter=',')
    i=0
    for row in csv_reader:
        i=i+1
        epoch = int(row[0])
        epochs.append(epoch)
        loss_value = float(row[1])
        loss.append(loss_value)
        accuracy_value = float(row[2])
        accuracy.append(accuracy_value)
        precision_values = [float(value) for value in row[4:14]]
        precision.append(precision_values)
        recall_values = [float(value) for value in row[15:]]
        recall.append(recall_values)

# Plotting
plt.figure(figsize=(12, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss')

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')

# Plot Precision
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(10):
    plt.plot(epochs, [p[i] for p in precision], label=f'Class {i}')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('Precision')
plt.legend()

# Plot Recall
plt.subplot(1, 2, 2)
for i in range(10):
    plt.plot(epochs, [r[i] for r in recall], label=f'Class {i}')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.title('Recall')
plt.legend()

plt.tight_layout()
plt.show()
