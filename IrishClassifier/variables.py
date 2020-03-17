# model variables
input_size = 4
hidden1 = 64
hidden2 = 16
output = 3
batch_size = 8
learning_rate = 1e-3
num_epochs = 30

# data variables
label_encode = {"setosa": 0, "versicolor": 1, "virginica": 2}
cutoff = 0.8

# data paths
csv_path = "iris.csv"
file_name = "preprocessed_iris.csv"
model_path = "iris_classifier.pt"
