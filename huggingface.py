from datasets import load_dataset

ds = load_dataset("csv", data_files="preprocessed_train_data.csv")
print(ds)

print(ds["train"].column_names)