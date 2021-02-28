# Dataset
Implement a loader for the datasets of the challenge.
Two data loaders are available: `Dataset` and `MergedDataset`. Each has the following attributes:
* **X**: Xtr*.csv file.
* **y**: Ytr*.csv file.
* **X_test**: Xte*.csv file.
* **X_mat**: Xtr*_mat100.csv file.
* **X_mat_test**: Xte*_mat100.csv file.

When initializing the `Dataset` class, one can specify one of the 3 datasets with the attribute `k`.

The `MergedDataset` class allows to load the full concatenated data and access it with the same attributed.

**Example**
```python
    from dataset import Dataset, MergedDataset

    ds = Dataset(k=0)
    print(ds.X)
    print(ds.y)
    print(ds.X_test)
    print(ds.X_mat)
    print(ds.X_mat_test)

    ds = MergedDataset()
    print(ds.X)
    print(ds.y)
    print(ds.X_test)
    print(ds.X_mat)
    print(ds.X_mat_test)
```
