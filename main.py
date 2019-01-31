from surface_loader import surface_loader
import numpy as np

# Kauhan branch

test_size = 0.2
n_splits = 10

data,le = surface_loader(test_size,n_splits)

print("keys:",len(data))

for i in data:
    print("{} shape: {}".format(i,data[i].shape))

    for j in data[i]:
        if np.array_equal(data[i].shape,np.array([n_splits,])):
            print(j.shape)

print(data['y_train'])
print(le.inverse_transform(np.unique(data['y_train'])))