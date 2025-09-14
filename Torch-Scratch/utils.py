import torch

### Assume no padding
def custom_unfold(inputs, k_size, stride, flatten=False):
    x = inputs.detach().clone()
    steps = ((x.shape[-1] - k_size) // stride) + 1
    # print(steps)
    batch_output = []
    for img in range(x.shape[0]):
        output = []
        for i in range(x.shape[1]):
            unfolded_temp = []
            for j in range(steps):
                for k in range(steps):
                    unfolded_temp.append(x[img, i,    j*stride:(j*stride)+k_size,   k*stride:(k*stride)+k_size])

            output.append(torch.stack(unfolded_temp, dim=0))
        batch_output.append(torch.stack(output, dim=0))
    batch_output = torch.stack(batch_output, dim=0)
    if flatten:
        return batch_output.reshape(batch_output.shape[0], batch_output.shape[1], batch_output.shape[2], -1)
    return batch_output