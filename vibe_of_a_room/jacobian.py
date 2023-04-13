import sys
print(sys.version)
print(sys.path)
import numpy as np
from matplotlib import pyplot as plt

# Arbitrary vector field example.
shape = 11, 11
dims = len(shape)
domain = -2, 2
x, y = np.meshgrid(*(np.linspace(*domain, num=d) for d in shape))
field = np.stack((-2 * x * y, x ** 2 + 2 * y - 4))
print(f'{field.shape=}')

# Compute jacobian using NumPy's gradient function.
partials = tuple(np.gradient(i) for i in field)
print(f'{len(partials)=}')
jacobian = np.stack(partials).reshape(*(j := (dims,) * 2), *shape)

# Extract divergence and curl from jacobian.
divergence = np.trace(jacobian)
curl_mask = np.triu(np.ones(j, dtype=bool), k=1)

# Only valid for 2D! Higher dimensions have more complex 
# levi-civita symbols so the signs and numbers of components will be different.
curl = (jacobian[curl_mask] - jacobian[curl_mask.T]).squeeze()

# Generate affine map by augmenting jacobian.
affine = np.pad(jacobian, ((0, 1), (0, 1), *((0, 0),) * dims))
affine[:dims, dims] = field
affine[dims, dims] = 1
affine = np.moveaxis(affine, range(-dims, 0), range(dims))

# Visualize curl.
plt.matshow(curl.T, cmap='RdYlBu')
plt.quiver(*field, angles='xy')
plt.show()

# Get the affine transformation at a point.
print(affine[0, 0])
