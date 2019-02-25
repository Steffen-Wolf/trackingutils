from inferno.io.transform import Transform


class SliceTransform(Transform):

    def __init__(self, slice_index, slice_dim, **super_kwargs):
        super().__init__(**super_kwargs)
        self.slice_index = slice_index
        self.slice_dim = slice_dim

    def tensor_function(self, tensor):
        return tensor.take(self.slice_index,
                           self.slice_dim)
