from torch.utils.data import DataLoader


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            el = next(self.iterator)
        except StopIteration:
            self.iterator = super().__iter__()
            el = next(self.iterator)
        return el
