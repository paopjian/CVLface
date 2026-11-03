class IdentityAugmenterV2Numpy:
    """Keep the decoded numpy image unchanged for downstream view transforms."""

    def augment(self, sample):
        return sample
