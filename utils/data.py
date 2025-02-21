def debug_getitem(func):
    """
    Decorator to optionally visualize the image for debugging
    (only on the first call if self.debug == True).
    """
    def wrapper(self, idx):
        data = func(self, idx)
        if self.debug and not self._debug_shown:
            image = data["image"]  # shape [C,H,W], torch tensor
            self._debug_show_image(image)
            self._debug_shown = True
        return data
    return wrapper