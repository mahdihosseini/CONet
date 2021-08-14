"""
MIT License

Copyright (c) 2021 Mahdi S. Hosseini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
# import collections

import numpy as np


class EarlyStop:
    def __init__(self, patience: int = 10, threshold: float = 1e-2) -> None:
        # self.queue = collections.deque([0] * patience, maxlen=patience)
        self.patience = patience
        self.threshold = threshold
        self.wait = 0
        self.best_loss = np.Inf

    def __call__(self, train_loss: float) -> bool:
        """
        @monitor: value to monitor for early stopping
                  (e.g. train_loss, test_loss, ...)
        @mode: specify whether you want to maximize or minimize
               relative to @monitor
        """
        if np.less(self.threshold, 0):
            return False
        if train_loss is None:
            return False
        # self.queue.append(train_loss)
        if np.less(train_loss - self.best_loss, -self.threshold):
            self.best_loss = train_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False
