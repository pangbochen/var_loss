# -*- coding: UTF-8 -*-
from model import LM

import numpy as np
import torch
from torch.autograd import Variable
from plot import make_dot

if __name__ == '__main__':
    x = Variable(torch.randn(32,42))

    a = LM(42)

    y = a(x)
    g = make_dot(y)
    #g.view()
    g.render('here', view=False)
