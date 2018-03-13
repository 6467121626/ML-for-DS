import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

f  = open('movies.txt', 'r')
movie = f.readlines()

for items in movie:
    if "God" in items:
        print items