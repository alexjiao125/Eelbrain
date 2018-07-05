"""Univariate repeated measures ANOVA

Rutherford (2001) Examples, cross-checked results:

factorial anova

Independent Measures (p. 53):

             SS   df       MS   MS(denom)   df(denom)          F        p
-------------------------------------------------------------------------
A        432.00    1   432.00        9.05          42   47.75***   < .001
B        672.00    2   336.00        9.05          42   37.14***   < .001
A x B    224.00    2   112.00        9.05          42   12.38***   < .001
-------------------------------------------------------------------------
Total   1708.00   47


Repeated Measure (p. 86):

             SS   df       MS   MS(denom)   df(denom)          F        p
-------------------------------------------------------------------------
A        432.00    1   432.00       10.76           7   40.14***   < .001
B        672.00    2   336.00       11.50          14   29.22***   < .001
A x B    224.00    2   112.00        6.55          14   17.11***   < .001
-------------------------------------------------------------------------
Total   1708.00   47

"""
import numpy as np
from eelbrain import *


y = Var([ 7,  3,  6,  6,  5,  8,  6,  7,
          7, 11,  9, 11, 10, 10, 11, 11,
          8, 14, 10, 11, 12, 10, 11, 12,
         16,  7, 11,  9, 10, 11,  8,  8,
         16, 10, 13, 10, 10, 14, 11, 12,
         24, 29, 10, 22, 25, 28, 22, 24])

a = Factor([1, 0], repeat=3 * 8, name='A')
b = Factor(list(range(3)), tile=2, repeat=8, name='B')

# Independent Measures:
subject = Factor(list(range(8 * 6)), name='subject', random=True)
print((test.anova(y, a * b + subject(a % b), title="Independent Measures:")))

# Repeated Measure:
subject = Factor(list(range(8)), tile=6, name='subject', random=True)
print((test.anova(y, a * b * subject, title="Repeated Measure:")))
