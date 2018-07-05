from eelbrain import *
import numpy as np

y = np.empty(21)
y[:14] = np.random.normal(0, 1, 14)
y[14:] = np.random.normal(1.5, 1, 7)

y = Var(y, 'y')
a = Factor('abc', 'a', repeat=7)

print(Dataset((y, a)))
print(table.frequencies(a))
print(test.anova(y, a))
print(test.pairwise(y, a, corr='Hochberg'))

t = test.pairwise(y, a, corr='Hochberg')
print(t.get_tex())

plot.Boxplot(y, a, title="My Boxplot", ylabel="value", corr='Hochberg')
