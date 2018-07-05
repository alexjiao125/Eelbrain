"""Univariate ANCOVA

cross-checked with exercise (page 8) from
http://www.bio.ic.ac.uk/research/crawley/statistics/exercises/R6Ancova.pdf

Crawley, M. J. (2005). Statistics: an introduction using R. Chichester:
      J. Wiley.

                	 SS     	df 	 MS    	    F      	 p
-------------------------------------------------------------
Hours            	 7.0583 	 1 	7.0583 	54.8981*** 	.0000
Genotype         	27.8750 	 5 	5.5750 	43.3611*** 	.0000
Genotype x Hours 	 3.1488 	 5 	0.6298 	 4.8981*   	.0113
Residuals        	 1.5429 	12 	0.1286
-------------------------------------------------------------
Total            	39.6250 	23

"""
import numpy as np
from eelbrain import *

y = Var([2, 3, 3, 4,
         3, 4, 5, 6,
         1, 2, 1, 2, 
         1, 1, 2, 2,
         2, 2, 2, 2,
         1, 1, 2, 3], name="Growth Rate")


genot = Factor(np.arange(6).repeat(4), name="Genotype")

hrs = Var([8, 12, 16, 24]*6, name="Hours")

# show the model
print(hrs * genot)

# print ANOVA table
print(test.anova(y, hrs*genot, title="ANOVA"))

# plot the slopes
plot.Regression(y, hrs, genot)

