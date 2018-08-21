import re
import numpy as np

alldata = []
with open('featureinfo.txt', 'r') as input_file:
    for line in input_file:
        row = re.search('\(([^)]+)', line).group(1)
        alldata.append(row)
#print "alldata"
#print alldata

alldata_numpy = np.asarray(alldata)
print alldata_numpy

print "int indices"
print np.where(alldata_numpy == 'int')
print "bool indices"
print np.where(alldata_numpy == 'bool')
