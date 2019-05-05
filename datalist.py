import glob

testlist = glob.glob('./data/test*')
flist = open('test.txt', 'w')
for f in testlist:
    flist.write(f)
    flist.write('\n')
flist.close()
