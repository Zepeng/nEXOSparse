import glob, pickle

testlist = glob.glob('./data/test_*')
datalist = []
for f in testlist:
    unpickle = pickle.load(open(f, 'rb'))
    if len(datalist) > 5000:
        break
    for index, item in enumerate(unpickle):
        datalist.append(item)

pickle.dump(datalist, open ("./data/test.p" , "wb"))
