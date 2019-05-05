import glob, os, sys, ntpath,subprocess
import time

def createJobScript(img_type, batch_num):
    datadir = '/gpfs/loomis/project/fas/david_moore/zl423/dlphysics/SparseConvNet/examples/nEXO/'
    script = open('%s/job/%s_%d.sh' % ( datadir,img_type, batch_num) ,'w')
    script.write('#!/bin/bash \n')
    script.write('#SBATCH -t 24:00:00 \n')
    script.write('date\n')
    script.write('source /gpfs/loomis/project/fas/david_moore/zl423/dlphysics/SparseConvNet/setup.sh\n')
    script.write('cd /gpfs/loomis/project/fas/david_moore/zl423/dlphysics/SparseConvNet/examples/nEXO\n')
    script.write('python prepare_data.py --bn=%d --img_type=%s\n' % (batch_num, img_type))
    script.write('date\n')
    jobout = '%s/job/%s_%d.out' % (datadir, img_type, batch_num)
    joberr = '%s/job/%s_%d.err' % (datadir, img_type, batch_num)
    script.close()
    os.system('sbatch  -o %s -e %s %s/job/%s_%d.sh' % (jobout, joberr, datadir, img_type, batch_num))

if __name__ == '__main__':
    for i in range(132):
        createJobScript('train', i)
    for i in range(33):
        createJobScript('test', i)
