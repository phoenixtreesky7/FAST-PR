
import os
import numpy as np


def path2txt(dataroot, skip):
    listpath = '%s%s%s' % (dataroot, '/cropped/test', '/test_id.txt')

    f=open(listpath,'w')

    filenames=os.listdir('%s%s' % (dataroot, '/cropped/test'))
    
    for filename in filenames:
        
        if filename[-4:] != '.txt':
            returnpath = filename
            print('Listing the file: ', returnpath)
            subfilenames=os.listdir('%s%s%s' % (dataroot, '/cropped/test/', filename))
            #print('lyx', subfilenames)
            for subfilename in subfilenames:
                #print(subfilename)
                if os.path.splitext(subfilename)[1] == '.png':
                    out_path=subfilename
                    
                    f.write(filename +'/'+ out_path + ' 0'+ '\n')
                    #if filename == 'no_pulsar':
                    #    f.write( 'test/' + filename +'/'+ out_path + ' 0'+ '\n')
                    #elif filename == 'pulsar':
                    #    f.write( 'test/' + filename +'/'+ out_path + ' 1'+ '\n')
    f.close()
    return dataroot+'/cropped' + '/test'