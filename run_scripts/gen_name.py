import os


def main():
    filepath = '/home/zhaoyf/star/train_id.txt'
    datapath_yes = '/home/zhaoyf/star/train/pulsar/'
    datapath_no = '/home/zhaoyf/star/train/no_pulsar/'
    print("Hello World")
    filelist_yes = os.listdir(datapath_yes)
    filelist_no = os.listdir(datapath_no)
    with open(filepath,'w') as f:
        for file1 in filelist_yes:
            strings = 'train/pulsar/'+file1+' '+'1'+'\n'
            f.write(strings)
        for file2 in filelist_no:
            strings = 'train/no_pulsar/'+file2+' '+'0'+'\n'
            f.write(strings)

def main_test():
    filepath = '/home/zhaoyf/star/test_id.txt'
    datapath_yes = '/home/zhaoyf/star/test/pulsar/'
    datapath_no = '/home/zhaoyf/star/test/no_pulsar/'
    print("Hello World test")
    filelist_yes = os.listdir(datapath_yes)
    filelist_no = os.listdir(datapath_no)
    with open(filepath,'w') as f:
        for file1 in filelist_yes:
            strings = 'test/pulsar/'+file1+' '+'1'+'\n'
            f.write(strings)
        for file2 in filelist_no:
            strings = 'test/no_pulsar/'+file2+' '+'0'+'\n'
            f.write(strings)
if __name__=="__main__":
    main()
    main_test()