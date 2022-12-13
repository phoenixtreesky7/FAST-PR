from .bases import BaseImageDataset
import os.path as osp
import glob
import re
import os


class UnifiedLoader(BaseImageDataset):
    def __init__(self, dataset_name, data_dir='data_dir', istrain=True, verbose=True):
        super(UnifiedLoader, self).__init__()

        #self.root = 'D:/dzhao/CODE/RFI_CLASS/DensNet/Experiments/DenseNet/data/fast_pulsar_data_d/'
        self.root = data_dir
        self.istrain = istrain
        print('root', self.root)

        ## parse dataset names support aircraft_train/car_train/dog_train
        if dataset_name == 'fast':
            #data_dir = '/data_zfs/fast/zhaodong/pulsar_class_resnet50/datadir/'
            if self.istrain == True:
                print('reading the samples ...' )
                train_txt_file = open(os.path.join(self.root, 'train_id.txt'))
            #else:
            #    train_txt_file = open(os.path.join(self.root, 'test_id.txt'))
            test_txt_file = open(os.path.join(self.root, 'test_id.txt'))
        self.dataset_dir = data_dir

        data_len = None

        test_img_list = []

        train_img_list = []

        train_label_list = []
        test_label_list = []

        if self.istrain == True:
            for line in train_txt_file:
                #print(line)
                train_img_list.append(line[:-1].split(' ')[0])
                #print(line[:-1].split(' ')[0])
                #print(line[:-1].split(' ')[-1])
                train_label_list.append(int(line[:-1].split(' ')[-1]))

        for line in test_txt_file:
            test_img_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[-1]))


        dataset_train = []
        dataset_test = []


        # retaining the dataset format
        train_cluster = train_label_list
        test_cluster = test_label_list


        if self.istrain == True:
            for idx in range(len(train_label_list)):
                #imgpath = os.path.join(data_dir,'train', train_img_list[idx])
                imgpath = os.path.join(data_dir, train_img_list[idx])
                dataset_train.append((imgpath, train_label_list[idx]))


        for idx in range(len(test_label_list)):
            #imgpath = os.path.join(data_dir,'test', test_img_list[idx])
            imgpath = os.path.join(data_dir, test_img_list[idx])
            dataset_test.append((imgpath, test_label_list[idx]))

        self.train = dataset_train
        self.test = dataset_test

        if self.istrain == True:
            self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

        if verbose:
            print("successful load UNFIED dataset!!")

    def _process_dir_old(self, data_dir, relabel=True):
        img_paths = glob.glob(osp.join(data_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 2501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
