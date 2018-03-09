import os
from os import path, mkdir, remove
import requests
from zipfile import ZipFile

class download_dataset:
    '''This class is used to download and prepare (unzip) codeneuro dataset.
       The methods are loosly based on download util module created by Alex Klibisz:
       https://github.com/alexklibisz
    '''
    def __init__(self):
        '''Set all the constants
        '''
        self.PATH = os.getcwd()
        self.datasets_dir = PATH + '/codeneuro_data'
        self.dataset_names = sorted([
                            'neurofinder.00.00', 'neurofinder.00.01', 'neurofinder.00.02', 'neurofinder.00.03',
                            'neurofinder.00.04', 'neurofinder.00.05', 'neurofinder.00.06', 'neurofinder.00.07',
                            'neurofinder.00.08', 'neurofinder.00.09', 'neurofinder.00.10', 'neurofinder.00.11',
                            'neurofinder.01.00', 'neurofinder.01.01', 'neurofinder.02.00', 'neurofinder.02.01',
                            'neurofinder.03.00', 'neurofinder.04.00', 'neurofinder.04.01',
                            'neurofinder.00.00.test', 'neurofinder.00.01.test', 'neurofinder.01.00.test',
                            'neurofinder.01.01.test', 'neurofinder.02.00.test', 'neurofinder.02.01.test',
                            'neurofinder.03.00.test', 'neurofinder.04.00.test', 'neurofinder.04.01.test'])

    def download_codeneuro(self):
        '''Download the dataset from the server and unzip them into predifined directory
        '''
        for name in self.dataset_names:
            url = 'https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/' + name + '.zip'
            zip_name = name + '.zip'
            zip_path = self.datasets_dir +'/'+ zip_name

            print('downloading '+zip_name+'...')
            download = requests.get(url)
            print('Download complete, Now writting it to '+zip_path)
            with open(zip_path, 'wb') as zip_file:
                zip_file.write(download.content)
            print('Unzipping..')
            zip_ref = ZipFile(zip_path, 'r')
            zip_ref.extractall(datasets_dir)
            zip_ref.close()

            print('deleting zip file')
            remove(zip_path)
