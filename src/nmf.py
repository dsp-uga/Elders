"""NMF for the neuron images.

NMF(non-negative factorization) is the factorization method for a matrix to find features from it. For 
the neuron images, it would identify the possible neurons that can spark in the given scenario.


The implementation uses the thunder API, which can do NMF parallel with Spark. The code is more or less similar
to the given neuron example link in the references except updated parameters and use of spark.


Example:
	How to run::

        $ python nmf.py -a<Download Path> -b<File Names for the files to download(comma separated) (e.g neurofinder.00.00.test,neurofinder.00.01.test)> -c<Storage Path in local machine>
			-d<Output Path in local machine> -e<Spark parameters settings (comma separated) (e.g. spark.driver.memory=6G,spark.executor.memory=6G)>



Todo: Remove parameter hardcodings for NMF parameters.


References: 
NMF Wiki: https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
Thunder extraction API : https://github.com/thunder-project/thunder-extraction
Neuron Result comparison : https://github.com/codeneuro/neurofinder
NMF example for neuron images in thunder: https://gist.github.com/freeman-lab/330183fdb0ea7f4103deddc9fae18113


"""

from pyspark import SparkContext,SparkConf
from extraction import NMF
import thunder as td
import json
from neurofinder import load, match,centers,shapes
import zipfile
import os
import argparse




def fetch_data(file_name):
    """Fetch images from source

    This function fetches the image data from the Google cluster and then stores them into
    the local storage and extracts the images from zip file. 

    Args:
        file_name : is the name of the file to download and extract.
    
    Return:
        None.
    """
    print('Inside fetch')
    #Create a string to get the file with wget command.
    wget_cmd = (str('wget ' + data_path.value + '/'+ file_name + '.zip -o ' + log_file.value 
                    + ' -p ' + store_path.value))
    print(wget_cmd)
    #Run on the shell.
    os.system(wget_cmd)
    print('File Downloaded:')
    #Unzip the file.
    zip_read = zipfile.ZipFile(store_path.value + '/' + file_name + '.zip', 'r')
    zip_read.extractall(store_data_path.value)
    zip_read.close()
    print('File Extracted')
    return


"""
This section defines the spark context and config.
"""
#Create the Spark Config and Context.
conf = SparkConf().setAppName('P3ImageClassification')
sc = SparkContext.getOrCreate(conf=conf)


"""
This section sets and gets the runtime arguments which would give the required input and output paths, 
filenames and spark properties to set. 
"""
#Setting up argument parser.
parser = argparse.ArgumentParser(description='CodeNeuro Image Classification')
parser.add_argument('-a','--dw_path', type=str,
                    help='Download Path')
parser.add_argument( '-b','--file_names' ,help='List of files to download separated by commas.')
parser.add_argument('-c','--store_path', type=str,
                    help='Storage Path for current server')
parser.add_argument('-d','--output_path', type=str,
                    help='Final output path')
parser.add_argument('-e','--spark_params', type=str,
                    help='Spark parameters to set separated by commas.')

args = vars(parser.parse_args())
print(args)

#Get 1.Download path, 2. storage path 3.Setting spark parameters. 4. List of files 5. Final output path.
#Broadcast the required paths and create arrays from comma separated values.
dw_path = args['dw_path']
data_path = sc.broadcast(dw_path) #Broadcast.
print(data_path.value)
file_names = args['file_names']
input_array = file_names.split(',')
print(input_array)
str_path = args['store_path']
store_path = sc.broadcast(str_path) #Broadcast.
print(store_path.value)
output_path = args['output_path']
store_data_path = sc.broadcast(output_path) #Broadcast.
print(store_data_path.value)
spark_params = args['spark_params']
spark_param_arrays = spark_params.split(',')
print(spark_param_arrays) 
#Set the Spark Parameters.
#The input should be param_name=param_value.
for paramString in spark_param_arrays:
    param = paramString.split('=')
    conf =conf.set(param[0], param[1])

"""
This section declares some extra variables, and fetches the data from the source.
"""
#Set extra logfile name as broadcast variable.
log_file = sc.broadcast('log.log')
#Fetch and extract the data.
#image_rdd = sc.parallelize(input_array)
#print(image_rdd.take(1))
#image_rdd.map(fetch_data)
#image_rdd.collect()


for input in input_array:
    fetch_data(input)

print('Data Fetched.')



"""
This section processes the image folders one by one. The parameters are set after some experiments, but needed to be passed from 
arguments. 
"""
#Declare the output array.
output_array= []
#Get the list of directories inside the
for root,dirs,files in os.walk(store_data_path.value):
    for dir in dirs:
        #Load data from the source path and convert into RDD for parallalization.
        images = td.images.fromtif(store_data_path.value + '/' + dir + '/images', ext='tiff')
        images = td.series.fromarray(images,engine=sc)
        
        #Run the NMF over the algorithms.
        #TODO: Set the parameters from command-line
        nmf_algo = NMF(k=10, max_iter=50) #Use default percentile 95 and overlap.
        nmf_model = nmf_algo.fit(images, chunk_size=(64,64),padding=(8,8)) #Set after some experiments.
        nmf_merge = nmf_model.merge()

        #Write the output to the final array to convert 
        #The output is array, not RDD. So no reason to use RDD operations for sequential tasks.
        nmf_regions = [{'coordinates': region.coordinates.tolist()} for region in nmf_merge.regions]
        output = {'dataset': dir, 'regions': nmf_regions}
        output_array.append(output)
    break

#Writing the output file.
with open(store_data_path.value + '/output.json', 'w') as output_file:
    output_file.write(json.dumps(output_array))

print('Done.')


#Note : Kept here for information purpose. Also, requires different file format for comparison.

#In case to compare the results for training files.
#Remove extra pairs of parenthesis from the output.json
#Compare outputs.
#a = load('/home/maulik/DataSciencePracticum/Project3/neurofinder.00.01/regions/regions.json')
#b = load('/home/maulik/DataSciencePracticum/Project3/neurofinder.00.01/regions/output.json')
# print(match(a, b))
# print(centers(a, b))
# print(shapes(a, b))
#Set the parameters for the threshold
#Does not seem to  work properly.
# print(match(a, b,10))

