from pyspark import SparkContext,SparkConf
from extraction import NMF
import thunder as td
import json
from neurofinder import load, match


#Create the Spark Config and Context.
conf = SparkConf().setAppName('P3ImageClassification')
sc = SparkContext.getOrCreate(conf=conf)

#Declare the output array.
outputArray= []

#Load data from the source path.
#TODO: Need to remove hardcodings.
images = td.images.fromtif('/home/maulik/DataSciencePracticum/Project3/neurofinder.00.01/testimages', ext='tiff')
images = td.series.fromarray(data,engine=sc)

#Run the NMF over the algorithms.
nmf_algo = NMF(k=5, percentile=99, max_iter=50, overlap=0.1)
nmf_model = nmf_algo.fit(images, chunk_size=(50,50), padding=(25,25))
merged = nmf_model.merge(0.1)
regions = [{'id': '00.01','coordinates': region.coordinates.tolist()} for region in merged.regions]
#result = {'id': '00.01', 'regions': regions}
outputArray.append(regions)

with open('/home/maulik/DataSciencePracticum/Project3/neurofinder.00.01/regions/output.json', 'w') as file:
    file.write(json.dumps(outputArray))

print('Done')


#Compare outputs.
#TODO: Need to remove hardcodings.
a = load('/home/maulik/DataSciencePracticum/Project3/neurofinder.00.01/regions/regions.json')
b = load('/home/maulik/DataSciencePracticum/Project3/neurofinder.00.01/regions/output.json')
match(a, b)
