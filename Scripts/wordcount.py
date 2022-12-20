from nltk.corpus import stopwords
import boto3
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.context import SparkContext as sc
from pyspark import SparkConf
from pyspark import SparkFiles
#spark = SparkSession.builder.getOrCreate()
sc = sc.getOrCreate(SparkConf().setMaster("local[*]"))

s3 = boto3.client('s3')

text_file=s3.download_file('advancedcloud-tp2', '219-0.txt', '219-0.txt')
#removing punctuations and stop word
punc = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
stop_words = set(stopwords.words('english'))

# removing puntuation
def uni_to_clean_str(x):
    converted = x.encode('utf-8')
    lowercased_str = converted.lower()
   # lowercased_str = lowercased_str.replace('--',' ')
    #clean_str = lowercased_str.translate(None, punc)
    #return clean_str
    return lowercased_str

one_RDD = sc.textFile('/home/ubuntu/219-0.txt').flatMap(lambda x: uni_to_clean_str(x).split())
#removing stopword
one_RDD = one_RDD.filter(lambda x: x.lower() not in stop_words)

#   map each word and assign 1
one_RDD = one_RDD.map(lambda w: (w, 1))
#reduce each word
one_RDD = one_RDD.reduceByKey(lambda x,y: x + y)
one_RDD = one_RDD.sortByKey(False)

# Show the top 20 most frequent words and their frequencies
for word in one_RDD.take(20):
     print("{} has {} counts". format(word[1], word[0]))
one_RDD.saveAsTextFile("s3a://advancedcloud-tp2/output ")
