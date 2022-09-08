## SETUP LIBRARIES
#library and code setup
!apt-get update
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!pip install -q pyspark

import pyspark, os, sys
from pyspark import SparkConf, SparkContext
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pyspark.sql import SQLContext
import pandas as pd
import urllib.request
from operator import add
import time
import numpy as np


os.environ["PYSPARK_PYTHON"]="python3"
os.environ["JAVA_HOME"]="/usr/lib/jvm/java-8-openjdk-amd64/"

## DOWNLOAD GDELT TONE FILES

# Remove files from previous runs
!rm -rf articles
!rm *.csv

# Multiprocess the query
e = ProcessPoolExecutor()

### Example URL Format ###
# "https://api.gdeltproject.org/api/v2/doc/doc?format=csv&startdatetime=20190301000000&enddatetime=20220401235959&query=%22health%22%20-mental%20sourcecountry:NZ&mode=tonechart"
###

URLbase1 = "https://api.gdeltproject.org/api/v2/doc/doc?format=csv"
# startDate= "&startdatetime=" + queryDate + "000000"
# queryDate = "20190301"          (Example)
# endDate = "&enddatetime=" + queryDate +"235959"
URLbase2 = "&query=%22health%22%20-mental%20sourcecountry:"
#queryCountry = "NZ"
URLbase3 = "&mode=tonechart"

# queryURL = URLbase1 + startDate + endDate + URLbase2 + queryCountry + URLbase3




# Pull down the Tone files as .csv for each country and each date
def getFilename(dates, countryCode):
  # Formats the date and returns a filename
  date = dates.strftime('%Y%m')
  return "{}_{}_tones.csv".format(countryCode, date)

def intoFile(filename):
  # Pulls down the tone data and saves it in the given filename
    try:
        if not os.path.exists(filename):
          queryDate = filename.split("_")[1]
          queryCountry =filename.split("_")[0]
          startDate= "&startdatetime=" + queryDate + "01000000"
          endDate = "&enddatetime=" + queryDate +"28235959"

          queryURL = URLbase1 + startDate + endDate + URLbase2 + queryCountry + URLbase3
          print(queryURL)

          with urllib.request.urlopen(queryURL) as testFile, open(filename, 'w') as f:
            f.write(testFile.read().decode())
        return filename
    except Exception as inst:
        print(type(inst))    # the exception instance
        print(inst.args)     # arguments stored in .args
        print("Error occurred")

# Pull the data from GDELT into multi files; this may take a long time
countries = ['NZ', 'US', 'UK', 'CH', 'RS', 'SF']

resultList = []

for countryCode in countries:
  # Date Range is in months rather than days
  dateRange = [getFilename(dates, countryCode) for dates in pd.period_range('2019 March 1','2022 April 1', freq='M')]
  resultList.append(list(e.map(intoFile, dateRange)))

## ANALYZE TONE FILES USING MAP-REDUCE ALGORITHMS

def getTone(data):
  '''
  Takes a dataframe containing the CSVs of each country
  Uses Map-Reduce algorithms to find the weighted average tone for each CSV
  Returns a list of the average tones for each csv
  '''
  # Sum the total articles within each csv (sum of 'Count' column)
  totalArticles = data.rdd.map(lambda row: (1, int(row['Count']))).reduceByKey(lambda a, b: a + b).map(lambda x: x[1]).collect()

  # Sum the Label*Count Rows (Sum of Row1*Row2)
  countryTone = data.rdd.map(lambda row: (1, (int(row['Label'])*int(row['Count'])))).reduceByKey(lambda a, b: a+b)

  # Find the (weighted) average tone and allocate to and integer on the tone scale 
  countryTone = countryTone.map(lambda tone: toneOnScale(tone[1]/totalArticles[0]))

  # For the Sentiment Progression graph, having the unrounded tone makes the trends more clear
  #countryTone = countryTone.map(lambda tone: tone[1]/totalArticles[0])

  return countryTone.collect()


def toneOnScale(tone):
  '''
  Takes a float of the tone
  Matches it to an integer tone on the scale
  Returns the integer tone

  To avoid a division by zero error while calculating similarities,
  scale was shifted so 1 is very negative and 5 is very positive
  '''
  if tone > 2:
    #Tone is Very Positive
    return 5
  elif tone <= 1.5 and tone > 0.5:
    # Tone is positive
    return 4
  elif tone <= 0.5 and tone > -0.5:
    # Tone is neutral
    return 3
  elif tone <= -0.5 and tone > -1.5:
    # Tone is negative
    return 2
  else:
    # tone is Very Negative
    return 1

NZCSV = resultList[0]
USCSV = resultList[1]
UKCSV = resultList[2]
CHCSV = resultList[3]
RSCSV = resultList[4]
SFCSV = resultList[5]

# Start the Spark Context
sqlContext = SQLContext(sc)

csvNames = [NZCSV, USCSV, UKCSV, CHCSV, RSCSV, SFCSV]
toneVectors = []
# For each country
# Create an RDD of each csv and identify the tone
for i in range(len(csvNames)):
  newList = []
  for j in range(len(NZCSV)):
    newList.append(getTone(sqlContext.read.option("header", "true").csv(csvNames[i][j]))[0])
  #Append the tone of each csv to that country's tone vector
  toneVectors.append(newList)

## FIND SIMILARITY BETWEEN COUNTRIES

def cosineSimilarity(u, v):
  '''
  Finds the similarities between two vectors
  Returns a float representing the similarity
  '''
  similarity = (u.dot(v)) / ((np.linalg.norm(u))*np.linalg.norm(v))
  return similarity

for i in range(len(toneVectors)):
  print("NZ similarity to {}: {:.5f}\n".format(countries[i], cosineSimilarity(NZtone, toneVectors[i])))

## PLOT RESULTS

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'monospace'

# biggerData contains the tones before they were passed through the getTone function
# biggerData will be used to plot the sentiment progression in more detail
# biggerData was hard-coded so that I don't have to rerun the entire code everytime I want to change the plot
biggerData = [[-1.1106271777003485, -0.9692077727952167, -0.8203546646106399, -0.5348301669545192, -0.6661747343565525, -0.6609686609686609, -0.6178102926337034, -0.42286348501664817, -0.45313336904124263, -0.6061162079510704, -1.140822272580123, -1.5743507239715009, -1.5651846978099446, -1.5049794426678849, -0.8694480253134889, -1.2040910489080283, -1.5073636510762258, -1.2523124357656732, -0.9725523012552302, -0.9614502065167508, -0.8317378497790869, -1.0304610890882537, -1.3029008320215876, -0.8028576288484436, -0.6652479644707624, -0.47463337296868807, -0.7186544342507645, -0.6067146282973621, -0.9394896079978953, -1.223968565815324, -0.9839294971487818, -0.6225456225456225, -0.8343243243243244, -0.7282296650717703, -1.117058144012322, -1.1108590870428126, -1.112934996717006, -0.7742699289660616], [-1.2706111752502665, -1.2946058810278802, 0.4407740601090896, 1.6979970543812322, 2.010660163734541, 1.7172846693836032, -1.5332649577088107, -1.2875422930282312, -1.0746235295390063, -1.0969155067264942, -1.357946647946648, -1.9721707403934907, -2.1380669818947986, -2.0293016604943728, -1.7604287541070496, -1.8927157350491999, -1.9039310678594414, -1.5228006781196795, -1.6216533599692247, -1.7829458268230685, -1.5797193750621952, -1.461935384416683, -1.4574814071091164, -1.3017302068666738, -1.310562259376658, -1.4493426041688016, -1.4066877043032815, -1.1912506753548684, -1.5756050969515067, -1.5381760491593803, -1.6277307539706396, -1.3115936294391137, -1.247577605360857, -1.4597271445699003, -1.4165824794342619, -1.4051799262370575, -1.3031616558819443, -1.3830341312056738], [-1.491427806054112, -1.6641093522070345, -1.4819901061527363, -1.334437751004016, -1.3503293622985284, -1.4110062237574819, -1.418343012653256, -1.6242324888226527, -0.04388378977433319, -0.0774944071588367, -1.0349685834962614, -2.513547611895464, -2.2131304021539493, -1.7986939432813822, -1.8007559246282452, -1.7596062133805697, -1.6212448271338447, -1.7899192809951119, -2.0684709190029373, -1.9207075498019084, -1.5897704826332442, -1.7524495209224273, -1.9872346182464753, -1.6668564095133613, -1.687392648574373, -1.6966203783792284, -1.6087287061804871, -1.527838387436731, -1.8259051547556824, -1.5926839393421406, -1.585263622325171, -1.4726349257133715, -1.6280792556448904, -1.571891261925412, -1.8225164920450136, -1.7307625234337534, -1.8170605594370317, -1.7029934150116746], [-0.052085170595959004, -0.14533509559706728, -0.0769155371894346, -0.20873346768600637, -0.18483420045875484, -0.11788250020929426, 0.1360425546976523, 0.5688536967843106, 0.4293568036341111, 0.49326607734438727, -0.41239805220627407, -1.6501810524518556, -1.2586802767901517, -1.019733528276403, -0.5033630647755519, -0.4044667482564574, -0.3413983113249227, 0.02721298194064051, 0.14464725716508883, 0.007212307692307692, 0.38093797276853253, -0.06957880991641012, -0.6679276057223706, -0.027238018236535957, 0.2736867807717452, 0.12534442120638714, 0.14383388883884335, 0.27870979286896613, 0.1231585969203402, -0.6270523255185534, 0.42814161383058325, 0.22213862270250606, 0.30271562851658174, 0.2077629486419907, 0.06953487984812404, 0.6095366333247841, -0.046449845027945094, -0.3348570485581088], [-1.9695654624358903, -1.8013042245534447, -1.821415750138703, -1.8117126887461417, -2.0864945629617333, -2.0741881833301523, -1.7162379421221865, -1.869594704573816, -1.8149876036362667, -1.6265967719685475, -2.2013239664955417, -2.7169275839970704, -2.898279518177645, -2.884759557182655, -2.518602971057028, -2.1731482044114965, -2.262507720815318, -2.293818706626543, -2.107291203654032, -2.3755074424898512, -2.180122829568872, -1.7586206896551724, -2.03998017839445, -1.972182610616803, -1.9283890287141991, -1.8858176663682482, -1.8548824103175463, -1.8215763137453258, -2.0932106315646415, -2.082593949260616, -1.7320604427784936, -1.9807556328472873, -2.135363939399284, -1.8751595562986534, -2.378649792674499, -2.175857947754823, -1.992905788876277, -1.8605566817402153], [-1.8673000306466443, -2.062642740619902, -1.7627230667547917, -1.9338143036386448, -2.0483059376048307, -1.6091772151898733, -1.882134914751668, -1.685456595264938, -2.1151574803149606, -2.056873977086743, -2.67264, -3.012888107791447, -2.8904914068572416, -2.6325331125827813, -2.4003185781354794, -2.3989260543684976, -2.400287318793261, -2.2138647207409488, -1.9838497652582159, -1.9220047449584816, -1.758857929136567, -1.8323256751309955, -1.972876304023845, -1.4165047092240992, -1.6409759876065066, -1.4659489819798737, -1.577247191011236, -1.5507833127970427, -2.0656152647975077, -1.9058073654390935, -1.4408743169398908, -1.2098182676421996, -1.7678652708238507, -1.6375661375661377, -1.8438107582631238, -1.7204391891891893, -1.7734700416533162, -1.7262723521320495]]

# data contains the tones after they were passed through the getTone function
# i.e it contains the simplified tone vectors for each country
# data will be used to calculate and plot the similarities between New Zealand and the other countries
data = [[2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 3, 1, 5, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2], [2, 1, 2, 2, 2, 2, 2, 1, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 4, 3, 3], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1]]

# months is the x-axis for the sentiment progression graph
months = [i for i in range(len(data[0]))]

# A hard-coded similarity list
Similarity = [1.00000, 0.91581, 0.91856, 0.98323, 0.97725, 0.95418]

# formatting for the graphs (keeping colours consistent between then grpahs)
colors = ['black', 'blue', 'green', 'red', 'purple', 'orange']

fig, ax = plt.subplots()

for i in range(len(biggerData)):
  country = biggerData[i]
  plt.plot(months, country, label = countries[i], color = colors[i])
  plt.title("Sentiment Progression")
  plt.xlabel("Months from March 2019")
  plt.ylabel("Average Sentiment")
  plt.legend(ncol = 2)
  plt.grid()

fig.savefig("SentimentProgression.svg", dpi = 2000, bbox_inches='tight')
fig.savefig("SentimentProgression.pdf", dpi = 2000, bbox_inches='tight')




