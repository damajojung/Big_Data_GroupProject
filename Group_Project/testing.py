def load_json_data(self):
    # no difference in movies
    d = set(self.directing_df['movie'])
    w = set(self.writing_df['movie'])
    diff = set.difference(d, w)
    print(diff)
    print(len(diff))


import requests
import os
import ast
import json

# make request to Open Movie Database for a specific tconst
key = os.getenv('API_KEY_PATREON')
url = f"http://www.omdbapi.com/?i=tt3896198&apikey={key}"
res = requests.get(url)

# decode byte content and load into python dict
dict_str = res.content.decode("utf-8")
movie_data = ast.literal_eval(dict_str)

print(json.dumps(movie_data, indent=4))


def applying_mapper(dm):
    for char, val in dm.mapper.items():
        dm.train_df.replace(to_replace=char, value=val, regex=True, inplace=True)
        dm.validation_df.replace(to_replace=char, value=val, regex=True, inplace=True)
        dm.test_df.replace(to_replace=char, value=val, regex=True, inplace=True)
    dm.train_df[dm.train_df['special_chars'].str.len() > 1]



def f(partition):
    x = 0
    for row in partition:
        if str(row.numVotes) == 'nan':
            pass
        else:
            x += row.numVotes
    print(x)

dm.spark_train_df.foreachPartition(f=f)


from pyspark.sql.functions import regexp_replace

for char, val in dm.mapper.items():
    dm.spark_train_df = dm.spark_train_df\
        .withColumn('originalTitle', regexp_replace('originalTitle', pattern=char, replacement=val))\
        .withColumn('primaryTitle', regexp_replace('primaryTitle', pattern=char, replacement=val))

dm.spark_train_df.show()






# -------------------------------- REMOVING CORRUPTED CHARS -------------------------------------------

    def clean_corrupted_columns(self, train_only=True):
        columns = ['originalTitle', 'primaryTitle']
        dfs = [self.train_df, self.validation_df, self.test_df]
        for df in dfs:
            for col in columns:
                special_chars_df = self.isolate_special_chars(df, col)
                self.load_mapper(special_chars_df)
                self.apply_mapper(train_only)
            
            if train_only:
                break

    def isolate_special_chars(self, df, col):
        pattern = r'[a-zA-Z0-9 ,°!?@#$%&:;+~_/\-\"\'\^\*\(\)\.\[\]]'
        df['special_chars'] = df[col].apply(lambda x: re.sub(pattern, '', x))
        return df[['special_chars']].copy()

    def get_corrupted_rows(self, df):
        special_chars_df = df[df['special_chars'] != ''][['special_chars']]
        return special_chars_df.index.tolist()
    
    def load_mapper(self, df):
        for chars in df['special_chars'].tolist():
            for char in chars:
                if char in self.mapper:
                    continue
                self.add_char_to_mapper(char)

    def add_char_to_mapper(self, char):
        hex_value = hex(ord(char))[2:]
        unicode_code_point = f"U+{hex_value.zfill(4).upper()}"

        response = requests.get(f"{self.url}/{unicode_code_point}")
        soup = BeautifulSoup(response.text, 'html.parser').table.find("tbody")

        rows = soup.find_all("tr")
        nr_rows = len(rows)
        for i, row in enumerate(rows):
            if nr_rows != i + 1:
                continue
            last_row = row
        td = last_row.find('td', {"class": 'second-column'})
        # val = td.findChild().text.split(' ')[0]
        val = td.findChild().text[0]
        if re.match('[a-zA-Z]', val): 
            self.mapper[char] = val

    def apply_mapper(self, train_only=True):
        for char, val in self.mapper.items():
            self.train_df.replace(to_replace=char, value=val, regex=True, inplace=True)
            if not train_only:
                self.validation_df.replace(to_replace=char, value=val, regex=True, inplace=True)
                self.test_df.replace(to_replace=char, value=val, regex=True, inplace=True)
    
# -------------------------------- END -------------------------------------------




all_imdb_ids = []
dfs = [dm.spark_train_df, dm.spark_validation_df, dm.spark_test_df]

for df in dfs:
    imdb_ids = df.select('tconst').rdd.flatMap(lambda x: x).collect()
    all_imdb_ids += imdb_ids



import json
from tqdm import tqdm

nr_movies = len(all_imdb_ids)
with open('imdb/data/api/imdb_to_tmdb_ids.json', 'r') as f:
    imdb_to_tmdb_ids = json.loads(f.read())

for imdb_id in tqdm(all_imdb_ids):
    if imdb_id in imdb_to_tmdb_ids:
        continue
    tmdb_id = dm.get_tmdb_id(imdb_id)
    if tmdb_id:
        imdb_to_tmdb_ids [imdb_id] = tmdb_id
    else:
        with open('error.txt', 'a') as f:
            f.write(f"[ERROR]: No movie found for {imdb_id}\n")
    
with open('imdb/data/api/imdb_to_tmdb_ids.json', 'w') as f:
    json.dump(imdb_to_tmdb_ids, f, indent=2)



import json
import pandas as pd
from tqdm import tqdm

current_movie_data = pd.read_parquet("imdb/data/api/scraped_movie_data")

with open("imdb/data/api/imdb_to_tmdb_ids.json", 'r') as f:
    imdb_to_tmdb_ids = json.loads(f.read())

country_df = pd.read_csv('imdb/data/csv/country_info.csv', sep=',', quotechar='"', index_col='alpha-2')

basic_cols = ["imdb_id", "original_language", "popularity", "revenue", "runtime", "vote_average", "vote_count"] 
genre_col = "genres" 
production_countries_col = "production_countries"

data_records = []
for imdb_movie_id, tmdb_movie_id in tqdm(imdb_to_tmdb_ids.items()):

    if imdb_movie_id in current_movie_data['imdb_id']:
        continue
    else:
        data = dm.get_tmdb_movie_data(tmdb_movie_id)

    try:
        genres = [genre['name'] for genre in data[genre_col]]
    except:
        genres = []
    try:
        pc = data[production_countries_col][0]
        continent = country_df.loc[pc['iso_3166_1'].upper(), "region"] 
    except Exception as e: 
        with open('ohjeej.txt', 'a') as f:
            f.write(str(e))
        continent  = 'World'

    row = {key: val for key, val in data.items() if key in basic_cols}
    row[genre_col] = genres
    row['continent'] = continent

    data_records.append(row)

df = pd.DataFrame.from_records(data_records)
df.to_csv("imdb/data/api/scraped_movie_data.csv", sep=';', index=False)

df_exploded = df[['genres']].explode('genres')
dummies = pd.get_dummies(df_exploded['genres']).reset_index(drop=False)
genres_df = dummies.groupby('index').sum().reset_index(drop=True)
genres_df.columns = [col.lower().replace(' ', '_') for col in genres_df.columns]

df = df.join(genres_df)
final_df = pd.concat([current_movie_data, df]).dropna(subset=['tconst'])

final_df.to_parquet("imdb/data/api/scraped_movie_data", index=False)















import pandas as pd
import json

with open("imdb/data/api/tmdb_movie_data_all.json", 'r') as f:
    all_movie_data = json.loads(f.read())

data = []
columns = ["imdb_id", "original_language", "popularity", "revenue", "runtime", "vote_average", "vote_count"]

# add some genres, question is how? One hot encoding?
all_genres = [
    "Action", 
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction", 
    "Thriller",
    "TV Movie",
    "War",
    "Western"
]

# add production company
production_companies = [
      {
        "id": 15997,
        "logo_path": None,
        "name": "Mabel Normand Feature Film Company",
        "origin_country": ""
      }
    ]

# add production country
production_countries =  [
      {
        "iso_3166_1": "US",
        "name": "United States of America"
      }
    ]

country_df = pd.read_csv('imdb/data/csv/country_info.csv', sep=',', quotechar='"', index_col='alpha-2')

# production_comps = {}
for tid, movie_data in all_movie_data.items():
    try:
      # straightforward columns 
      info = {col: movie_data[col] for col in columns}
    except KeyError as e:
      with open("logs.txt", 'a') as f:
        f.write(f"Issue with {tid} with straightforward columns: {e}\n") 
        f.write(f"-------------------------------------------------------------------------------\n\n")

    # add movie genres
    try:
        movie_genres = {genre["name"]: True for genre in movie_data['genres']}
        for genre in all_genres:
          genre_col = genre.lower().replace(' ', '_')
          if genre in movie_genres:
            info[genre_col] = 1
          else: 
            info[genre_col] = 0

    except KeyError as e:
      with open("logs.txt", 'a') as f:
        f.write(f"Issue with {tid} with getting genres: {e}\n") 
        f.write(f"-------------------------------------------------------------------------------\n\n")

    # extract production countries
    try: 
      pc = movie_data['production_countries']
      if len(pc) > 0:
        info['continent'] = country_df.loc[pc[0]['iso_3166_1'].upper(), "region"]
      else:
        info['continent'] = 'World'

    except Exception as e:
      info['continent'] = 'World'
      with open("logs.txt", 'a') as f:
        f.write(f"Issue with {tid} with getting continent: {e}\n") 
        f.write(f"{movie_data}\n") 
        f.write(f"-------------------------------------------------------------------------------\n\n")
    
    data.append(info)

df = pd.DataFrame.from_records(data).rename(columns={'imdb_id': 'tconst'})
df.to_parquet("imdb/data/api/additional_movie_data", index=False)



# from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
# from pyspark.ml.evaluation import BinaryClassificationEvaluator
# import numpy as np

# evaluator = BinaryClassificationEvaluator(labelCol="labelIndex", rawPredictionCol="rawPrediction")
# evaluator.evaluate(output)
# evaluator.evaluate(output, {evaluator.metricName: "areaUnderPR"})

# paramGrid = ParamGridBuilder() \
#     .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 10, stop = 50, num = 3)]) \
#     .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 3)]) \
#     .build()


# rfc_cv_model = CrossValidator(estimator=pipeline,
#                           estimatorParamMaps=paramGrid,
#                           evaluator=BinaryClassificationEvaluator(),
#                           numFolds=3)

# cvModel = rfc_cv_model.fit(df)
# predictions = cvModel.transform(dm.spark_validation_df)

# cvModel = rfc_cv_model.fit(model)











# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

# STEP 1

all_imdb_ids = []
dfs = [dm.spark_train_df, dm.spark_validation_df, dm.spark_test_df]

for df in dfs:
    imdb_ids = df.select('tconst').rdd.flatMap(lambda x: x).collect()
    all_imdb_ids += imdb_ids

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

# STEP 2

import json
from tqdm import tqdm

nr_movies = len(all_imdb_ids)
with open('imdb/data/api/imdb_to_tmdb_ids.json', 'r') as f:
    imdb_to_tmdb_ids = json.loads(f.read())

for imdb_id in tqdm(all_imdb_ids):
    if imdb_id in imdb_to_tmdb_ids:
        continue
    tmdb_id = dm.get_tmdb_id(imdb_id)
    if tmdb_id:
        imdb_to_tmdb_ids[imdb_id] = tmdb_id
    else:
        with open('error.txt', 'a') as f:
            f.write(f"[ERROR]: No movie found for {imdb_id}\n")
    
with open('imdb/data/api/imdb_to_tmdb_ids.json', 'w') as f:
    json.dump(imdb_to_tmdb_ids, f, indent=2)


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

# STEP 3

import json
import pandas as pd
from tqdm import tqdm

current_movie_data = pd.read_parquet("imdb/data/api/scraped_movie_data")

with open("imdb/data/api/imdb_to_tmdb_ids.json", 'r') as f:
    imdb_to_tmdb_ids = json.loads(f.read())

country_df = pd.read_csv('imdb/data/csv/country_info.csv', sep=',', quotechar='"', index_col='alpha-2')

basic_cols = ["imdb_id", "original_language", "popularity", "revenue", "runtime", "vote_average", "vote_count"] 
genre_col = "genres" 
production_countries_col = "production_countries"

data_records = []
for imdb_movie_id, tmdb_movie_id in tqdm(imdb_to_tmdb_ids.items()):

    if imdb_movie_id in current_movie_data['imdb_id']:
        continue
    else:
        data = dm.get_tmdb_movie_data(tmdb_movie_id)

    try:
        genres = [genre['name'] for genre in data[genre_col]]
    except:
        genres = []
    try:
        pc = data[production_countries_col][0]
        continent = country_df.loc[pc['iso_3166_1'].upper(), "region"] 
    except Exception as e: 
        with open('error.txt', 'a') as f:
            f.write(str(e))
        continent  = 'World'

    row = {key: val for key, val in data.items() if key in basic_cols}
    row[genre_col] = genres
    row['continent'] = continent

    data_records.append(row)

df = pd.DataFrame.from_records(data_records)
df.to_csv("imdb/data/api/scraped_movie_data.csv", sep=';', index=False)

df_exploded = df[['genres']].explode('genres')
dummies = pd.get_dummies(df_exploded['genres']).reset_index(drop=False)
genres_df = dummies.groupby('index').sum().reset_index(drop=True)
genres_df.columns = [col.lower().replace(' ', '_') for col in genres_df.columns]

df = df.join(genres_df)

final_df = pd.concat([current_movie_data, df]).dropna(subset=['tconst'])
final_df.to_parquet("imdb/data/api/scraped_movie_data", index=False)

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #


from pyspark.sql.functions import lit 

scraped_movie_data = dm.spark.read.parquet(dm.scraped_data_location)

df_train = dm.spark_train_df.join(scraped_movie_data, how='inner', on='tconst').persist()
df_validation = dm.spark_validation_df.join(scraped_movie_data, how='inner', on='tconst').persist().withColumn('label', lit(''))
df_test = dm.spark_test_df.join(scraped_movie_data, how='inner', on='tconst').persist().withColumn('label', lit(''))

# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #


from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import desc

# see which label is represented the most
most_frequent_label = 1 if df_train.groupby('label').count().orderBy(desc('count')).take(1)[0].label == 'True' else 0
if most_frequent_label == 1:
    string_order_type = 'frequencyAsc'
else:
    string_order_type = 'frequencyDesc'


df_train = df_train.withColumn("runtimeMinutes", df_train.runtimeMinutes.cast(IntegerType()))
df_validation = df_validation.withColumn("runtimeMinutes", df_validation.runtimeMinutes.cast(IntegerType()))
df_test = df_test.withColumn("runtimeMinutes", df_test.runtimeMinutes.cast(IntegerType()))


imputer = Imputer(inputCols=['numVotes', 'runtimeMinutes'], outputCols=['numVotes', 'runtimeMinutes'], strategy='median')

indexer1 = StringIndexer(inputCol='label', outputCol='labelIndex', handleInvalid='keep', stringOrderType=string_order_type)

indexer2 = StringIndexer(inputCol='continent', outputCol='continentIndex', handleInvalid='keep')

one_hot_encoder = OneHotEncoder(inputCol=indexer2.getOutputCol(), outputCol='continentOHE')

assembler = VectorAssembler(inputCols=['startYear', 'runtimeMinutes', 'numVotes', 
                                'popularity', 'revenue', 'vote_average', 'vote_count', 
                                'action', 'adventure', 'animation', 'comedy', 'crime', 
                                'documentary', 'drama', 'family', 'fantasy', 'history', 
                                'horror', 'music', 'mystery', 'romance', 'science_fiction', 
                                'thriller', 'tv_movie', 'war', 'western', 'continentOHE'], outputCol="features")

rfc = RandomForestClassifier(labelCol=indexer1.getOutputCol(), featuresCol=assembler.getOutputCol())

rfc_pipeline = Pipeline(stages=[imputer, indexer1, indexer2, one_hot_encoder, assembler, rfc])


model = rfc_pipeline.fit(df_train)


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #

def predict_with_val_or_test(spark_df, original_left: pd.DataFrame, filename_for_saving: str):

    output = model.transform(spark_df)

    model_output = output.toPandas()

    original_and_predictions = pd.merge(original_left[['tconst']], model_output[['tconst', 'prediction']], how='left', on='tconst')
    original_and_predictions['prediction'] = original_and_predictions['prediction'].astype(bool)
    original_and_predictions['prediction'].to_csv(f"imdb/data/predictions/{filename_for_saving}", index=False, header=None)


# PREDICTIONS FOR VALIDATION SET ==> CHANGE FILENAME FIRST
filename_validation = 'CHANGE_ME.csv'
predict_with_val_or_test(df_validation, dm.validation_df, filename_validation)


# PREDICTIONS FOR TEST SET ==> CHANGE FILENAME FIRST
filename_test = 'CHANGE_ME.csv'
predict_with_val_or_test(df_test, dm.test_df, filename_test)