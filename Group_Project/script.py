import pandas as pd
import os
import json
import re
import requests

from tqdm import tqdm
from bs4 import BeautifulSoup

# import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, ArrayType, StringType, IntegerType, FloatType

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier


class DataManager():
    column_dtypes = {
        "tconst": 'str',
        "primaryTitle": 'str',
        "originalTitle": 'str',
        "startYear": 'str',
        "endYear": 'str',
        "runtimeMinutes": 'str',
        "label": 'str'
    }
    mapper = {
        'ï': 'i',
        'Ü': 'U',
        'Á': 'A',
        'î': 'i',
        'Ô': 'O',
        'é': 'e',
        'Ö': 'O',
        'ä': 'a',
        'ù': 'u',
        'ë': 'e',
        'À': 'A',
        'è': 'e',
        'ß': 'S',
        'Â': 'A',
        'ã': 'a',
        'ý': 'y',
        'ó': 'o',
        'ớ': 'o',
        'ò': 'o',
        'ú': 'u',
        'á': 'a',
        'ì': 'i',
        'ü': 'u',
        'Ç': 'C',
        'Ớ': 'O',
        'í': 'i',
        'ô': 'o',
        'ê': 'e',
        'û': 'u',
        'ç': 'c',
        'à': 'a',
        'É': 'E',
        'ñ': 'n',
        'å': 'a',
        'Ú': 'U',
        'â': 'a',
        'Å': 'A',
        'ö': 'o',
        'Î': 'I'
    }
    
    def __init__(self) -> None:
        self.spark = SparkSession\
                    .builder\
                    .master("local[6]")\
                    .config("spark.driver.bindAddress","localhost")\
                    .config("spark.ui.port","4050")\
                    .getOrCreate()

        self.imdb_data_folder = "imdb/data"
        self.scraped_data_location = "imdb/data/api/scraped_movie_data"
        self.pat = r'[a-zA-Z0-9 ,°!?@#$%&:;+~_/\-\"\'\^\*\(\)\.\[\]]'
        self.unicode_url = "https://www.compart.com/en/unicode"

        # these 3 come from the same file but are split up --> before training we need to do the same operations on all 3 datasets
        self.load_train_data()
        self.load_validation_data()
        self.load_test_data()

        self.restore_misplaced_years()
        self.clean_corrupted_columns_spark(train_only=False)
        self.load_json_data()

    def load_train_data(self):
        path = f"{self.imdb_data_folder}/csv/train" 
        training_data_files = os.listdir(path=path)

        train_df = pd.DataFrame()
        for file in training_data_files:
            if '0' in file:
                continue
            df = pd.read_csv(f"{path}/{file}", index_col=[0])
            train_df = pd.concat([train_df, df], ignore_index=False)

        self.train_df = train_df.sort_index().astype(dtype=self.column_dtypes) 
        self.spark_train_df = self.spark.createDataFrame(self.train_df).replace(to_replace='\\N', value=None)

        del self.column_dtypes['label'] 
    
    def load_validation_data(self):
        path = f"{self.imdb_data_folder}/csv/test_and_validation" 
        df = pd.read_csv(f"{path}/validation_hidden.csv", index_col=[0])
        self.validation_df = df.sort_index().astype(dtype=self.column_dtypes)
        self.spark_validation_df = self.spark.createDataFrame(self.validation_df).replace(to_replace='\\N', value=None)

    def load_test_data(self):
        path = f"{self.imdb_data_folder}/csv/test_and_validation" 
        df = pd.read_csv(f"{path}/test_hidden.csv", index_col=[0])
        self.test_df = df.sort_index().astype(dtype=self.column_dtypes)
        self.spark_test_df = self.spark.createDataFrame(self.test_df).replace(to_replace='\\N', value=None)

    def load_json_data(self):
        path = f"{self.imdb_data_folder}/json"
        
        self.directing_df = pd.read_json(f"{path}/directing.json")
        self.writing_df = pd.read_json(f"{path}/writing.json")

        self.joined_df = pd.merge(self.writing_df, self.directing_df, how='left', on='movie')


    # Cleaning step #1
    def restore_misplaced_years(self):
        self.spark_train_df = self.spark_train_df\
                                    .withColumn('startYear', F.coalesce('startYear', 'endYear').cast(IntegerType()))\
                                    .drop('endYear')\

        self.spark_validation_df = self.spark_validation_df\
                                    .withColumn('startYear', F.coalesce('startYear', 'endYear').cast(IntegerType()))\
                                    .drop('endYear')\

        self.spark_test_df = self.spark_test_df\
                                    .withColumn('startYear', F.coalesce('startYear', 'endYear').cast(IntegerType()))\
                                    .drop('endYear')\


# -------------------------------- REMOVING CORRUPTED CHARS PYSPARK -------------------------------------------
    # Cleaning step #2
    def clean_corrupted_columns_spark(self, train_only=True):
        columns = ['primaryTitle', 'originalTitle']
        dfs = [self.spark_train_df, self.spark_validation_df, self.spark_test_df]

        make_string_arrays = F.udf(lambda row: list(row.corruptedChars), ArrayType(StringType()))
        schema = StructType([StructField('chars', StringType(), True)])

        all_chars_df = self.spark.createDataFrame([], schema)
        for df in dfs:
            for column in columns:
                special_chars_groups = df\
                            .select(F.regexp_replace(column, pattern=self.pat, replacement='')\
                            .alias(f'corruptedChars'))\
                            .filter("corruptedChars != ''")
                            
                special_chars = special_chars_groups\
                            .withColumn(
                                'corruptedCharsArr', 
                                make_string_arrays(F.struct(special_chars_groups['corruptedChars']))
                            )\
                            .select(F.explode('corruptedCharsArr'))\

                all_chars_df = all_chars_df.union(special_chars).distinct()
        
        self.all_special_chars = {row.chars for row in all_chars_df.collect()}
        self.load_mapper_spark()
        for column in columns: 
            self.apply_mapper_spark(column, train_only)
        
        # for col in columns:
        #     self.spark_train_df = self.spark_train_df.drop(col)
        #     self.spark_validation_df = self.spark_validation_df.drop(col)
        #     self.spark_test_df = self.spark_test_df.drop(col)
        
    def load_mapper_spark(self):
        for char in self.all_special_chars:
            if char in self.mapper:
                continue
            self.add_char_to_mapper_spark(char)

    def add_char_to_mapper_spark(self, char):
        hex_value = hex(ord(char))[2:]
        unicode = f"U+{hex_value.zfill(4).upper()}"

        response = requests.get(f"{self.unicode_url}/{unicode}")
        soup = BeautifulSoup(response.text, 'html.parser').table.find("tbody")

        rows = soup.find_all("tr")
        nr_rows = len(rows)
        for i, row in enumerate(rows):
            if nr_rows != i + 1:
                continue
            last_row = row
        td = last_row.find('td', {"class": 'second-column'})
        val = td.findChild().text[0]
        if re.match('[a-zA-Z]', val): 
            self.mapper[char] = val

    def apply_mapper_spark(self, column: str, train_only=True):
        for char, val in self.mapper.items():
            self.spark_train_df = self.spark_train_df\
                .withColumn(column, F.regexp_replace(column, pattern=char, replacement=val))
            if not train_only:
                self.spark_validation_df = self.spark_validation_df\
                        .withColumn(column, F.regexp_replace(column, pattern=char, replacement=val))
                self.spark_test_df = self.spark_test_df\
                        .withColumn(column, F.regexp_replace(column, pattern=char, replacement=val))
# -------------------------------- END -------------------------------------------



# -------------------------------- CREATING SCRAPED DATAFRAME FROM TMDB -------------------------------------------

    def update_imdb_to_tmdb_id_mapper(self):
        all_imdb_ids = []
        dfs = [self.spark_train_df, self.spark_validation_df, self.spark_test_df]

        for df in dfs:
            imdb_ids = df.select('tconst').rdd.flatMap(lambda x: x).collect()
            all_imdb_ids += imdb_ids

        with open('imdb/data/api/imdb_to_tmdb_ids.json', 'r') as f:
            imdb_to_tmdb_ids = json.loads(f.read())

        for imdb_id in tqdm(all_imdb_ids):
            if imdb_id in imdb_to_tmdb_ids:
                continue
            tmdb_id = self.get_tmdb_id(imdb_id)
            if tmdb_id:
                imdb_to_tmdb_ids[imdb_id] = tmdb_id
            else:
                with open('error.txt', 'a') as f:
                    f.write(f"[ERROR]: No movie found for {imdb_id}\n")
            
        with open('imdb/data/api/imdb_to_tmdb_ids.json', 'w') as f:
            json.dump(imdb_to_tmdb_ids, f, indent=2)

    def get_tmdb_id(self, movie_id):
        key = os.getenv("TMDB_API_KEY")
        url = f"https://api.themoviedb.org/3/find/{movie_id}?api_key={key}&external_source=imdb_id"
        res = requests.get(url)

        dict_str = res.content.decode("utf-8").replace("true", "True").replace("false", "False").replace('null', 'None')
        movie_data = eval(dict_str)
        try:
            tmdb_id = movie_data['movie_results'][0]['id']
            return tmdb_id
        except:
            return None 

    def update_scraped_movie_data(self):
        current_scraped_movie_data = pd.read_parquet(self.scraped_data_location)
        current_imdb_ids = set(current_scraped_movie_data['tconst'])

        with open("imdb/data/api/imdb_to_tmdb_ids.json", 'r') as f:
            imdb_to_tmdb_ids = json.loads(f.read())

        country_df = pd.read_csv('imdb/data/csv/country_info.csv', sep=',', quotechar='"', index_col='alpha-2')

        basic_cols = ["imdb_id", "original_language", "popularity", "revenue", "runtime", "vote_average", "vote_count"] 
        genre_col = "genres" 
        production_countries_col = "production_countries"

        data_records = []
        for imdb_movie_id, tmdb_movie_id in tqdm(imdb_to_tmdb_ids.items()):
            if imdb_movie_id in current_imdb_ids:
                continue
            else:
                data = self.get_tmdb_movie_data(tmdb_movie_id)

            try:
                genres = [genre['name'] for genre in data[genre_col]]
            except:
                genres = []
            try:
                production_country = data[production_countries_col][0]
                continent = country_df.loc[production_country['iso_3166_1'].upper(), "region"] 
            except Exception as e: 
                with open('error.txt', 'a') as f:
                    f.write(str(e))
                continent  = 'World'

            row = {key: val for key, val in data.items() if key in basic_cols}
            row[genre_col] = genres
            row['continent'] = continent

            data_records.append(row)

        df = pd.DataFrame.from_records(data_records).rename(columns={'imdb_id': 'tconst'})

        genres_df = self.one_hot_encode_genre(df)
        new_scraped_movie_data = df.join(genres_df)

        final_df = pd.concat([current_scraped_movie_data, new_scraped_movie_data]).dropna(subset=['tconst'])
        final_df.to_parquet(self.scraped_data_location, index=False) 

    def get_tmdb_movie_data(self, movie_id):
        key = os.getenv("TMDB_API_KEY")
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={key}"
        res = requests.get(url)

        dict_str = res.content.decode("utf-8").replace("true", "True").replace("false", "False").replace('null', 'None')
        movie_data = eval(dict_str)
        return movie_data

    def one_hot_encode_genre(self, df):
        df_exploded = df[['genres']].explode('genres')
        dummies = pd.get_dummies(df_exploded['genres']).reset_index(drop=False)

        genres_df = dummies.groupby('index').sum().reset_index(drop=True)
        genres_df.columns = [col.lower().replace(' ', '_') for col in genres_df.columns]
        
        return genres_df

# -------------------------------- END -------------------------------------------



# -------------------------------- MACHINE LEARNING PIPELINE -------------------------------------------

    def prepare_datasets(self):
        scraped_movie_data = self.spark.read.parquet(self.scraped_data_location)

        X_train = self.spark_train_df.join(scraped_movie_data, how='inner', on='tconst').persist()
        X_validation = self.spark_validation_df.join(scraped_movie_data, how='inner', on='tconst').persist().withColumn('label', F.lit(''))
        X_test = self.spark_test_df.join(scraped_movie_data, how='inner', on='tconst').persist().withColumn('label', F.lit(''))      
    
        self.X_train = X_train.withColumn("runtimeMinutes", X_train.runtimeMinutes.cast(FloatType()))
        self.X_validation = X_validation.withColumn("runtimeMinutes", X_validation.runtimeMinutes.cast(FloatType()))
        self.X_test = X_test.withColumn("runtimeMinutes", X_test.runtimeMinutes.cast(FloatType()))

    def setup_ml_pipeline(self):
        # see which label is represented the most
        most_frequent_label = 1 if self.X_train.groupby('label').count().orderBy(F.desc('count')).take(1)[0].label == 'True' else 0
        if most_frequent_label == 1:
            string_order_type = 'frequencyAsc'
        else:
            string_order_type = 'frequencyDesc'

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

        return rfc_pipeline 

    def fit_model(self, pipeline: Pipeline):
        model = pipeline.fit(self.X_train)
        return model

    def make_predictions(self, model, unseen_data, original_left, filename_for_saving):
        output = model.transform(unseen_data)

        model_output = output.toPandas()

        original_and_predictions = pd.merge(original_left[['tconst']], model_output[['tconst', 'prediction']], how='left', on='tconst')
        original_and_predictions['prediction'] = original_and_predictions['prediction'].astype(bool)
        original_and_predictions['prediction'].to_csv(f"imdb/data/predictions/{filename_for_saving}", index=False, header=None) 

# -------------------------------- END -------------------------------------------








if __name__ == '__main__':
    m = DataManager()