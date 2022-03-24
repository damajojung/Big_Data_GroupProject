# Big-Data-group13

### Current findings
-----------
- **primaryTitle** 
    > is never empty, but corrupted with special characters. Have some methods in place that can handle this.
- **originalTitle**   
    > can be empty, especially when the movie is American/English. Is cleaned along with **primaryTitle**
- **endYear**  
    > should only be filled for a TV Series  (endYear (YYYY) – TV Series end year. ‘\N’ for all other title types)
    our assignment is on *movies*
    this means if **endYear** is filled it should actually be stored in **startYear**
    after checking this works out well: all *NaN*s in **startYear** can be filled with the years found in **endYear**.
    This is handled by the method `restore_misplaced_years()`
- **runtimeMinutes**  
    > only 13 empty values, which we need to fill.
- **numVotes**  
    > 790 empty values. This will be challenging to solve: we have to look at imputing techniques or some library that fixes this.
- **label**  
    > Never empty, 3969 *False* labels and 3990 *True* labels.


## API

### Open Movie Database
> The OMDB api works really well and retrieving data is very simple. See *imdb/data/api/response.json* for a response example for a specific **tconst**. Unfortunately, this is not allowed.  


### The Movie Database
> I have found another one, completely independent from IMDb, namely The Movie Database (TMDb). If we do not replace all our data with this API, I think we can use it. We can use the IMDb *movie_id* to retrieve the corresponding *movie_id* on TMDb. This allows us to request the TMDb database for that movie. Adding features is then allowed. 

### The MovieLens
> GroupLens Research has collected and made available rating data sets from the MovieLens web site (https://movielens.org). The data sets were collected over various periods of time, depending on the size of the set. (https://grouplens.org/datasets/movielens/). Some additional metadata for this dataset: https://www.kaggle.com/rounakbanik/the-movies-dataset) 

### The Internet Movie Script Database
> Biggest collection of movie scripts available anywhere on the web

### Large Movie Review Dataset
> This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. 

### https://www.the-numbers.com/
> https://github.com/ds-leehanjin/movie-dataset-wrangling-and-visualization/blob/master/zippedData/tn.movie_budgets.csv.gz