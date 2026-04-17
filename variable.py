# pretrain weight save folder
absolute_path = "/remote-home/share/dmb_nas2/yii/COTA/"

# amazon dataset original files
amazon_music = "./data/amazon/music/reviews_CDs_and_Vinyl_5.json"
amazon_movie = "./data/amazon/movie/reviews_Movies_and_TV_5.json"
amazon_book = "./data/amazon/book/reviews_Books_5.json"
# amazon_music = "/remote-home/share/dmb_nas2/yii/amazon_2014/reviews_CDs_and_Vinyl_5.json"
# amazon_movie = "/remote-home/share/dmb_nas2/yii/amazon_2014/reviews_Movies_and_TV_5.json"
# amazon_book = "/remote-home/share/dmb_nas2/yii/amazon_2014/reviews_Books_5.json"

# CDR training test save folder
book_movie_save = "./data/amazon/book-movie/"
movie_music_save = "./data/amazon/movie-music/"
music_movie_save = "./data/amazon/music-movie/"

# preprocess user-mapping file path
user_map_paths = {
    "book": "./data/amazon/book/user-map.txt",
    "music": "./data/amazon/music/user-map.txt",
    "movie": "./data/amazon/movie/user-map.txt",
}
item_map_paths = {
    "book": "./data/amazon/book/item-map.txt",
    "music": "./data/amazon/music/item-map.txt",
    "movie": "./data/amazon/movie/item-map.txt",
}
overlap_save = "./data/amazon/overlap/"

train_data = {
    "book_movie_save": absolute_path + "data/amazon/book-movie/",
    "movie_music_save": absolute_path + "data/amazon/movie-music/",
    "music_movie_save": absolute_path + "data/amazon/music-movie/",
}
emcdr_weight = {
    "movie_music_20": "./data/weight/emcdr/movie-music-20.pt",
    "movie_music_50": "./data/weight/emcdr/movie-music-50.pt",
    "movie_music_80": "./data/weight/emcdr/movie-music-80.pt",
    "music_movie_20": "./data/weight/emcdr/music-movie-20.pt",
    "music_movie_50": "./data/weight/emcdr/music-movie-50.pt",
    "music_movie_80": "./data/weight/emcdr/music-movie-80.pt",
    "book_movie_20": "./data/weight/emcdr/book-movie-20.pt",
    "book_movie_50": "./data/weight/emcdr/book-movie-50.pt",
    "book_movie_80": "./data/weight/emcdr/book-movie-80.pt",
}
main_weight = {
    "movie_music_20": "./data/weight/main/movie-music-20/",
    "movie_music_50": "./data/weight/main/movie-music-50/",
    "movie_music_80": "./data/weight/main/movie-music-80/",
    "music_movie_20": "./data/weight/main/music-movie-20/",
    "music_movie_50": "./data/weight/main/music-movie-50/",
    "music_movie_80": "./data/weight/main/music-movie-80/",
    "book_movie_20": "./data/weight/main/book-movie-20/",
    "book_movie_50": "./data/weight/main/book-movie-50/",
    "book_movie_80": "./data/weight/main/book-movie-80/",
}
mf_weight = {
    "book_100": absolute_path + "data/weight/mf/book_100.pt",
    "movie_100": absolute_path + "data/weight/mf/movie_100.pt",
    "music_100": absolute_path + "data/weight/mf/music_100.pt",
    "movie_music_20": absolute_path + "data/weight/mf/movie_music_20.pt",
    "movie_music_50": absolute_path + "data/weight/mf/movie_music_50.pt",
    "movie_music_80": absolute_path + "data/weight/mf/movie_music_80.pt",
    "music_movie_20": absolute_path + "data/weight/mf/music_movie_20.pt",
    "music_movie_50": absolute_path + "data/weight/mf/music_movie_50.pt",
    "music_movie_80": absolute_path + "data/weight/mf/music_movie_80.pt",
    "book_movie_20": absolute_path + "data/weight/mf/book_movie_20.pt",
    "book_movie_50": absolute_path + "data/weight/mf/book_movie_50.pt",
    "book_movie_80": absolute_path + "data/weight/mf/book_movie_80.pt",
}

# -1-1
RATES_MULTIPLY = 4
RATES_ADD = 1
EPS = 1e-10
# Define global hyperparameters
EMCDR_LR = 0.01
MF_LR = 0.001
EMBBEDDING_DIM = 32
NUM_LAYERS = 3  # Number of GCN layers for LightGCN
BATCH_SIZE = 2048 # 256
ITERATIONS = 40 # 10000
EMCDR_ITERATIONS = 300
ITERS_PER_EVAL = 10  # 100
MAXITER = 100
LAMBDA_E = 0.1
WEIGHT_DECAY = 1e-5
OT_WEIGHT_DECAY = 1e-5