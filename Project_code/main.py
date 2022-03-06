# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import data_generator as dg
import aux_functions as au


def write_filtered_movie_data_to_file():
    dataset_path_n_parents = "../../../"
    # dg.count_valid_rows()
    rating_path = dataset_path_n_parents + "datasets/title.ratings.tsv/data.tsv"
    ds = dg.DataSet(rating_path)
    print(len(ds.data_map))

    ds.filter_dataset(dg.ratings_filter(1000))
    print(len(ds.data_map))

    basics_path = dataset_path_n_parents + "datasets/title.basics.tsv/data.tsv"
    ds.extend_attributes(basics_path, ["titleType", "genres", "isAdult", "startYear", "runtimeMinutes"])
    ds.filter_dataset(dg.title_type_filter("movie"))
    print(len(ds.data_map))
    ds.write_to_file("collected_movie_data.csv",
                     ["tconst", "titleType", "genres", "runtimeMinutes", "startYear", "isAdult", "averageRating",
                      "numVotes"])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #write_filtered_movie_data_to_file()
    au.read_csv_to_np("collected_movie_data.csv")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
