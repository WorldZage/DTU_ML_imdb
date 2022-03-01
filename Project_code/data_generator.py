# Functions for checking the size of our dataset
def valid_ratings_set():
    rating_path = "../datasets/title.ratings.tsv/data.tsv"
    min_counts = 1000
    valid_ratings = set()
    with open(rating_path, "r", encoding="utf-8") as rating_f:
        attr = rating_f.readline()
        while True:
            try:
                line = rating_f.readline()
                if not line:
                    break
                line = line.strip()
                split_line = line.split("\t")
                row_tconst = split_line[0]
                row_rating = split_line[2]
                row_rating = int(row_rating)
                if row_rating >= min_counts:
                    valid_ratings.add(row_tconst)
            except Exception as e:
                print(e)

    print(f"n_valids from rating: {len(valid_ratings)}")
    return valid_ratings


def valid_movies_set():
    # number of valid movies:
    basics_path = "../datasets/title.basics.tsv/data.tsv"
    valid_movies = set()
    c = 0
    with open(basics_path, "r", encoding="utf-8") as basic_f:
        attr = basic_f.readline()
        while True:
            try:
                line = basic_f.readline()
                if not line:
                    break
                c += 1
                line = line.strip()
                split_line = line.split("\t")
                row_tconst = split_line[0]
                row_type = split_line[1]
                if row_type.lower() == "movie":
                    valid_movies.add(row_tconst)

            except Exception as e:
                print(e)
    print(f"number of IMDb titles: {c}")
    print(f"n_valids from movie: {len(valid_movies)}")
    return valid_movies


def count_valid_rows():
    valid_ratings = valid_ratings_set()
    valid_movies = valid_movies_set()
    print(f"n_valids from intersection: {len(valid_ratings.intersection(valid_movies))}")


# Part of extracting and writing
class DataObj:
    def __init__(self):
        self.attributes = []
        self.value_map = {}


class DataSet:
    def __init__(self, init_ds_filepath):
        self.attributes = []
        self.data_map = {}
        self.initialize_dataset(init_ds_filepath)

    def initialize_dataset(self, init_ds_filepath):
        with open(init_ds_filepath, "r", encoding="utf-8") as init_f:
            attribute_names = init_f.readline().strip().split("\t")
            self.attributes = attribute_names

            while True:
                try:
                    # read the line and format it to be useful
                    line = init_f.readline()
                    if not line:
                        break
                    line = line.strip()
                    split_line = line.split("\t")

                    # Create the object which will hold the row's data.
                    row = DataObj()
                    row.attributes = attribute_names

                    # tconst seems to be always stored in the first column, but we want to make sure.
                    tconst = split_line[attribute_names.index("tconst")]  # the tconst of the row.
                    for (idx, att) in enumerate(attribute_names):
                        row.value_map[att] = split_line[idx]
                    self.data_map[tconst] = row
                except Exception as e:
                    print(e)

    def extend_attributes(self, ds_extension_filepath: str, wanted_attributes: [str]):

        with open(ds_extension_filepath, "r", encoding="utf-8") as ext_f:
            ext_file_attribute_names = ext_f.readline().strip().split("\t")
            attr_to_col_idx = {}
            for att in wanted_attributes:
                col_idx = ext_file_attribute_names.index(att)
                attr_to_col_idx[att] = col_idx
            self.attributes += wanted_attributes

            while True:
                try:
                    # read the line and format it to be useful
                    line = ext_f.readline()
                    if not line:
                        break
                    line = line.strip()
                    split_line = line.split("\t")

                    tconst = split_line[ext_file_attribute_names.index("tconst")]
                    # only extend the attributes of titles already present in the dataset:
                    if tconst in self.data_map.keys():
                        row = self.data_map[tconst]

                        for (att) in wanted_attributes:
                            col_idx = ext_file_attribute_names.index(att)
                            row.value_map[att] = split_line[col_idx]
                except Exception as e:
                    print(e)

    def filter_dataset(self, filter_func):
        filter_func(self)

    def write_to_file(self, write_path, attr_order: [str] = None):
        """
        :param write_path:  the destination of the file
        :param attr_order: an optional parameter for the
        :return: nothing
        """
        # if an order for the attributes isn't supplied, do it alphabetically
        if attr_order is None:
            attr_order = sorted(self.attributes)
        # Create the entire list of strings which will be written, & *then* we will write it to a file.
        # first, the attributes
        payload = ["\t".join(attr_order)]
        # then, add a row for each of our dataobjects:
        for (tconst, dataobj) in self.data_map.items():
            row_str = "\n"
            for att in attr_order:
                if dataobj.value_map[att]:
                    # If the attribute in the given order actually exists
                    row_str += f"{dataobj.value_map[att]}\t"
                else:
                    raise Exception("Invalid attributes", "attribute_order")
            row_str = row_str.rstrip() # remove the single excessive \t on the rhs.
            payload += row_str

        with open(write_path, "w", encoding="utf-8") as write_file:
            write_file.writelines(payload)  # Write all the attributes to the first line


# filter on the dataset for only using titles who have above a certain number of votes
def ratings_filter(min_n_votes):
    def ds_arg_func(dataset):
        filtered_dataset = {}
        for tconst, row in dataset.data_map.items():
            num_votes = row.value_map.get("numVotes")
            if num_votes:
                if int(num_votes) > min_n_votes:
                    filtered_dataset[tconst] = row
            else:
                raise Exception("Dataset is missing attribute", "numVotes")
        dataset.data_map = filtered_dataset

    return ds_arg_func


# filter on the dataset for only using titles which are movies.
def title_type_filter(only_title_type):
    def ds_arg_func(dataset):
        filtered_dataset = {}
        for tconst, row in dataset.data_map.items():
            title_type = row.value_map.get("titleType")
            if title_type:
                if title_type == only_title_type:
                    filtered_dataset[tconst] = row
            else:
                raise Exception("Dataset is missing attribute", "title type")
        dataset.data_map = filtered_dataset

    return ds_arg_func
