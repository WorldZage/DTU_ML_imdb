# Functions for checking the size of our dataset
dataset_path_n_parents = "../../../"


# valid_ratings_set(), valid_movies_set() and count_valid_rows() are used purely for very early explorative analysis:
# figuring out the size of a dataset which had been slightly filtered (titleType, number of ratings).

def valid_ratings_set():
    rating_path = dataset_path_n_parents + "datasets/title.ratings.tsv/data.tsv"
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
    basics_path = dataset_path_n_parents + "datasets/title.basics.tsv/data.tsv"
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

class DataObj:
    """
    This class corresponds to the data each row holds.
    The value_map is a dictionary, mapping attribute names (strings) to the row's value for that attribute.
    for example, value_map["ttconst"] would return the title-id for the DataObj instance.
    """
    def __init__(self):
        self.attributes = []
        self.value_map = {}


class DataSet:
    """
    This class corresponds to the entire collected data, stored in memory.
    It contains a list of all the rows, with their respective attribute values, as DatObj instances.
    The data_map variable is a dictionary, mapping ttconst, the unique title ids, to its respective DataObj instance.
    In addition, the class has functions needed to extend or filter the dataset.
    """
    def __init__(self, init_ds_filepath):
        self.attributes = []
        self.data_map = {}
        self.initialize_dataset(init_ds_filepath)

    def initialize_dataset(self, init_ds_filepath):
        # the delimiter used in the original IMDb dataset file:
        file_delimiter = "\t"

        # read the dataset from the specified file.
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
                    split_line = line.split(file_delimiter)

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

    def run_func_on_ds(self, ds_func):
        ds_func(self)

    def write_to_file(self, write_path, attr_order: [str] = []):
        """
        :param write_path:  the destination of the file
        :param attr_order: an optional parameter for the
        :return: nothing
        """

        # Check if we're missing any attributes from the ordering list
        missing_attr_from_order = [attr for attr in self.attributes if attr not in attr_order]
        if missing_attr_from_order:
            raise ValueError(f"Missing attributes {','.join(missing_attr_from_order)} from attribute-order list")

        # Check if we're including any attributes in the ordering which don't exist:
        invalid_attr_from_order = [attr for attr in attr_order if attr not in self.attributes]
        if invalid_attr_from_order:
            raise ValueError(f"Invalid attributes: {','.join(invalid_attr_from_order)}\nfound in attribute-order list")

        col_delimiter = "\t"
        row_delimiter = "\n"
        # Create the entire list of strings which will be written, & *then* we will write it to a file.
        # first, the attributes
        payload = [col_delimiter.join(attr_order)]
        # then, add a row for each of our dataobjects:
        for (tconst, dataobj) in self.data_map.items():
            row_str_list = []
            for att in attr_order:
                if att in dataobj.value_map.keys():
                    # If the attribute in the given ordering-list actually exists
                    row_str_list.append(str(dataobj.value_map[att]))
                else:
                    raise Exception("attribute_order", f"Invalid attributes: {att}")
            # turn the list of strings containing the row's attribute values into a single string, csv, then
            # append to the payload.
            csv_string = row_delimiter + (col_delimiter.join(row_str_list))
            payload += csv_string

        with open(write_path, "w", encoding="utf-8") as write_file:
            write_file.writelines(payload)  # Write all the attributes to the first line


def extend_by_file_and_tconst(ds_extension_filepath: str, wanted_attributes: [str]):
    def ds_arg_func(dataset):
        with open(ds_extension_filepath, "r", encoding="utf-8") as ext_f:
            ext_file_attribute_names = ext_f.readline().strip().split("\t")
            attr_to_col_idx = {}
            for att in wanted_attributes:
                col_idx = ext_file_attribute_names.index(att)
                attr_to_col_idx[att] = col_idx
            dataset.attributes += wanted_attributes

            while True:
                try:
                    # read the line and format it to be useful
                    line = ext_f.readline()
                    if not line:
                        break
                    line = line.strip()
                    split_line = line.split("\t")

                    # TODO: use attr_to_col_idx["tconst"] here instead.
                    tconst = split_line[ext_file_attribute_names.index("tconst")]
                    # only extend the attributes of titles already present in the dataset:
                    if tconst in dataset.data_map.keys():
                        row = dataset.data_map[tconst]

                        for (att) in wanted_attributes:
                            col_idx = ext_file_attribute_names.index(att)
                            row.value_map[att] = split_line[col_idx]
                except Exception as e:
                    raise e

    return ds_arg_func


def extend_n_episodes(episode_file_path):
    # extension function function for adding attribute of how many episodes a show had.
    def ds_arg_func(dataset):
        new_attr_name = "nEpisodes"
        # add the attribute name to the list of attributes for the dataset:
        dataset.attributes += [new_attr_name]

        # then add a default value of 0 for row
        for row in dataset.data_map.values():
            row.value_map[new_attr_name] = 0

        # Then, traverse the episode_file, and for each episode, increment the number of episodes for the parent-show tconst.
        with open(episode_file_path, "r", encoding="utf-8") as ext_f:
            ext_file_attribute_names = ext_f.readline().strip().split("\t")
            # the columnn index for the parent tconst:
            p_tconst_col_name = "parentTconst"
            p_tconst_col_idx = ext_file_attribute_names.index(p_tconst_col_name)
            while True:
                try:
                    # read the line and format it to be useful
                    line = ext_f.readline()
                    if not line:
                        break
                    line = line.strip()
                    split_line = line.split("\t")

                    p_tconst = split_line[p_tconst_col_idx]
                    # increment the number of episodes for the parent-show:
                    dataset.data_map[p_tconst].value_map[new_attr_name] += 1
                except KeyError as e:
                    # case of the parent show's tconst not being in our dataset (removed by filtering)
                    pass

    return ds_arg_func

def extend_show_duration():
    # extension function for adding the duration of shows, i.e. how many years they ran between first year and final year.
    def ds_arg_func(dataset):
        new_attr_name = "durationYears"
        dataset.attributes += [new_attr_name]
        for row in dataset.data_map.values():
            try:
                startYear = int(row.value_map["startYear"])
                endYear = int(row.value_map["endYear"])
                duration = endYear - startYear
                row.value_map[new_attr_name] = duration
            except ValueError as e:
                # in case either startYear or endYear was missing or N/A
                row.value_map[new_attr_name] = "\\N"
    return ds_arg_func


def ratings_filter(min_n_votes):
    # filter on the dataset for only using titles who have above a certain number of votes
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


def missing_data_filter():
    # remove rows missing any data:
    def ds_arg_func(dataset):
        filtered_dataset = {}
        for tconst, row in dataset.data_map.items():
            row_data = row.value_map.values()
            if "\\N" not in row_data:
                filtered_dataset[tconst] = row
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
