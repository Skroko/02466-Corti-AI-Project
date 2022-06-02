import pandas as pd

class logger:
    def __init__(self, load_path: str = None) -> None:
        """
        Stores data for the logger as lists\n
        Store as numpy arrays?? or something else?\n
        This can then be used for plots etc.\n
        """
        if load_path is None:
            self.logged_loss = []
        else:
            tmp = pd.read_csv(load_path)
            self.logged_loss = tmp.values.tolist()

    def save_log_as_csv(self, headers: list, save_path: str) -> None:
        """
        Saves the logged data as a pandas dataframe with a given headers to a given path.\n

        Input:\n
            headers: (list) a list of strings denoting what the headers for each of the saved rows should be\n
                make sure headers is same dims as logged data\n
            save_path: (str) A string denoting the save path.
        """
        df_logged_data = pd.DataFrame(self.logged_loss)
        df_logged_data.columns = headers
        df_logged_data.to_csv(save_path, index=None)

    def log_data(self, data_to_be_logged: list) -> None:
        """
        logs the data, by appending to list.\n
        The dims og the logged data should always be the same.\n
        """
        self.logged_loss.append(data_to_be_logged)

    def __call__(self, data_to_be_logged: list) -> None:
        self.log_data(data_to_be_logged = data_to_be_logged)

if __name__ == "__main__":
    logs = logger()

    for _ in range(5):
        logs.log_data([1,2,3,4])
        logs([0,0,0,2])

    logs.save_log_as_csv(headers = ["Klaus", "er", "en", "fisk"], save_path = "./logged_data/epic_csv.csv")

    logs = logger("./logged_data/epic_csv.csv")
    logs.save_log_as_csv(headers = ["Klaus", "er", "en", "fisk"], save_path = "./logged_data/epic_csv.csv")
       


