import pickle
import os


class ProgressTracker:
    """
    A progress tracker for logging when this programme is running
    """
    def __init__(self, progress_file, progress_key):
        """

        :param progress_file:
        :param progress_key:
        """
        self.progress_file = progress_file
        self.progress = []
        self.initialize_progress(progress_key)

    def read_progress(self):
        """
        Read the local pickle file to generate a python object with pickle

        :return: return a python raw object
        """
        with open(self.progress_file, 'rb') as f:
            progress = pickle.load(f)
            return progress

    def write_progress(self, progress):
        """
        Write the raw python object into local pickle file

        :param progress: Object, a python raw object
        """
        with open(self.progress_file, 'wb') as f:
            pickle.dump(progress,f)

    def update_progress(self, key, res,data=None):
        """
        Update the python dict object in local pickle file with new value based on the unique key

        :param key:
        :param res:
        :param data:
        """
        for item in self.progress:
            if key in item.keys():
                item[key]["res"] = res
                item[key]["data"] = data
                break
        self.write_progress(self.progress)

    def get_progress(self,key):
        """

        :param key:
        :return:
        """
        for progress in self.progress:
            if key in progress.keys():
                return progress[key]["res"]
        pass

    def get_progress_data(self,key):
        """

        :param key:
        :return:
        """
        for progress in self.progress:
            if key in progress.keys():
                return progress[key]["data"]
        pass

    def initialize_progress(self, progress_key=None):
        """

        :param progress_key:
        """
        if os.path.exists(self.progress_file):
            # with open(self.progress_file, 'rb') as f:
            self.progress = self.read_progress()
        else:
            # self.progress = []
            for key in progress_key:
                self.progress.append({key: {"res": False, "data":None}})
