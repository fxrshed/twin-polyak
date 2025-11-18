class BaseCallback:

    def __init__(self, trainer, db_path: str = 'np_experiments_db.csv'):

        self.db_path = db_path
        self.trainer = trainer

    def on_fit_start(self):
        pass 

    def on_fit_end(self):
        pass