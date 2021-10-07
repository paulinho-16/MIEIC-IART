from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class Executor:
    _instance = None

    @staticmethod
    def getInstance():
        if Executor._instance == None:
            Executor()
        return Executor._instance

    @staticmethod
    def get_thread_executor():
        return ThreadPoolExecutor(max_workers=20)

    @staticmethod
    def get_process_executor():
        return ProcessPoolExecutor(max_workers=10)

    def __init__(self):
        if Executor._instance != None:
            raise Exception("This class is an Executor!")
        else:
            Executor._instance = self