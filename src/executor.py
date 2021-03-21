from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class Executor:
    _instance = None
    _thread_pool = None
    _process_pool = None

    @staticmethod 
    def getInstance():
        if Executor._instance == None:
            Executor()
        return Executor._instance

    @staticmethod 
    def get_thread_executor():
        if Executor._instance == None:
            Executor()
        return Executor._thread_pool

    @staticmethod 
    def get_process_executor():
        if Executor._instance == None:
            Executor()
        return Executor._process_pool

    def __init__(self):
        if Executor._instance != None:
            raise Exception("This class is an Executor!")
        else:
            Executor._instance = self
            Executor._thread_pool = ThreadPoolExecutor(max_workers=20)
            Executor._process_pool = ProcessPoolExecutor(max_workers=10)