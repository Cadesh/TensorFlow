
from multiprocessing import Queue, Process
from myProcess import myProcess

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
class threadCtrl:

    #------------------------------------------------------------------------------------------------
    # 3. INITIALIZE SCHEDULER 
    #------------------------------------------------------------------------------------------------
    def __init__(self, gpuids):
        self._queue = Queue() # https://docs.python.org/3.4/library/multiprocessing.html?highlight=process#multiprocessing.Queue
        self._gpuids = gpuids

        self.__init_myProcesses()
    #------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------
    # 4. Initiate processes - notice the myThreads list that holds each thread
    #------------------------------------------------------------------------------------------------
    def __init_myProcesses(self):
        self._myThreads = list() # a list that will hold the threads of multiprocessing
        for anyGPU in self._gpuids:
            self._myThreads.append(myProcess(anyGPU, self._queue)) # creates a thread for each gpu
    #------------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------
    # 5. START - initiate threads
    #------------------------------------------------------------------------------------------------
    def start(self, filelist):

        # put all of files into queue
        for any in filelist:
            self._queue.put(any)

        #add a None into queue to indicate the end of task
        self._queue.put(None)

        #start the processes
        for any in self._myThreads:
            any.start()

        # wait all fo processes finish
        for any in self._myThreads:
            any.join()
        print("all threads end")
        #------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
