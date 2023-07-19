import cv2

def is_cuda_cv(): # 1 == using cuda, 0 = not using cuda
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            return 1
        else:
            return 0
    except:
        return -1
    
def find_cuda():
    for i in range(10):
        try:
            cv2.cuda.printShortCudaDeviceInfo(i)
        except:
            print(i , " was not available")
            continue
if __name__ == '__main__':
    print(is_cuda_cv())
    find_cuda()
    