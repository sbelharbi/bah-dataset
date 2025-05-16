import os

if __name__ == '__main__':
    directory = '/projets2/AS84330/Datasets/Abaw6/compacted_48'
    
    i = 0
    for x in os.walk(directory):
        if (len(x[2]) == 0):
            continue        
        old_file = os.path.join(x[0], 'video128.npy')
        new_file = os.path.join(x[0], 'video.npy')
        print(old_file, new_file)
        os.rename(old_file, new_file)
        i+=1
    print(i)