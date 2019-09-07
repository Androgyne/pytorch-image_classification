import numpy as np
import matplotlib.pyplot as plt

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def preprocess_data(input_dir, output_dir, p=0.7):
    '''
    this function is for our srtp data
    please modify data dir name 1 to normal
    dir name 2 to fragment
    dir name 3 to gather
    this function is mainly handle complex '(' ')' ' ' in the image name

    '''
    classes = os.listdir(input_dir)
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, 'train'))
    os.mkdir(os.path.join(output_dir, 'val'))
    for cls in classes:
        os.mkdir(os.path.join(output_dir, 'train', cls))
        os.mkdir(os.path.join(output_dir, 'val', cls))
        jpegs = os.listdir(os.path.join(input_dir, cls))
        idxs = np.random.random((len(jpegs)))
        for i in range(len(jpegs)):
            jpgs = re.split('\(|\)' ,jpegs[i])
            if cls == 'normal':
                jpg = '1\ '+'\('+jpgs[1]+'\)'+jpgs[2]
            else:
                jpg = jpgs[0]+'\('+jpgs[1]+'\)'+jpgs[2]
            if idxs[i] > p:
                os.system('cp {} {}'.format(os.path.join(input_dir, cls, jpg),
                        os.path.join(output_dir, 'val', cls, jpg)
                        ))
            else:
                os.system('cp {} {}'.format(os.path.join(input_dir, cls, jpg),
                        os.path.join(output_dir, 'train', cls, jpg)
                        ))


