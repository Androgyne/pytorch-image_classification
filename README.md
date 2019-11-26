# pytorch image classification
'''
this is image classification for our srtp 2019

'''

## Colab
Google Colab provide us with Tesla K80 GPU. Follow the following instructions to use

1. 




## Local


### install
we recommend you use gpu to test for better experience

ubuntu16.04 + cuda9.0 + pytorch0.4.1 is our test environment(garbage gpu mx940)

### usage

1. preprocess data

  we need to make the test data like the following
```
  data/
       data_dir/
           train/
              class1/
              class2/
              ...
              classn/
           val/
              class1/
              class2/
              ...
              classn/

```  
  we have writen a preprocess for our srtp data, you can refer to it

```
  mkdir data

  python run.py --type prepare_data
```

2. train
```
  python run.py --type main --dataset data_dir --network resnet18
```

3. test

```
  python run.py --test True
```


## solution

We are committed to solving the problem of small data sets, we use excellent transfer learning and data augmentaion to solve the problem of small data sets.

We use ConvNet as fixed feature extractor stragey 

## reference

[pytorch transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)




