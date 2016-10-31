## HCNN
Hybrid Convolutional Neural Network for sentiment strength prediction

## Running the program
Run the baseline model:

    python baseline.py PA PB
    
where PA, PB represent model name and dataset name respectively. PA is `CNN` or `LSTM`, where the former is the **CNN-non-static** proposed in [Kim's paper](http://www.aclweb.org/anthology/D14-1181) and the latter is the basic **LSTM** model using the last hidden state as the sentence representation.

Run HCNN:
 
    python model.py
   
If GPU and CUDA toolkit are available on your machine, Keras will use GPU in default case. To accelerate the execution, you can add theano flags `lib.cnmem=1` (Note: it only works on theano backend):

    THEANO_FLAGS="lib.cnmem=1" python model.py
    
## Settings
## Environments
* OS: REHL Server 6.4 (Santiago)
* GPU: GeForce GTX Titan Black
* CUDA: 7.0
* Python: Interpreter (2.7.12), Keras(1.0.1), Theano(0.8.2), scikit-learn(0.17.1), nltk(3.1), numpy(1.11.0)

Note: the code is only tested on the above environment

