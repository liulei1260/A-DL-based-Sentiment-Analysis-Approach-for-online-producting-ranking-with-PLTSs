# train.py

from utils import *
from model import *
from config import Config
import sys
import torch
import torch.optim as optim
from torch import nn
import scipy
import scipy.io
import time

if __name__=='__main__':
    config = Config()
    #train_file = 'C:/Users/Zixu.Liu/PycharmProjects/testmining/ag_news.train'
    #train_file = 'C:/Users/Zixu.Liu/PycharmProjects/testmining/ag_news - Copy.train'
    #train_file = './data/train.train'
    # if len(sys.argv) > 2:
    #     train_file = sys.argv[1]
    #test_file = 'C:/Users/Zixu.Liu/PycharmProjects/testmining/ag_news.test'
    #test_file = './data/test.test'
    # if len(sys.argv) > 3:
    #     test_file = sys.argv[2]
    #final_test='./data/final_test.test'
    
    train_file = '../all_data/train.train'
    test_file = '../all_data/test.test'
    final_test='../all_data/final_test.test'
    
    dataset = Dataset(config)
    dataset.load_data(train_file, test_file,final_test)
    since = time.time()
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = CharCNN(config, len(dataset.vocab), dataset.embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(loss_fn)
    ##############################################################
    
    train_losses = []
    val_accuracies = []
    
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_loss,val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        #print('train_loss: {:.4f}'.format(train_loss))
        #print('val_accuracy: {:.4f}'.format(val_accuracy))
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc,test_f1,test_precision,test_recall = evaluate_model_all(model, dataset.test_iterator)
    final_acc,final_f1,final_precision,final_recall=evaluate_model_all(model, dataset.final_iterator)

#    print ('Final Training Accuracy: {:.4f}'.format(train_acc))
#    print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
#    print ('Final Test Accuracy: {:.4f}'.format(test_acc))
#    print ('Final test f1: {:.4f}'.format(test_f1))
#    print ('Final test precision: {:.4f}'.format(test_precision))
#    print ('Final test recall: {:.4f}'.format(test_recall))
#    print ('TV Test Accuracy: {:.4f}'.format(final_acc))
#    print ('TV Test f1: {:.4f}'.format(final_f1))
#    print ('TV Test precision: {:.4f}'.format(final_precision))
#    print ('TV Test recall: {:.4f}'.format(final_recall))
    
    print ('Final Training Accuracy: {:.4f}'.format(train_acc))
    print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print ('Final Test Accuracy: {:.4f}'.format(test_acc))
    print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(test_precision)))
    print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(test_recall)))
    print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(test_f1)))
    #print (test_f1)
    #print (test_precision)
    #print (test_recall)
    print ('TV Test Accuracy: {:.4f}'.format(final_acc))
    print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(final_precision)))
    print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(final_recall)))
    print(' '.join('{:.2f}'.format(k[1]*100) for k in enumerate(final_f1)))
    #print (final_f1)
    #print (final_precision)
    #print (final_recall)
    
    all_preds, all_predict_label=predict_pro(model, dataset.final_iterator)
    scipy.io.savemat('./results_'+config.data_len+'_'+config.model_name+'.mat', mdict={'train_acc': train_acc, 
    'val_acc': val_acc, 'test_acc': test_acc,'test_f1':test_f1,'test_precision':test_precision,'test_recall':test_recall,
    'final_acc': final_acc,'final_f1':final_f1,'final_precision':final_precision,'final_recall':final_recall,'pre_probality':all_preds,'pre_label':all_predict_label})
    torch.save(model.state_dict(), './'+config.data_len+'_'+config.model_name+'.pt')
    print(config.model_name)