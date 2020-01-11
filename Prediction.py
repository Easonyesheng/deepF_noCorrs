"""
Test the model trained before.
@Eason
"""
import os
import numpy as np 
import tensorflow as tf 
from models import HomographyNet 
import data_loader


MateName = ''
ModelPath = r'D:\F_Estimation\deepF_noCorrs\Code\log\oneside_edloss_sqrt_W-5_lr-5_time1578644678'

DataPath = 'D:\\F_Estimation\\deepF_noCorrs\\pred_npy\\norm'

savepath = 'D:\\F_Estimation\\deepF_noCorrs\\Pred_Result\\'

keywords = 'onesqrt_edw-5'

def img_prep(img_path, target_size = (256, 256)):
    img = cv2.imread(img_path, 0)
    # print ('img before reshping: ', img.shape, 'target size: ', target_size)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    # print ('img after reshaping: ', img.shape)
    return img



# Use data_ytil.py to generate npy file
def Genedata(DataPath):
    X = np.load(os.path.join(DataPath, 'Pre_X.npy'))
    return X

def LoadModel(ModelPath,X):
    with tf.Graph().as_default():
        image_shape = [512,1392,1]

        x1 = tf.placeholder(
                    tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name='x1')
        x2 = tf.placeholder(
                    tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name='x2')

        net = HomographyNet(
                    use_reconstruction_module=True,
                    norm_method='norm',
                    use_coor=True, use_idx=True)

        y_ = net(x1, x2, image_shape, False, reuse=False)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            ckpt=tf.train.get_checkpoint_state(ModelPath)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)

           
            for i in range(X.shape[0]):

                img1 = X[i ,:,:,0]
                img2 = X[i ,:,:,1]
                
                img1 = np.expand_dims(img1, axis=3)
                img2 = np.expand_dims(img2, axis=3)

                img1 = np.expand_dims(img1, axis=0)
                img2 = np.expand_dims(img2, axis=0)
                
                feed_dict = {x1 : img1, x2 : img2}
            
        
                fmat = sess.run(y_, feed_dict)
                print('F_Mat: ',fmat)
                np.savetxt(savepath+keywords+'\\'+ str(i) +'.txt',fmat)

        




if __name__ == "__main__":
    X = Genedata(DataPath)
    LoadModel(ModelPath,X)

    

