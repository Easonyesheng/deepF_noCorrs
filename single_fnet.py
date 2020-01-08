"""
Model for single network direct regression
'''
The main training code
'''
"""
import os
import time
import argparse
import data_loader
import numpy as np
import tensorflow as tf
from models import HomographyNet
from evaluation import sampson_dist, sym_epilolar_dist, epipolar_constraint,epipolar_constraint_abs

class SingleFNet(object):

    def __init__(self, tr_data_loader, val_data_loader, test_data_loader=None, net=None,
                 lr=0.001, l1_weight=0., l2_weight=2.,ep_weight=0.001 , batch_size=128,
                 gradient_clipping=False, prefix="test", use_internal_layer=False,
                 use_coor=False, use_idx=False, norm_method='norm', resume=None):
        self.resume = resume
        self.tr_data_loader = tr_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.ep_weight = ep_weight
        self.batch_size = batch_size
        self.prefix = prefix
        self.lr = lr
        self.gc = gradient_clipping
        self.use_internal_layer = use_internal_layer
        self.norm_method = norm_method
        self.use_coor = use_coor
        print("Use coor in Single FNet:%s"%self.use_coor)
        self.use_idx = use_idx
        if net == None:
            self.net = HomographyNet(
                    use_reconstruction_module=self.use_internal_layer,
                    norm_method=self.norm_method,
                    use_coor=self.use_coor, use_idx=self.use_idx)
        else:
            self.net = net


        # these 4 metrics can output 4 value
        self.metrics = {
                "sampson_dist      " : sampson_dist,
                "sym_epilolar_dist " : sym_epilolar_dist,
                "epi_constraint    " : epipolar_constraint,
                "epi_constraint_abs" : epipolar_constraint_abs
        }

        # image shape required for UCN
        self.img_shape = list(self.tr_data_loader.img_shape())[:2] + [1]
        self.image_size_x = self.img_shape[0]
        self.image_size_y = self.img_shape[1]
        self.image_channels = self.img_shape[2]

        # Graph
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),
                                   graph=self.train_graph)
            # Place holders.
            self.x1 = tf.placeholder(
                tf.float32, [None, self.image_size_x, self.image_size_y, self.image_channels], name='x1')
            self.x2 = tf.placeholder(
                tf.float32, [None, self.image_size_x, self.image_size_y, self.image_channels], name='x2')
            # self.correspondences = tf.placeholder(tf.float32, [None, self.nb_corres, 4], name='corres')
            # self.x = tf.placeholder(tf.float32, shape=(
            #          [None] + list(self.tr_data_loader.img_shape())))
            self.y = tf.placeholder(tf.float32, shape=(
                     [None] + list(self.tr_data_loader.fmat_shape())))
        
            self.bp1 = tf.placeholder(tf.float32, shape=([4,54,2]), name='bp1')
            self.bp2 = tf.placeholder(tf.float32, shape=([4,54,2]), name='bp2')

            print("Building training graph...")
            self.y_, self.loss, self.l1_loss, self.l2_loss, self.ep_loss = \
                    self.build_graph(self.tr_data_loader, is_training=True, is_reuse=False)

            # Optimizers.
            print("Building optimizer...")
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optim = tf.train.AdamOptimizer(learning_rate=self.lr)
                grads_and_vars = self.optim.compute_gradients(
                        self.loss, var_list=self.net.vars)
                if self.gc:
                    grads_and_vars = [\
                        (tf.clip_by_norm(g, clip_norm=1.) if g is not None else g, v) \
                        for g,v in grads_and_vars if g is not None]
                self.train_op = self.optim.apply_gradients(grads_and_vars)

            # Validation
            print("Building validation graph...")
            self.val_y_, self.val_l1_loss, self.val_l2_loss, self.val_loss, self.val_ep_loss= \
                    self.build_graph(self.val_data_loader, is_training=False, is_reuse=True)

            # Testing
            print("Building testing graph...")
            self.test_y_, self.test_l1_loss, self.test_l2_loss, self.test_loss, self.test_ep_loss= \
                    self.build_graph(self.test_data_loader, is_training=False, is_reuse=True)

            # Summary
            tf.summary.scalar("l1_loss", self.l1_loss)
            tf.summary.scalar("l2_loss", self.l2_loss)
            tf.summary.scalar("loss",    self.loss)
            tf.summary.scalar("ep_loss", self.ep_loss)

            self.tr_summary = tf.summary.merge_all()
            self.tr_log_writer = tf.summary.FileWriter("log/%s" %self.prefix, \
                                                       self.sess.graph, flush_secs=30)
            self.tr_saver = tf.train.Saver()


    def build_graph(self, data_loader, is_training=True, is_reuse=False):
        # Network.
        y_ = self.net(self.x1, self.x2, self.img_shape, is_training, reuse=is_reuse)
        # print("pre shape:",y_.shape)
        # print("y shape:",self.y.shape)
        # x
        # Loss
        l1_loss = tf.reduce_mean(tf.abs(self.y - y_)) * self.l1_weight
        l2_loss = tf.losses.mean_squared_error(self.y, y_) * self.l2_weight

        '''
        add the loss of epi_constraint_abs
        '''
        # bx, by, bp1, bp2 = self.tr_data_loader(self.batch_size)
        # # a, e_c_loss, b = epipolar_constraint_abs(y_, by, bp1 ,bp2)
        # # r_score, m_score, base_score = m(fmat, by, bp1, bp2)
        # err = 0.0
        # e_c_loss = 0.0
        # count = 0.0
        # bp1 = bp1.reshape([-1,2])
        # bp2 = bp2.reshape([-1,2])
        # # print(bp1)
        # for i in range(self.batch_size):
        #     for p1, p2 in zip(bp1, bp2):
        #         # print(p1.shape)
        #         p1 = p1.reshape([2,1])
        #         p2 = p2.reshape([2,1])
        #         # hp1 = tf.Variable(tf.ones([3,1]), name="hp1")  
        #         # hp2 = tf.Variable(tf.ones([3,1]), name="hp2")  
        #         hp1, hp2 = np.ones([3,1],dtype=np.float32), np.ones([3,1],dtype=np.float32)
        #         hp1[:2], hp2[:2] = p1, p2
        #         y_re = tf.reshape(y_, [self.batch_size,3,3])
        #         err += tf.abs(tf.matmul(hp2.T, tf.matmul(y_re[i,:,:], hp1)))
        #         count+=1
        #     e_c_loss += err
        # count = self.batch_size*count 
        # e_c_loss = (e_c_loss[0,0]/count)*self.ep_weight
        '''
        add loss of sym_epipolar_dist
        '''
        # bx, by, bp1, bp2 = self.tr_data_loader(self.batch_size)
        pts1 = self.bp1
        pts2 = self.bp2
        # pts1 = bp1.reshape([-1,2])
        # pts2 = bp2.reshape([-1,2])
        print(pts1.shape)
        err = 0.
        e_d_loss = 0.
        count = 0
        epsilon = 1e-5
        # door = tf.constant([[0.15]], dtype=tf.float32)
        a = tf.constant([[1]], dtype=tf.float32)
        #(len(pts2)/4)
        for i in range(self.batch_size):
            for j in range(54):
                door = tf.constant(0.15, dtype=tf.float32)
                p1 = pts1[i,j,:]
                p2 = pts2[i,j,:]
                p1 = tf.reshape(p1,[2,1])
                p2 = tf.reshape(p2,[2,1])
                p1 = tf.concat([p1,a],0)
                p2 = tf.concat([p2,a],0)
                # print('p1 shape:',p1.shape)
                # p1 = tf.reshape(p1,[3,1])
                # p2 = tf.reshape(p2,[3,1])
                # print('type ',type(p1[0,0]))
                # hp1, hp2 = np.ones([3,1],dtype=np.float32), np.ones([3,1],dtype=np.float32)
                # hp1[:2,0], hp2[:2,0] = p1, p2
                # hp1[0,0], hp2[0,0] = p1[0], p2[0]
                # hp1[1,0], hp2[1,0] = p1[1], p2[1]
                y_re = tf.reshape(y_, [self.batch_size,3,3])

                # use Gt to choose matching points
                y_gt = tf.reshape(self.y, [self.batch_size,3,3])
                err_gt = tf.abs(tf.matmul(tf.transpose(p2), tf.matmul(y_gt[i,:,:], p1)))
                thre = err_gt[0,0]
                # print(thre.shape)
                # print(type(thre))
                def f1(): return door-0.15
                def f2(): return tf.add(door, 0.85)
                result = tf.cond(thre > door, f1, f2)
                # print(door)
                fp, fq = tf.matmul(y_re[i,:,:], p1), tf.matmul(tf.transpose(y_re[i,:,:]),p2)
                sym_jjt = 1./(fp[0]**2 + fp[1]**2 + epsilon) + 1./(fq[0]**2 + fq[1]**2 + epsilon)
                err = err + ((tf.matmul(tf.transpose(p2), tf.matmul(y_re[i,:,:], p1))**2) * (sym_jjt + epsilon))*door
                count+=1*door
            e_d_loss += err
            print("Use ",count," Points")
        count *= self.batch_size
        e_d_loss = (e_d_loss[0,0]/count)*self.ep_weight

        loss = l1_loss + l2_loss + e_d_loss
        # only use ep_loss
        # loss = e_c_loss

        return y_, loss, l1_loss, l2_loss, e_d_loss

    def validate(self, iters, best_score, val_data_loader=None):
        if val_data_loader == None:
            val_data_loader = self.val_data_loader

        val_loss = 0
        # generate 3 diction with same keys in metrics @Eason
        relative =  { k:0 for k in self.metrics.keys() }
        scores   =  { k:0 for k in self.metrics.keys() }
        base     =  { k:0 for k in self.metrics.keys() }
        # bs = len(self.val_data_loader)
        num_batches = len(self.val_data_loader)//self.batch_size+1
        print ('num batches: ', num_batches, 'val data: ', len(val_data_loader))
        # num_batches = len(self.val_data_loader)//bs+1
        for i in range(num_batches):
            # bx, by, bp1, bp2 = self.val_data_loader(self.batch_size)
            bx, by, bp1, bp2 = self.val_data_loader(self.batch_size)
            img1 = bx[:,:,:,0]
            img2 = bx[:,:,:,1]
            img1 = np.expand_dims(img1, axis=3)
            img2 = np.expand_dims(img2, axis=3)
            feed_dict = {self.x1 : img1, self.x2 : img2, self.y : by, self.bp1 : bp1, self.bp2 : bp2}
            loss, fmat = self.sess.run([self.val_loss, self.val_y_], feed_dict)
            val_loss += loss
            #metrics is a dictionary
            for k in self.metrics.keys():
                m = self.metrics[k]
                # fmat is the networks output, f_pred;
                # by is the f_gt
                r_score, m_score, base_score = m(fmat, by, bp1, bp2)
                # the final output when testing are 3 items below @Eason
                scores[k] += m_score.mean()
                relative[k] += r_score.mean()
                base[k] += base_score.mean()

        output_data = {}
        print("\nValidation:  loss\t%.5f" \
              %(val_loss / float(num_batches)))
        output_data['val_loss'] = val_loss/float(num_batches)
        for k in scores.keys():
            output_data[k] = {}
        ind = 0
        for k in scores.keys():
            s = scores[k] # 
            v = base[k]
            r = relative[k]
            print("\t%s\t%.5f %.5f %.5f"\
                  %(k, r/float(num_batches), s/float(num_batches), v/float(num_batches)))
            
            best_score[ind][0] = min(r/float(num_batches), best_score[ind][0])
            best_score[ind][1] = min(s/float(num_batches), best_score[ind][1])
            best_score[ind][2] = min(v/float(num_batches), best_score[ind][2])
            ind += 1

            output_data[k]['pred'] = s/float(num_batches)
            output_data[k]['gtrs'] = v/float(num_batches)

        output_data['pred_all'] = scores
        output_data['gtrs_all'] = base

        # save_path = "log/single_fnet_%s" %self.prefix
        # save_file = os.path.join(save_path, "val-iteres%d.npy"%iters)
        # np.save(save_file, output_data)

        return best_score

    def test(self, iters, best_score, test_data_loader=None):
        if test_data_loader == None:
            test_data_loader = self.test_data_loader

        test_loss = 0
        relative =  { k:0 for k in self.metrics.keys() }
        scores   =  { k:0 for k in self.metrics.keys() }
        base     =  { k:0 for k in self.metrics.keys() }
        # bs = len(self.val_data_loader)
        num_batches = len(self.test_data_loader)//self.batch_size+1
        print ('num batches: ', num_batches, 'val data: ', len(test_data_loader))
        # num_batches = len(self.val_data_loader)//bs+1
        for i in range(num_batches):
            # bx, by, bp1, bp2 = self.val_data_loader(self.batch_size)
            bx, by, bp1, bp2 = self.test_data_loader(self.batch_size)
            img1 = bx[:,:,:,0]
            img2 = bx[:,:,:,1]
            img1 = np.expand_dims(img1, axis=3)
            img2 = np.expand_dims(img2, axis=3)
            feed_dict = {self.x1 : img1, self.x2 : img2, self.y : by, self.bp1 : bp1, self.bp2 : bp2}
            loss, fmat = self.sess.run([self.test_loss, self.test_y_], feed_dict)
            test_loss += loss
            for k in self.metrics.keys():
                m = self.metrics[k]
                r_score, m_score, base_score = m(fmat, by, bp1, bp2)  # by is the F_GT got from calib_cam_to_cam.txt
                scores[k] += m_score.mean()
                relative[k] += r_score.mean()
                base[k] += base_score.mean()

        output_data = {}
        print("\nTesting:  loss\t%.5f" \
              %(test_loss / float(num_batches)))
        output_data['test_loss'] = test_loss/float(num_batches)
        for k in scores.keys():
            output_data[k] = {}
        ind = 0
        for k in scores.keys():
            s = scores[k]
            v = base[k]
            r = relative[k]
            print("\t%s\t%.5f %.5f %.5f"\
                  %(k, r/float(num_batches), s/float(num_batches), v/float(num_batches)))
            
            best_score[ind][0] = min(r/float(num_batches), best_score[ind][0])
            best_score[ind][1] = min(s/float(num_batches), best_score[ind][1])
            best_score[ind][2] = min(v/float(num_batches), best_score[ind][2])
            ind += 1

            output_data[k]['pred'] = s/float(num_batches)
            output_data[k]['gtrs'] = v/float(num_batches)

        output_data['pred_all'] = scores
        output_data['gtrs_all'] = base

        # save_path = "log/single_fnet_%s" %self.prefix
        # save_file = os.path.join(save_path, "val-iteres%d.npy"%iters)
        # np.save(save_file, output_data)

        return best_score

    def train(self, epoches=10, log_interval=10):
        ttl_iter = 0
        num_batches = len(self.tr_data_loader) // self.batch_size + 1
        with self.train_graph.as_default():
            if self.resume is None:
                print("Initializing variables..")
                self.sess.run(tf.global_variables_initializer())
            else:
                print("Resuming varibles from :%s"%self.resume)
                self.tr_saver.restore(self.sess, self.resume)

            max_val = 1000000000000000.0
            best_score_val = [[max_val]*3 for i in range(4)]
            best_score_test = [[max_val]*3 for i in range(4)]

            print("Validation")
            self.validate(-1, best_score_val)
            print("Testing")
            self.test(-1, best_score_test)

            print("Start training...")
            for epo in range(epoches):
                print("Training epoch : [%03d/%03d]"%(epo, epoches))
                for i in range(num_batches):
                    ttl_iter += 1
                    bx, by, bp1, bp2 = self.tr_data_loader(self.batch_size)
                    img1 = bx[:,:,:,0]
                    img2 = bx[:,:,:,1]
                    img1 = np.expand_dims(img1, axis=3)
                    img2 = np.expand_dims(img2, axis=3)
                    feed_dict = {self.x1 : img1, self.x2 : img2, self.y : by, self.bp1 : bp1, self.bp2 : bp2}
                    if i % log_interval == 0:
                        summary, loss, l1_loss, l2_loss, ep_loss = self.sess.run([
                            self.tr_summary, self.loss, self.l1_loss, self.l2_loss ,self.ep_loss], feed_dict)
                        print("[epo=%02d/%02d][%05d/%05d] loss = %.5f l1_loss = %.5f l2_loss = %.5f ep_loss = %.5f"\
                              %(epo, epoches, i, num_batches, loss, l1_loss, l2_loss, ep_loss))
                        self.tr_log_writer.add_summary(summary, ttl_iter)
                    self.sess.run(self.train_op, feed_dict)

                best_score_val = self.validate(epo, best_score_val)
                best_score_test = self.test(epo, best_score_test)
                
                scores   =  { k:0 for k in self.metrics.keys() }
                ind = 0
                print ('\nbest validation values till now:')
                for k in scores.keys():
                    print("\t%s\t%.5f %.5f %.5f"\
                            %(k, best_score_val[ind][0], best_score_val[ind][1], best_score_val[ind][2]))
                    ind += 1
                ind = 0
                print ('\nbest test values till now:')
                for k in scores.keys():
                    print("\t%s\t%.5f %.5f %.5f"\
                            %(k, best_score_test[ind][0], best_score_test[ind][1], best_score_test[ind][2]))
                    ind += 1
		
		
                if epo % 10 == 0:
                    print("Saving the model...")
                    save_path = "log\%s" %self.prefix # \ for windows and / for linux & Mac
                    #save the model
                    self.tr_saver.save(self.sess, os.path.join(save_path, "model-%d.ckpt"%epo))
                    
                    print("Model saved to %s"%save_path)
		

# Training scripts
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single FMatrixNet, direct regression.')
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('--prefix', default="default_posidx_absnorm", type=str,
                        help="Prefix for naming the log/model.")
    parser.add_argument('--batch-size', default=4, type=int,
                        help="Batch size.")
    parser.add_argument('--epoches', default=100, type=int,
                        help="Number of epoches.")
    parser.add_argument('--data-size', default=2000, type=int,
                        help="Dataset size as number of (img1, img2, fmat) tuple") # default = 30000
    parser.add_argument('--l1-weight', default=10., type=float,
                        help="Weight for L1-loss")
    parser.add_argument('--l2-weight', default=1., type=float,
                        help='Weight for L2-loss')
    parser.add_argument('--ep-weight', default=1, type=float,
                        help='Weight for ep-loss')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate')
    parser.add_argument('--use-gc', default=True, type=bool,
                        help='Use gradient clipping.')
    parser.add_argument('--norm-method', default='norm', type=str,
                        help='Normalization method: (norm)|(abs)|(last)')
    parser.add_argument('--resume', default=None, type=str,
                        help='Resuming model')
    parser.add_argument('--use-coor', default=False, type=bool,
                        help='Whether using coordinate as input')
    parser.add_argument('--use-pos-index', default=False, type=bool,
                        help='Whether using pos index')
    parser.add_argument('--use-reconstruction', default=False, type=bool,
                        help='Whether using reconstruction layer')
    parser.add_argument('--dataset', default='kitti', type=str,
                        help='whether synthetic(syn) / kitti(kitti) / ALOI(aloi)')
    args = parser.parse_args()
    print(args)

    if args.dataset == 'syn':
        tr_ds, val_ds, te_ds = data_loader.make_povray_datasets(
                max_num=args.data_size, norm=args.norm_method)
        # print('train set size: ', tr_ds.img_shape())
        # print('train set fmat size: ', tr_ds.fmat_shape())

    elif args.dataset == 'kitti':
        tr_ds, val_ds, te_ds = data_loader.make_kitti_datasets(
                norm=args.norm_method)
        # print('train set size: ', tr_ds.img_shape())
        # print('train set fmat size: ', tr_ds.fmat_shape())

    elif args.dataset == 'aloi':
        tr_ds, val_ds, te_ds = data_loader.make_aloi_datasets(
                norm=args.norm_method)
        # print('train set size: ', tr_ds.img_shape())
        # print('train set fmat size: ', tr_ds.fmat_shape())

    elif args.dataset == 'mvs':
        tr_ds, val_ds, te_ds = data_loader.make_mvs_datasets(
                norm=args.norm_method)
        # print('train set size: ', tr_ds.img_shape())
        # print('train set fmat size: ', tr_ds.fmat_shape())

    model = SingleFNet(tr_ds, val_ds, te_ds, net=None, lr=args.lr,
                       l1_weight=args.l1_weight, l2_weight=args.l2_weight,
                       ep_weight=args.ep_weight,
                       batch_size=args.batch_size, gradient_clipping=args.use_gc,
                       resume=args.resume, norm_method=args.norm_method,
                       use_coor=args.use_coor, use_idx=args.use_pos_index,
                       use_internal_layer=args.use_reconstruction,
                       prefix="%s_time%d"%(args.prefix, int(time.time())))
    model.train(epoches=args.epoches)

'''
for UCN model:
python .\single_fnet.py --prefix real_all --batch-size 2 --epoches 1000 --use-coor True --use-pos-index True --use-reconstruction True --lr 0.00001
'''
