import numpy as np
import tensorflow as tf
from DataSet import dataSet
import config
import sacqa
import random



#load data
train_graph_path='datasets/train_graph.txt'
text_path='datasets/data.txt'
val_graph_path='datasets/test_graph.txt'
train_y='datasets/train_y.txt'
val_y='datasets/test_y.txt'
val_q='datasets/test_q.txt'

data=dataSet(text_path,train_graph_path,val_graph_path,train_y,val_y,val_q)

# start session

# with tf.Graph().as_default() as  g:
sess=tf.Session()


with sess.as_default():
    model=sacqa.Model(data.num_vocab)
    # for var in model.collection:
    #     print(var.get_shape())
    print(tf.trainable_variables())
    # assert False
    opt=tf.train.AdamOptimizer(config.lr)
    # opt = tf.train.GradientDescentOptimizer(config.lr)
    train_op=opt.minimize(model.loss)
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    saver = tf.train.Saver(max_to_keep=1)
    save_file = './saved_model/model.ckpt'

    #training
    print ('start training.......')
    acc_final = 0
    MAP = 0.0
    MAP_final=0.0
    for epoch in range(config.num_epoch):
        loss_epoch=0
        batches,ys_t=data.generate_batches(mode='train')
        h1=0
        num_batch=len(batches)
        for i in range(num_batch):
            batch=batches[i]
            # print(np.shape(batch))
            # print(batch)
            t1, t2, t3,  = zip(*batch)
            t1, t2, t3 = np.array(t1), np.array(t2), np.array(t3)
            # print(t1)
            text1, text2, text3, ys =  data.text[t1], data.text[t2], data.text[t3],ys_t[i]
            # rs = np.random.normal(loc=0.0, scale=1.0, size=(config.batch_size, 800)).astype(np.float32)
            # ws = np.random.normal(loc=0.0, scale=1.0, size=(config.batch_size, 800)).astype(np.float32)
            # print (text1)
            feed_dict = {
                model.Text_p: text1,
                model.Text_q: text2,
                model.Text_r: text3,
                model.ys:ys
            }

            # run the graph
            _, loss_batch = sess.run([train_op, model.loss], feed_dict=feed_dict)
            # saver.save(sess, save_file, global_step=i+ 1)
            # print('==========')
            # for kk in p_r:
            #     print(kk)
            # print('==========')
            # for ll in prediction_s:
            # #     print (ll)
            # for kks in prediction_ss:
            #     print(kks)
            # print('==========')
            loss_epoch += loss_batch
            # print('epoch: ', epoch + 1,' batch: ', i + 1,  ' loss: ', loss_batch)
        ave_loss= loss_epoch/num_batch
        print ('epoch: ', epoch + 1, ' loss: ', loss_epoch,' average_loss: ',ave_loss)




        val_batches,ys_v,qs_v=data.generate_batches(mode='validation')
        val_num_batch=len(val_batches)
        correct_num = 0
        p_s=[]
        y_s=[]
        q_s=[]
        for i in range(val_num_batch):
            val_batch=val_batches[i]
            ys=ys_v[i]
            qs=qs_v[i]


            t1,t2,t3=zip(*val_batch)
            t1,t2,t3=np.array(t1),np.array(t2),np.array(t3)
            text1,text2,text3=data.text[t1],data.text[t2],data.text[t3]
            feed_dict={
                model.Text_p:text1,
                model.Text_q:text2,
                model.Text_r:text3,
                model.ys: ys
            }

            # run the graph
            prediction = sess.run([model.prediction], feed_dict=feed_dict)
            # print(np.shape(prediction[0]))
            prediction=tf.squeeze(prediction[0])
            # print(np.shape(prediction))
            # print(prediction.eval())
            # prediction=prediction.get_shape().as_list()
            # print(prediction)
            p_s.extend(prediction.eval())
            y_s.extend(ys)
            q_s.extend(qs)


            # pre_max=0
            # pre_y=None
            # for m in range(0,len(prediction)):
            #     if prediction[m] >pre_max:
            #         pre_max=prediction[m]
            #         pre_y=ys_v[m]

            # c_num = 0
            # for m in prediction_gap[0]:
            #     if m == 0:
            #         c_num = c_num + 1
            #
            # correct_num=correct_num+c_num
        # print(p_s)
        pre_q=None
        nn=0
        nns=[]
        for i in q_s:
            if pre_q != i:
                nns.append(nn)
            nn+=1
            pre_q=i
        nns.append(nn)
        nns=nns[1:]
        # print(nns)
        start=0
        stop=nns[0]
        c_tt=0#预测正确 label正确
        c_ft=0#预测错误 label正确
        c_tf=0
        c_ff=0
        qnum = len(nns)
        for jj in nns:
            # print(jj)
            stop=jj
            temp_p=p_s[start:stop]
            # print(temp_p)
            temp_y=y_s[start:stop]
            # print(temp_y)
            pre_max = 0
            pre_y=None
            for m in range(0,len(temp_p)):
                # print(temp_p[m])
                if temp_p[m] >pre_max:
                    pre_max=temp_p[m]
                    pre_y=temp_y[m]
                    # print(pre_y)
                if temp_p[m]>0.5 and temp_y[m]>0:
                    c_tt+=1
                if temp_p[m]<0.5 and temp_y[m]>0:
                    c_ft+=1
                if temp_p[m] > 0.5 and temp_y[m] < 1:
                    c_tf+=1
                if temp_p[m] < 0.5 and temp_y[m] < 1:
                    c_ff+=1
            if pre_y>0:
                correct_num+=1

            for i in range(len(temp_p) - 1):  # 这个循环负责设置冒泡排序进行的次数
                for j in range(len(temp_p) - i - 1):  # ｊ为列表下标
                    if temp_p[j] < temp_p[j + 1]:
                        temp_p[j], temp_p[j + 1] = temp_p[j + 1], temp_p[j]
                        temp_y[j], temp_y[j + 1] = temp_y[j + 1], temp_y[j]
            avg_prec = 0
            precisions = []
            num_correct = 0
            for i in range(0, len(temp_y)):
                if temp_y[i] > 0:
                    num_correct += 1
                    precisions.append(num_correct / (i + 1))
            if precisions:

                avg_prec = sum(precisions) / len(precisions)
            MAP += avg_prec
            start=jj
        acc=correct_num*1.0/len(nns)

        nn_c=c_tt+c_ft+c_tf+c_ff
        Acc1 = (c_tt + c_ff) / nn_c
        MAP = MAP / qnum


        # if Acc1 > acc_final:
        #     acc_final=Acc1
        #     saver.save(sess, save_file, global_step=epoch + 1)
        if MAP > MAP_final:
            MAP_final=MAP
            saver.save(sess, save_file, global_step=epoch + 1)
        if MAP==MAP_final and Acc1>acc_final:
            acc_final = Acc1
            saver.save(sess, save_file, global_step=epoch + 1)



        print ('epoch: ',epoch+1,' MAP: ',MAP_final,' acc: ',acc,' standard acc: ',Acc1,'no of answer and all(c_tf..)is ',len(p_s),nn_c)






