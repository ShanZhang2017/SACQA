import numpy as np
import tensorflow as tf
from DataSet_demo import dataSet
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
demo_graph_path="datasets/demo_graph.txt"
demo_y='datasets/demo_y.txt'

data=dataSet(text_path,train_graph_path,val_graph_path,train_y,val_y,val_q,demo_graph_path,demo_y)
module_file =  tf.train.latest_checkpoint('F://PycharmProjects/SACQA/saved_model/')
# start session

with tf.Graph().as_default() as  g:
    sess=tf.Session()


    with sess.as_default():
        model=sacqa.Model(data.num_vocab)
        opt=tf.train.AdamOptimizer(config.lr)
        # opt = tf.train.GradientDescentOptimizer(config.lr)
        train_op=opt.minimize(model.loss)
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        saver = tf.train.Saver(max_to_keep=1)
        # save_file = './saved_model/model.ckpt'
        if module_file is not None:
            saver.restore(sess, module_file)

        #training
        print ('start testing.......')
        acc_final = 0.0
        MAP=0.0
        val_batches,ys_v,num,ans=data.generate_batches(mode='demo')
        val_num_batch=len(val_batches)
        correct_num = 0
        p_s=[]
        y_s=[]
        ans_s=[]
        # q_s=[]
        for i in range(val_num_batch):
            val_batch=val_batches[i]
            ys=ys_v[i]
            an=ans[i]
            # qs=qs_v[i]


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
            ans_s.extend(an)
            # q_s.extend(qs)


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
        # nns=[]
        # for i in q_s:
        #     if pre_q != i:
        #         nns.append(nn)
        #     nn+=1
        #     pre_q=i
        # nns.append(nn)
        # nns=nns[1:-3]
        # print(nns)
        start=0
        stop=num-1
        c_tt=0#预测正确 label正确
        c_ft=0#预测错误 label正确
        c_tf=0
        c_ff=0
        # qnum=len(nns)
        # print(qnum)
        # for jj in nns:
        # print(jj)
        # stop=jj
        temp_p=p_s[start:stop]
        # print(temp_p)
        temp_y=y_s[start:stop]
        temp_ans=ans_s[start:stop]
        # print(temp_y)
        pre_max = 0
        pre_y=None
        py = list(zip(temp_p,temp_y,temp_ans))
        #calculate acc
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

        #calculate MAP
        # for i in range(len(py) - 1):  # 这个循环负责设置冒泡排序进行的次数
        #     for j in range(len(py) - i - 1):  # ｊ为列表下标
        #         if py[j] > py[j + 1]:
        #             py[j], py[j + 1] = py[j + 1], py[j]
        # temp_p[:], temp_y[:] = zip(*py)
        for i in range(len(temp_p) - 1):  # 这个循环负责设置冒泡排序进行的次数
            for j in range(len(temp_p) - i - 1):  # ｊ为列表下标
                if temp_p[j] < temp_p[j + 1]:
                    temp_p[j], temp_p[j + 1] = temp_p[j + 1], temp_p[j]
                    temp_y[j], temp_y[j + 1] = temp_y[j + 1], temp_y[j]
                    temp_ans[j], temp_ans[j + 1] = temp_ans[j + 1], temp_ans[j]
        print(temp_y)
        print(temp_p)
        print(temp_ans)

        # print(temp_p)
        # avg_prec=0
        # precisions=[]
        # num_correct=0
        # for i in range(0,len(temp_y)):
        #     if temp_y[i]>0:
        #         num_correct+=1
        #         precisions.append(num_correct/(i+1))
        # if precisions:
        #
        #     avg_prec=sum(precisions)/len(precisions)
        # MAP +=avg_prec
        # print(MAP)

        # start=jj
        # acc=correct_num*1.0/len(nns)

        # nn_c=c_tt+c_ft+c_tf+c_ff
        # Acc1 = (c_tt + c_ff) / nn_c
        # MAP_final=MAP/qnum






        # print (' MAP: ',MAP_final,' acc: ',acc,' standard acc: ',Acc1,'no of answer and all(c_tf..)is ',len(p_s),nn_c)






