from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
from ReadData import *
import time, sys
import random

# CNN with residual followed by parallel inception
class CNNBLSTM:
    def __init__(self, nb_features, nb_classes, modelname):
        self.hybridmodel = tf.Graph()
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.model_name = modelname
        print("Empty Graph Created")

    def get_layer_shape(self, layer):
        thisshape = tf.Tensor.get_shape(layer)
        ts = [thisshape[i].value for i in range(len(thisshape))]
        return ts

    def readNetworkStructure(self, configfile):
        nw = {}
        f = open(configfile)
        line = f.readline()
        while line:
            info = line.strip("\n").split(",")
            nw[info[0]] = info[1]
            line = f.readline()
        self.filterwidth = int(nw['filterwidth'])
        self.nb_filters = [int(fi) for fi in nw['nb_filters'].split()]
        self.conv_stride = [int(fi) for fi in nw['conv_stride'].split()]
        self.pool_stride = [int(fi) for fi in nw['pool_stride'].split()]
        self.nb_hidden = int(nw['nb_hidden'])
        self.lr = float(nw['lr'])
        self.maxh = int(nw['maxh'])
        self.maxw = int(nw['maxw'])
        self.max_timesteps = self.maxw
        self.filterheight = self.maxh
        print('Network Configuration Understood')

    def createNetwork(self, configfile):
        self.readNetworkStructure(configfile)
        with self.hybridmodel.as_default():
            self.network_input_x = tf.placeholder(tf.float32, [None, self.maxh, self.maxw, self.nb_features])
            self.network_target_y = tf.sparse_placeholder(tf.int32)
            self.network_input_sequence_length = tf.placeholder(tf.int32, [None])

            self.network_input_x=tf.divide(self.network_input_x,255.0)

            #pre filtering
            nbfilters = 4
            f = [self.filterwidth, self.filterwidth, self.nb_features, nbfilters]
            W = tf.Variable(tf.truncated_normal(f, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[nbfilters]), name="b")
            conv = tf.nn.conv2d(self.network_input_x, W, strides=[1, 1, 1, 1], padding='SAME',
                                  name='pf_W')
            nl = tf.nn.relu(tf.add(conv, b))  # batchsize,1,maxsteps,filter2
            shape = self.get_layer_shape(nl)
            print("Prefilter Block = ", shape)
            #pre filtering ends
            bn_nl = tf.layers.batch_normalization(nl, training=True)
            nl=tf.add(self.network_input_x,bn_nl)
            shape = self.get_layer_shape(nl)
            print("Prefilter+Input Block = ", shape)

            # Inception block starts
            in_dim=shape[-1]
            nbfilters = 8
            f0 = [3, 3, in_dim, nbfilters]
            W0 = tf.Variable(tf.truncated_normal(f0, stddev=0.1), name="W0")
            b0 = tf.Variable(tf.constant(0.1, shape=[nbfilters]), name="b2")
            conv_0 = tf.nn.conv2d(nl, W0, strides=[1, 1, 1, 1], padding='SAME', name='In_W')
            nl_0 = bn_nl = tf.layers.batch_normalization(tf.nn.relu(tf.add(conv_0, b0)),training=True)  # batchsize,1,maxsteps,filter2
            shape = self.get_layer_shape(nl_0)
            print("Resolution W Block = ", shape)

            # resolution=H x 2Fw
            f1 = [5, 5, in_dim, nbfilters]
            W1 = tf.Variable(tf.truncated_normal(f1, stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.1, shape=[nbfilters]), name="b1")
            conv_1 = tf.nn.conv2d(nl, W1, strides=[1, 1, 1, 1], padding='SAME',name='In_2W')
            nl_1 = tf.layers.batch_normalization(tf.nn.relu(tf.add(conv_1, b1)),training=True)  # batchsize,1,maxsteps,filter1
            shape = self.get_layer_shape(nl_1)
            print("Resolution 2W Block = ", shape)

            # resolution=H x 4Fw
            f1_double = [7, 7, in_dim, nbfilters]
            W1_double = tf.Variable(tf.truncated_normal(f1_double, stddev=0.1), name="W1_double")
            b1_double = tf.Variable(tf.constant(0.1, shape=[nbfilters]), name="b1_double")
            conv_1_double = tf.nn.conv2d(nl, W1_double, strides=[1, 1, 1, 1], padding='SAME')
            nl_2 = tf.layers.batch_normalization(tf.nn.relu(tf.add(conv_1_double, b1_double)),training=True)
            shape = self.get_layer_shape(nl_2)
            print("Resolution 4W Block = ", shape)

            merge_conv1 = tf.concat([nl_0, nl_1, nl_2], 3)
            shape = self.get_layer_shape(merge_conv1)

            # Inception block ends
            print("First Inception Block = ", shape)


            # ---------------3rd Block Starts NORMAL CNN Block---------------------#
            nbfilters=32
            f3 = [self.filterheight, self.filterwidth, shape[-1], nbfilters]
            W3 = tf.Variable(tf.truncated_normal(f3, stddev=0.1), name="W3")
            b3 = tf.Variable(tf.constant(0.1, shape=[nbfilters]), name="b3")
            conv_3 = tf.nn.conv2d(merge_conv1, W3, strides=[1, self.filterheight, 2, 1], padding='SAME', name="3rd_CNN_block")
            nl_3 = tf.layers.batch_normalization(tf.nn.relu(tf.add(conv_3, b3)),training=True)   # batchsize,1,maxsteps,filter2
            # mp_3 = tf.nn.max_pool(nl_3, ksize=[1, 1, , 1], strides=[1, 1, 1, 1],padding='SAME')  # batchsize,1,maxsteps,filter3
            shape = self.get_layer_shape(nl_3)
            print("3rd Conv Block = ", shape)
            # --------------3rd Conv MP Block Ends-----------------------#


            # ---------------4th Block Starts NORMAL CNN Block---------------------#
            nbfilters = 64
            f4 = [1, 3, shape[-1], nbfilters]
            W4 = tf.Variable(tf.truncated_normal(f4, stddev=0.1), name="W3")
            b4 = tf.Variable(tf.constant(0.1, shape=[nbfilters]), name="b3")
            conv_4 = tf.nn.conv2d(nl_3, W4, strides=[1, 1, 2, 1], padding='SAME', name="4th_CNN_block")
            nl_4 = tf.layers.batch_normalization(tf.nn.relu(tf.add(conv_4, b4)),training=True)  # batchsize,1,maxsteps,filter2
            # mp_3 = tf.nn.max_pool(nl_3, ksize=[1, 1, , 1], strides=[1, 1, 1, 1],padding='SAME')  # batchsize,1,maxsteps,filter3
            shape = self.get_layer_shape(nl_4)
            print("4th Conv Block = ", shape)
            # --------------4th Conv MP Block Ends-----------------------#


            conv_reshape = tf.squeeze(nl_4, squeeze_dims=[1])  # batchsize,maxsteps,filter3
            shape = self.get_layer_shape(conv_reshape)
            print("CNN --> RNN Reshape = ", shape)

            self.max_time=shape[1]
            print("Max time for RNN is ",self.max_time)
            with tf.variable_scope("cell_def_1"):
                f_cell = tf.nn.rnn_cell.LSTMCell(self.nb_hidden, state_is_tuple=True)
                b_cell = tf.nn.rnn_cell.LSTMCell(self.nb_hidden, state_is_tuple=True)

            with tf.variable_scope("cell_op_1"):
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, conv_reshape,dtype=tf.float32)

            merge = tf.concat(outputs, 2)
            shape = self.get_layer_shape(merge)
            print("First BLSTM = ", shape)

            nb_hidden_2 = self.nb_hidden * 2

            with tf.variable_scope("cell_def_2"):
                f1_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden_2, state_is_tuple=True)
                b1_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden_2, state_is_tuple=True)

            with tf.variable_scope("cell_op_2"):
                outputs2, _ = tf.nn.bidirectional_dynamic_rnn(f1_cell, b1_cell, merge,dtype=tf.float32)

            merge2 = tf.concat(outputs2, 2)
            shape = self.get_layer_shape(merge2)
            print("Second BLSTM = ", shape)
            batch_s, timesteps = shape[0], shape[1]
            print(timesteps)

            blstm_features = shape[-1]

            output_reshape = tf.reshape(merge2, [-1, blstm_features])  # maxsteps*batchsize,nb_hidden
            shape = self.get_layer_shape(output_reshape)
            print("RNN Time Squeezed = ", shape)

            W = tf.Variable(tf.truncated_normal([blstm_features, self.nb_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0., shape=[self.nb_classes]), name="b")

            logits = tf.matmul(output_reshape, W) + b  # maxsteps*batchsize,nb_classes
            logits = tf.reshape(logits, [-1, timesteps, self.nb_classes])
            shape = self.get_layer_shape(logits)
            print("Logits = ", shape)

            logits_reshape = tf.transpose(logits, [1, 0, 2])  # maxsteps,batchsize,nb_classes
            shape = self.get_layer_shape(logits_reshape)
            print("RNN Time Distributed (Time Major) = ", shape)

            loss = tf.nn.ctc_loss(self.network_target_y, logits_reshape,self.network_input_sequence_length,ignore_longer_outputs_than_inputs=True)
            self.cost = tf.reduce_mean(loss)

            self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.cost)

            decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_reshape, self.network_input_sequence_length)

            self.decoded_words = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values)
            self.actual_targets = tf.sparse_to_dense(self.network_target_y.indices, self.network_target_y.dense_shape,
                                                     self.network_target_y.values)

            actual_ed = tf.edit_distance(tf.cast(decoded[0], tf.int32), self.network_target_y, normalize=False)
            self.ler = tf.reduce_sum(actual_ed)  # insertion+deletion+substitution
            self.new_saver = tf.train.Saver()
            print("Network Ready")

    def trainNetwork(self, nb_epochs, batchsize, x, y, transcription_length, weightfiles,
                     mode):
        x_train = x[0]  # Subset of data
        y_train = y[0]
        x_test = x[1]
        y_test = y[1]
        weightfile_last = weightfiles[0]
        weightfile_best = weightfiles[1]
        train_transcription_length = transcription_length[0]
        test_transcription_length = transcription_length[1]
        gpu_memory = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
        with tf.Session(graph=self.hybridmodel, config=tf.ConfigProto(gpu_options=gpu_memory)) as session:
            if (mode == "New"):
                init_op = tf.global_variables_initializer()
                session.run(init_op)
                print("New Weights Initiated")
            elif (mode == "Load"):
                self.new_saver.restore(session, weightfile_best)
                print("Previous weights loaded")
            else:
                print("Unknown Mode")
                return

            nb_train = len(x_train)  # Subset
            nb_test = len(x_test)
            trainbatch = int(np.ceil(float(nb_train) / batchsize))
            testbatch = int(np.ceil(float(nb_test) / batchsize))
            besttestacc = 0
            plt.ion()
            figure = plt.figure()
            figure.tight_layout()
            for e in range(nb_epochs):
                totalloss = 0
                totalacc = 0
                starttime = time.time()
                train_batch_start = 0
                logf = open("Training_log", "a")

                for b in range(trainbatch):
                    train_batch_end = min(nb_train, train_batch_start + batchsize)
                    sys.stdout.write("\rTraining Batch %d / %d" % (b, trainbatch))
                    sys.stdout.flush()
                    batch_x,batch_seq_len,_ = batch_rescale_image(x_train[train_batch_start:train_batch_end], 1250, 50)
                    batch_x = np.expand_dims(batch_x, -1)
                    batch_seq_len=adjustSequencelengths(batch_seq_len,4,self.max_time)
                    batch_target_sparse = y_train[b]

                    feed = {self.network_input_x: batch_x, self.network_target_y: batch_target_sparse,self.network_input_sequence_length:batch_seq_len}

                    batchloss, batchacc, _ = session.run([self.cost, self.ler, self.optimizer],
                                                                    feed)

                    totalloss = totalloss + batchloss
                    totalacc = totalacc + batchacc
                    train_batch_start = train_batch_end

                trainloss = totalloss / trainbatch

                print("\nTraining Edit Distance ", totalacc, "/", train_transcription_length)
                trainacc = (1 - (float(totalacc) / train_transcription_length)) * 100
                # Now save the model
                self.new_saver.save(session, weightfile_last)

                testloss = 0
                testacc = 0

                test_batch_start = 0
                output_words = []
                target_words = []
                for b in range(testbatch):
                    test_batch_end = min(nb_test, test_batch_start + batchsize)
                    sys.stdout.write("\rTesting Batch %d/%d" % (b, testbatch))
                    sys.stdout.flush()
                    batch_x,batch_seq_len,_ = batch_rescale_image(x_test[test_batch_start:test_batch_end], 1250, 50)
                    batch_x=np.expand_dims(batch_x,-1)
                    batch_seq_len = adjustSequencelengths(batch_seq_len, 4,self.max_time)
                    #batch_seq_len = seq_len_test[test_batch_start:test_batch_end]
                    batch_target_sparse = y_test[b]

                    testfeed = {self.network_input_x: batch_x, self.network_target_y: batch_target_sparse,self.network_input_sequence_length:batch_seq_len}

                    try:
                        batchloss, batchacc, output_words_batch, target_words_batch = session.run(
                            [self.cost, self.ler, self.decoded_words, self.actual_targets], testfeed)
                    except:
                        pass

                    output_words.extend(output_words_batch)
                    target_words.extend(target_words_batch)
                    testloss = testloss + batchloss
                    testacc = testacc + batchacc
                    test_batch_start = test_batch_end

                testloss = testloss / testbatch
                testacc = (1 - (float(testacc) / test_transcription_length)) * 100

                result = open("Decoded", "w")
                corrects = 0.0
                for w in range(nb_test):
                    result.write(str(target_words[w]) + "," + str(output_words[w]) + "\n")
                    if (len(output_words[w]) >= len(target_words[w])):
                        flag = False
                        for c in range(len(target_words[w])):
                            if (output_words[w][c] == target_words[w][c]):
                                flag = True
                            else:
                                flag = False
                                break
                        if (flag == True):
                            corrects = corrects + 1
                result.close()

                if (testacc > besttestacc):
                    besttestacc = testacc
                    print("\nNetwork Improvement")
                    self.new_saver.save(session, weightfile_best)
                endtime = time.time()
                timetaken = endtime - starttime
                msg = "\nEpoch " + str(e) + "(" + str(timetaken) + " sec) Training: Loss is " + str(
                    trainloss) + " Accuracy " + str(trainacc) + "% Testing: Loss " + str(testloss) + " Accuracy " + str(
                    testacc) + "% Best " + str(besttestacc) + "%\n"
                print(msg)
                logf.write(msg)
                logf.close()

    def predict(self, input_data, weightfile, dbfile, sampleids,total_target_len,testbatchsize):
        tempf=open('SampleIDS.txt','w')
        for s in sampleids:
            tempf.write(s+"\n")
        tempf.close()
        nb_predicts=len(input_data[0])
        target_y = input_data[1]
        print("Number of test cases ", nb_predicts)
        nb_batches=int(np.ceil(nb_predicts/float(testbatchsize)))
        gpu_memory = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
        with tf.Session(graph=self.hybridmodel, config=tf.ConfigProto(gpu_options=gpu_memory)) as predict_session:
            self.new_saver.restore(predict_session, weightfile)
            print("Saved Model Loaded")
            start=0
            f = open("Predicted.txt", "w")
            predicted_words = []
            sampleinds = 0
            acc=0
            for b in range(nb_batches):
                end=min(start+testbatchsize,nb_predicts-1)
                batch_x, batch_seq_len, _ = batch_rescale_image(input_data[0][start:end], 1250, 50)
                batch_x = np.expand_dims(batch_x, -1)
                batch_seq_len = adjustSequencelengths(batch_seq_len, 4, self.max_time)
                feed = {self.network_input_x: batch_x,self.network_input_sequence_length: batch_seq_len, self.network_target_y: target_y[b]}
                actual_words, output_words,ler = predict_session.run([self.actual_targets, self.decoded_words,self.ler], feed)
                acc+=ler
                total = len(output_words)
                print("Reading batch from %d to %d"%(start,end))
                for w in range(total):
                    word = output_words[w]
                    unicode_output, _ = int_to_bangla(word, "Character_Integer", dbfile)
                    unicode_output_p = reset_unicode_order(unicode_output, charposfile).encode('utf-8').replace("***", "")

                    word = actual_words[w]
                    unicode_output, _ = int_to_bangla(word, "Character_Integer", dbfile)
                    unicode_output_t = reset_unicode_order(unicode_output, charposfile).encode('utf-8').replace("***", "")
                    total_target_characters=len(unicode_output_t)
                    e=ed.eval(unicode_output_t,unicode_output_p)/float(total_target_characters)
                    tc,nbwords,avs=evaluate_word_accuracy(unicode_output_t,unicode_output_p)
                    wer=tc/float(nbwords)
                    f.write("%s,%s,%s,%0.2f,%0.2f,%s\n"%(sampleids[sampleinds],unicode_output_t,unicode_output_p,e,wer,avs))
                    sampleinds+=1
                start=end
            acc=(1-(acc/float(total_target_len)))*100
            f.close()
            print("Output Ready, transcription length %d accuracy %f"%(total_target_len,acc))

            return [predicted_words]


'''
Model should be fed with
Max_Time_steps,nb_features,nb_classes
Rest of the parameters written in Config file
'''


def main(task, mode, dbfile, files, weightfile, batchsize):
    generate_char_table = True
    if (mode == "Load"):
        generate_char_table = False
    if (task == "Train"):
        [x_train, x_test], nb_classes, seqlen, [train_y, test_y], max_target_length, \
        char_int, transcription_length, sampleids = load_data(
            files[0], files[1], batchsize, 100, generate_char_table)
        nb_features = 1
        x = [x_train, x_test]
        y = [train_y, test_y]

        print("Training Data X=", len(x_train), " Testing Data X=", len(x_test))
        print("Number of classes (including blank)", nb_classes)

        model = CNNBLSTM(nb_features, nb_classes, "Hybrid_crp")
        model.createNetwork("Config")

        weightfile_last = weightfile + "/residin_last"
        weightfile_best = weightfile + "/Best/residin_best"
        weightfiles = [weightfile_last, weightfile_best]
        # train network
        model.trainNetwork(10000, batchsize, x, y, transcription_length, weightfiles, mode)
    elif (task == "Predict"):
        # test network
        model = CNNBLSTM(1, 346, "Hybrid")
        model.createNetwork("Config")
        weightfile_best = weightfile + "/Best/residin_best"
        all_x, all_y, seq_lengths, all_sampleid,total_target_len=load_test_data(files[1],128)
        model.predict([all_x, all_y], weightfile_best, dbfile, all_sampleid,total_target_len,128)

dict_cmpd_file = "Dict/BanglaCompositeMap.txt"
dict_single_file = "Dict/AllCharcaters.txt"
dict_file = "Dict/CompositeAndSingleCharacters.txt"
data_file = "Data_main"
Encoded_gt = "Encoded_gt.txt"

train_path = sys.argv[1] 
test_path = sys.argv[2] 
load_weight = sys.argv[3] #Use New for creating a network , Load to load previously trained weights
task = sys.argv[4] # Use Predict for generating output, Train to train the network

dbfile = "Dict/CompositeAndSingleCharacters.txt"
charposfile = "Dict/bengalichardb.txt"
files = [train_path, test_path]
weightfile = "Weights"
batchsize = 64


if(task=='Predict'):
    batchsize=3000

main(task, load_weight, dbfile, files, weightfile, batchsize)
