from sklearn.metrics import auc,precision_score, recall_score, confusion_matrix,f1_score, roc_auc_score, average_precision_score,precision_recall_curve
import numpy as np
import logging,os
from tensorflow.keras import layers,models
import tensorflow as tf
import pandas as pd
import argparse
import pickle
import random
import math
from random import choice
from tensorflow.keras import Sequential
import utilize
def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument('--train_directory', type=str,
                        default="data/",
                        help='Directory of train data ')    # Indication_CUI_train_eval
    parser.add_argument('--train_filename', type=str, default="sider4_train_OneDrug_Sides_CUIs_Sider_threshld3_1757.csv",
                        help='Filename of the train data')  #
    parser.add_argument('--train_filename_assist', type=str, default="label_indication.pkl",
                        help='Filename of the train data')
    parser.add_argument('--train_directory_embedding', type=str,
                        default="data/",
                        help='Directory of train data ')
    parser.add_argument('--train_filename_embedding', type=str, default="VA_DB_embedding.pkl",
                        help='Filename of the train data')
    parser.add_argument('--test_directory', type=str,
                        default="data/",
                        help='Directory of test data ')     
    parser.add_argument('--test_filename', type=str, default="sider4_test_OneDrug_Sides_CUIs_Sider_threshld3_1757.csv",
                        help='Filename of the test data')  
    parser.add_argument('--save_directory', type=str,
                        default="result_metric/",
                        help='Directory to save the results')
    parser.add_argument('--results_filename', type=str, default="SIDER4.txt",
                        help='Filename to save the result data')  
    parser.add_argument('--results_filename_csv', type=str, default="SIDER4.csv",
                        help='Filename to save the result data')
    parser.add_argument('--comments_save', type=str,
                        default="_full_model_True negative ",
                        help='comments_save ')
    parser.add_argument('--fusion_mode', type=str, default="CUI_fusion",
                        help='fusion_mode ')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=95,
                        help='training epoches')
    parser.add_argument('--length_SMILE', type=int, default=120,
                        help='max length of moleculeSMILE ')
    parser.add_argument('--dim_emb', type=int, default=100,
                        help='dim_emb')
    parser.add_argument('--num_out', type=int, default=1,
                        help='num_out')
    parser.add_argument('--epoches', type=int, default=1,
                        help='epoches to train')
    parser.add_argument('--wight_embedding', type=float, default=0.5,
                        help='wights for learning the embedding')
    parser.add_argument('--weight_assist', type=float, default=0.8,
                        help='weight for the assist task ')
    parser.add_argument('--weight_emb_gan', type=float, default=0.2,
                        help='weight of GAN for invariant concept embeddding')
    parser.add_argument('--weight_pair_gan', type=float, default=0.05,
                        help='weight of GAN for invariant pair representation ')
    parser.add_argument('--dim_token', type=int, default=16,
                        help='dim for learning token embedding')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning_rate ')
    parser.add_argument('--file_ID_token', type=str, default="dic_dbID_CID_SMILE_tokens.pkl",
                        help='file containing the mapping from Drugbank ID to token list')
    parser.add_argument('--file_embedding_disease', type=str, default="embedding_diseases.csv",
                        help='file containing disease EHR embedddings')
    parser.add_argument('--file_embedding_drugs', type=str, default="embedding_drugs.csv",
                        help='file containing drug EHR embedddings')
    parser.add_argument('--file_ID_token', type=str, default="dic_dbID_CID_SMILE_tokens.pkl",
                        help='file containing drug EHR embedddings')
    parser.add_argument('--size_token', type=int, default=62,
                        help='size of the total tokens')
    parser.add_argument('--file_emb_all', type=str, default="embedding_all.csv",
                        help='file containing the mapping from Drugbank ID to token list')
    parser.add_argument('--file_emb_phecode', type=str, default="embedding_phecodes.csv",
                        help='file containing the mapping from Drugbank ID to token list')
    parser.add_argument('--', type=str,
                        default="model_SIDER4",
                        help='the name for the model to be saved')
    args = parser.parse_args()
    return args

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_metric = tf.keras.metrics.AUC(name='train_auc')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_metric = tf.keras.metrics.AUC(name='test_auc')
train_loss_gan_embedding = tf.keras.metrics.Mean(name='train_loss_gan_embedding')
train_loss_gan_fusion = tf.keras.metrics.Mean(name='train_loss_gan_fusion')
test_loss_gan_embedding = tf.keras.metrics.Mean(name='test_loss_gan_embedding')
test_loss_gan_fusion = tf.keras.metrics.Mean(name='test_loss_gan_fusion')
train_loss_distance = tf.keras.metrics.Mean(name='train_loss_distance')
test_loss_distance = tf.keras.metrics.Mean(name='test_loss_distance')

if True:
    PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ARGS = parse_arguments(PARSER)
    train_directory = ARGS.train_directory
    train_filename = ARGS.train_filename
    train_directory_embedding = ARGS.train_directory_embedding
    train_filename_embedding = ARGS.train_filename_embedding
    test_directory = ARGS.test_directory
    test_filename = ARGS.test_filename
    save_directory = ARGS.save_directory
    results_filename = ARGS.results_filename
    results_filename_csv = ARGS.results_filename_csv
    batch_size = ARGS.batch_size
    fusion_mode = ARGS.fusion_mode
    train_epoches = ARGS.epochs
    length_SMILE = ARGS.length_SMILE
    num_out = ARGS.num_out
    dim_emb = ARGS.dim_emb
    comments_save = ARGS.comments_sav
    train_filename_assist = ARGS.train_filename_assist
    wight_embedding = ARGS.wight_embedding
    weight_assist = ARGS.weight_assist
    weight_emb_gan = ARGS.weight_emb_gan
    weight_pair_gan = ARGS.weight_pair_gan
    dim_token = ARGS.dim_token
    learning_rate = ARGS.learning_rate
    size_token = ARGS.size_token
    file_embedding_disease =ARGS.file_embedding_disease
    file_embedding_drugs =ARGS.file_embedding_drugs
    file_ID_token =ARGS.file_ID_token

    cosine_T = ARGS.cosine_T
    train_ratio = ARGS.train_ratio


    flag_gan_progress = True
    norm_gan = 10
    norm_gan_discriniator = 50
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_dic = tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.8)

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    with open(train_directory + ARGS.file_ID_token, 'rb') as fid:
        (dic_charactor, dic_ID_SMILES) = pickle.load(fid)

    #############################loadding all embeddings
    df = pd.read_csv(train_directory + ARGS.file_emb_all)
    embeddings_ALL = np.array(df[df.columns[2:2 + dim_emb]] * 1.0)
    df = pd.read_csv(train_directory + ARGS.file_emb_phecode)
    embeddings_ALL_phecodes = np.array(df[df.columns[2:2 + dim_emb]] * 1.0)
    #############################loadding all embeddings


    drug_features_1_SMILES_train, ID_ID_train, drug_labels_train, embedding_CUI_train, label_name_train, \
    drug_features_1_SMILES_test, ID_ID_test, drug_labels_test, embedding_CUI_test, label_name_test=\
        utilize.loaddata(train_directory,train_filename,test_filename,
                          file_embedding_disease,file_embedding_drugs,
                          file_ID_token,train_filename_embedding,cosine_T,train_ratio)

    with open(train_directory + train_filename_assist, 'rb') as fid:
        (drug_features_1_SMILES_train_add, ID_ID_train_add, drug_labels_train_add, embedding_CUI_train_add,
         label_name_train_add) \
            = pickle.load(fid)

    with open(train_directory_embedding + train_filename_embedding, 'rb') as fid:
        (drug_SMILES_embedding, feature_embedding) = pickle.load(fid)
    total_train_num = max(len(drug_features_1_SMILES_train), len(drug_features_1_SMILES_train_add))

    if len(drug_features_1_SMILES_train) < total_train_num:
        drug_features_1_SMILES_train = list(drug_features_1_SMILES_train)
        ID_ID_train = list(ID_ID_train)
        drug_labels_train = list(drug_labels_train)
        embedding_CUI_train = list(embedding_CUI_train)
        label_name_train = list(label_name_train)
        valid_num = len(drug_features_1_SMILES_train)
        for i in range(total_train_num - valid_num):
            temp = random.randint(0, valid_num - 1)
            drug_features_1_SMILES_train.append(drug_features_1_SMILES_train[temp])
            ID_ID_train.append(ID_ID_train[temp])
            drug_labels_train.append(drug_labels_train[temp])
            embedding_CUI_train.append(embedding_CUI_train[temp])
            label_name_train.append(label_name_train[temp])
    else:
        drug_features_1_SMILES_train_add = list(drug_features_1_SMILES_train_add)
        ID_ID_train_add = list(ID_ID_train_add)
        drug_labels_train_add = list(drug_labels_train_add)
        embedding_CUI_train_add = list(embedding_CUI_train_add)
        label_name_train_add = list(label_name_train_add)
        valid_num = len(drug_features_1_SMILES_train_add)
        for i in range(total_train_num - valid_num):
            temp = random.randint(0, valid_num - 1)
            drug_features_1_SMILES_train_add.append(drug_features_1_SMILES_train_add[temp])
            ID_ID_train_add.append(ID_ID_train_add[temp])
            drug_labels_train_add.append(drug_labels_train_add[temp])
            embedding_CUI_train_add.append(embedding_CUI_train_add[temp])
            label_name_train_add.append(label_name_train_add[temp])

    SMILE_ALL_unlabel_train = []
    smilelist = list(dic_ID_SMILES.keys())
    for i in range(total_train_num):
        smilei = choice(smilelist)
        SMILE_ALL_unlabel_train.append(dic_ID_SMILES[smilei])
    drug_features_1_SMILES_train = tf.keras.preprocessing.sequence.pad_sequences(drug_features_1_SMILES_train,
                                                                                 maxlen=length_SMILE,
                                                                                 padding="post", value=0,
                                                                                 truncating="post")
    drug_features_1_SMILES_train_add = tf.keras.preprocessing.sequence.pad_sequences(drug_features_1_SMILES_train_add,
                                                                                     maxlen=length_SMILE,
                                                                                     padding="post", value=0,
                                                                                     truncating="post")
    drug_features_1_SMILES_test = tf.keras.preprocessing.sequence.pad_sequences(drug_features_1_SMILES_test,
                                                                                maxlen=length_SMILE,
                                                                                padding="post", value=0,
                                                                                truncating="post")

    drug_SMILES_embedding = tf.keras.preprocessing.sequence.pad_sequences(drug_SMILES_embedding,
                                                                          maxlen=length_SMILE,
                                                                          padding="post", value=0,
                                                                          truncating="post")

    SMILE_ALL_unlabel_train = tf.keras.preprocessing.sequence.pad_sequences(SMILE_ALL_unlabel_train,
                                                                            maxlen=length_SMILE,
                                                                            padding="post", value=0,
                                                                            truncating="post")
    drug_features_1_SMILES_train = np.array(drug_features_1_SMILES_train)
    ID_ID_train = np.array(ID_ID_train)
    drug_labels_train = np.array(drug_labels_train)
    embedding_CUI_train = np.array(embedding_CUI_train)
    label_name_train = np.array(label_name_train)

    drug_features_1_SMILES_train_add = np.array(drug_features_1_SMILES_train_add)
    ID_ID_train_add = np.array(ID_ID_train_add)
    drug_labels_train_add = np.array(drug_labels_train_add)
    embedding_CUI_train_add = np.array(embedding_CUI_train_add)
    label_name_train_add = np.array(label_name_train_add)

    ###############################select some drugs to learn embedding to explore if GAN help mapping
    test_num_at_least = 10
    drug_SMILES_embedding = list(drug_SMILES_embedding)
    feature_embedding = list(feature_embedding)
    # drug_SMILES_embedding, feature_embedding = shuffle(drug_SMILES_embedding, feature_embedding, random_state=0)

    mapping_train_num = int(len(drug_SMILES_embedding))
    mapping_test_num = max(len(drug_SMILES_embedding) - mapping_train_num, test_num_at_least)
    SMILE_mapping = list(drug_SMILES_embedding)
    embedding_mapping = list(feature_embedding)
    valid_num = len(SMILE_mapping)
    for i in range(total_train_num - valid_num):
        temp = random.randint(0, mapping_train_num - 1)
        SMILE_mapping.append(drug_SMILES_embedding[temp])
        embedding_mapping.append(feature_embedding[temp])
    SMILE_mapping = np.array(SMILE_mapping)
    embedding_mapping = np.array(embedding_mapping)
    embedding_mapping_train = np.array(embedding_mapping)

    SMILE_mapping_test = []
    embedding_mapping_test = []
    test_index_begin = min(len(drug_SMILES_embedding) - mapping_train_num,
                           len(drug_SMILES_embedding) - test_num_at_least)
    for i in range(len(drug_labels_test)):
        temp = random.randint(test_index_begin, len(drug_SMILES_embedding) - 1)
        SMILE_mapping_test.append(drug_SMILES_embedding[temp])
        embedding_mapping_test.append(feature_embedding[temp])

    #####################embedding for unsupervised###################
    embeddings_ALL_train = list(embeddings_ALL)
    valid_num = len(embeddings_ALL)
    for i in range(total_train_num - valid_num):
        temp = random.randint(0, len(embeddings_ALL) - 1)
        embeddings_ALL_train.append(embeddings_ALL[temp])
    embeddings_ALL_train = np.array(embeddings_ALL_train)
    embeddings_ALL_phecodes_train = list(embeddings_ALL_phecodes)
    valid_num = len(embeddings_ALL_phecodes)
    for i in range(total_train_num - valid_num):
        temp = random.randint(0, len(embeddings_ALL_phecodes) - 1)
        embeddings_ALL_phecodes_train.append(embeddings_ALL_phecodes[temp])
    embeddings_ALL_phecodes_train = np.array(embeddings_ALL_phecodes_train)
    ds_train = tf.data.Dataset.from_tensor_slices((drug_features_1_SMILES_train, drug_labels_train, ID_ID_train,
                                                   SMILE_mapping, embedding_mapping,
                                                   embedding_CUI_train, label_name_train,
                                                   drug_features_1_SMILES_train_add,
                                                   embedding_CUI_train_add, drug_labels_train_add,
                                                   embeddings_ALL_train,
                                                   SMILE_ALL_unlabel_train,
                                                   embeddings_ALL_phecodes_train)) \
        .shuffle(buffer_size=1024).batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE).cache()
    ds_test = tf.data.Dataset.from_tensor_slices((drug_features_1_SMILES_test, drug_labels_test,
                                                  ID_ID_test,
                                                  embedding_CUI_test, label_name_test)) \
        .shuffle(buffer_size=1024).batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE).cache()


def train_step(model, features_seq1,labels,embedding_seq,embedding_feature,cui_embedding,
               feature_seq1_add,embedding_add,label_add,embedding_unsuper,
               smile_unlabeled,gan_weights,embedding_phecode,wight_embedding_value=0.1):
    with tf.GradientTape(persistent=True) as tape:
        predictions,distance_similarity,predictions_add ,gan_loss_embedding, gan_loss_fusion,\
        sque_1_embedding= \
            model([features_seq1,embedding_seq,embedding_feature,cui_embedding,
                   feature_seq1_add,embedding_add,embedding_unsuper,smile_unlabeled,embedding_phecode], training=True)
        predictions=tf.cast(predictions,tf.float32)
        predictions_add = tf.cast(predictions_add, tf.float32)
        labels = tf.cast(labels, tf.float32)
        label_add = tf.cast(label_add, tf.float32)
        loss_binary = tf.nn.sigmoid_cross_entropy_with_logits(labels, predictions)
        loss_binary_add = tf.nn.sigmoid_cross_entropy_with_logits(label_add, predictions_add)
        loss_binary = tf.reduce_mean(tf.reduce_sum(loss_binary, axis=1), name="sigmoid_losses")
        loss_binary_add = tf.reduce_mean(tf.reduce_sum(loss_binary_add, axis=1), name="sigmoid_losses_add")
        loss=(loss_binary+weight_assist*loss_binary_add)*(1-wight_embedding) \
             +wight_embedding*distance_similarity
        mapping_loss_embedding=(-weight_emb_gan*gan_loss_embedding)*gan_weights
        mapping_loss_fusion = ( - weight_pair_gan * gan_loss_fusion) * gan_weights
        gan_loss_total=gan_loss_embedding+gan_loss_fusion
    all_variables = model.trainable_variables
    all_variables_NONdiscriminator = [v for v in all_variables if not 'discriminator_adv' in v.name]
    gradients = tape.gradient(loss, all_variables_NONdiscriminator)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, all_variables_NONdiscriminator))

    all_variables_embedding = [v for v in all_variables if 'extractor' in v.name]# or 'embedding' in v.name
    gradients = tape.gradient(mapping_loss_embedding, all_variables_embedding)
    grads = [tf.clip_by_norm(g, norm_gan) for g in gradients]
    optimizer.apply_gradients(grads_and_vars=zip(grads, all_variables_embedding))

    all_variables_fusion = [v for v in all_variables if 'classifier_predictor' in v.name or 'extractor dgfdg' in v.name]  # or 'embedding' in v.name
    gradients = tape.gradient(mapping_loss_fusion, all_variables_fusion)
    grads = [tf.clip_by_norm(g, norm_gan) for g in gradients]
    optimizer.apply_gradients(grads_and_vars=zip(grads, all_variables_fusion))

    all_variables_discriminator = [v for v in all_variables if 'discriminator_adv' in v.name]
    #print ("all_variables_discriminator: ",all_variables_discriminator)
    gradients_gan = tape.gradient(gan_loss_total,all_variables_discriminator)
    grads_gan = [tf.clip_by_norm(g, norm_gan_discriniator) for g in gradients_gan]
    optimizer_dic.apply_gradients(grads_and_vars=zip(grads_gan, all_variables_discriminator))

    train_loss.update_state(loss_binary)
    train_loss_gan_embedding.update_state(gan_loss_embedding)
    train_loss_gan_fusion.update_state(gan_loss_fusion)
    predictions=tf.nn.sigmoid(predictions)
    train_metric.update_state(labels, predictions)
    train_loss_distance.update_state(distance_similarity)
    loss_out=loss.numpy()
    return loss_out,predictions.numpy(),distance_similarity.numpy()

def valid_step(model, features_seq1,labels,embedding_seq,embedding_feature,cui_embedding,
               feature_seq1_add,embedding_add,label_add):
    predictions,distance_similarity,predictions_add,gan_loss_embedding, gan_loss_fusion,sque_1_embedding= \
        model([features_seq1,embedding_seq,embedding_feature,
              cui_embedding,feature_seq1_add,embedding_add,
               embedding_add,feature_seq1_add,embedding_feature], training=False)
    predictions=tf.cast(predictions,tf.float32)
    predictions_add = tf.cast(predictions_add, tf.float32)
    labels = tf.cast(labels, tf.float32)
    loss_binary = tf.nn.sigmoid_cross_entropy_with_logits(labels, predictions)
    loss_binary = tf.reduce_mean(tf.reduce_sum(loss_binary, axis=1), name="sigmoid_losses")
    valid_loss.update_state(loss_binary)
    test_loss_gan_embedding.update_state(gan_loss_embedding)
    test_loss_gan_fusion.update_state(gan_loss_fusion)
    predictions=tf.nn.sigmoid(predictions)
    predictions_add = tf.nn.sigmoid(predictions_add)
    valid_metric.update_state(labels, predictions)
    test_loss_distance.update_state(distance_similarity)
    return loss_binary.numpy(),predictions.numpy(),distance_similarity.numpy(),predictions_add.numpy(),sque_1_embedding.numpy()




def Model_interaction():
    inputs_smile = layers.Input(shape=(length_SMILE,))
    inputs_smile_embedding = layers.Input(shape=(length_SMILE,))
    inputs_smile_add = layers.Input(shape=(length_SMILE,))
    inputs_smile_unlabel = layers.Input(shape=(length_SMILE,))
    inputs_CUI_embedding = layers.Input(shape=(dim_emb,))
    inputs_CUI_embedding_add = layers.Input(shape=(dim_emb,))
    inputs_F_embedding = layers.Input(shape=(dim_emb,))
    inputs_embedding_unsuper = layers.Input(shape=(dim_emb,))
    inputs_embedding_phecodes = layers.Input(shape=(dim_emb,))
    embedding_layer = tf.keras.layers.Embedding(size_token, dim_token, input_length=length_SMILE, mask_zero=True)
    sque_1 = embedding_layer(inputs_smile)
    sque_embedding = embedding_layer(inputs_smile_embedding)
    sque_1_add = embedding_layer(inputs_smile_add)
    sque_1_unlabel = embedding_layer(inputs_smile_unlabel)
    conv_1 = tf.keras.layers.Conv1D(96, 7, strides=2, activation=tf.nn.leaky_relu,name="extractor/conv1")
    conv_2 = tf.keras.layers.Conv1D(96, 7, strides=2, activation=tf.nn.leaky_relu,name="extractor/conv2")
    GRU1 = tf.keras.layers.GRU(units=128, recurrent_dropout=0.1, return_sequences=False,name="extractor/GRU1")
    GRU1_bidirectional = tf.keras.layers.Bidirectional(GRU1,merge_mode="ave")
    sque_fcn1 = layers.Dense(128, activation=tf.nn.leaky_relu,name="extractor/fcn1")
    fcn_fusion = layers.Dense(64, activation=tf.nn.leaky_relu,name="classifier_predictor/fusion_drug")
    fcn_CUIs = layers.Dense(64, activation=tf.nn.leaky_relu,name="classifier_predictor/fusion_CUI")
    fcn_classifier = layers.Dense(48, activation=tf.nn.leaky_relu)
    fcn_classifier_add = layers.Dense(48, activation=tf.nn.leaky_relu)
    embedding_fcn_pre = layers.Dense(dim_emb, activation=tf.nn.leaky_relu,name="embedding/pre")
    embedding_fcn = layers.Dense(dim_emb, activation=None,name="embedding/prediction")
    domain_classifier_embedding = Sequential([
        layers.Dense(256, activation=tf.nn.relu, name="discriminator_adv/embedding/fcn1"),
        layers.Dropout(rate=0.5),
        layers.Dense(256, activation=tf.nn.leaky_relu, name="discriminator_adv/embedding/fcn2"),
        layers.Dropout(rate=0.5),
        layers.Dense(1, activation=None, name="discriminator_adv/embedding/fcn3")
    ])
    domain_classifier_fusion = Sequential([
        layers.Dense(256, activation=tf.nn.relu, name="discriminator_adv/fusion/fcn1"),
        layers.Dropout(rate=0.5),
        layers.Dense(256, activation=tf.nn.leaky_relu, name="discriminator_adv/fusion/fcn2"),
        layers.Dropout(rate=0.5),
        layers.Dense(1, activation=None, name="discriminator_adv/fusion/fcn3")
    ])

    def extractor(sque_smile):
        sque = conv_1(sque_smile)
        sque = conv_2(sque)
        sque = GRU1_bidirectional(sque)
        sque = tf.nn.dropout(sque, rate=0.1)
        sque = sque_fcn1(sque)
        return sque

    #########################
    sque_1 = extractor(sque_1)
    sque_1_unlabel=extractor(sque_1_unlabel)
    sque_1_add=extractor(sque_1_add)
    sque_embedding = extractor(sque_embedding)

    ##############getting the embeddings#####################
    sque_1_embedding = embedding_fcn_pre(sque_1)
    sque_embedding = embedding_fcn_pre(sque_embedding)
    sque_1_unlabel_embedding=embedding_fcn_pre(sque_1_unlabel)
    sque_1_embedding = embedding_fcn(sque_1_embedding)
    sque_embedding = embedding_fcn(sque_embedding)
    sque_1_unlabel_embedding = embedding_fcn(sque_1_unlabel_embedding)
    ##############getting the embeddings#####################
    distance_similarity = tf.reduce_mean(tf.reduce_mean(tf.square(sque_embedding - inputs_F_embedding), -1))

    ########################embedding adversarial traiing
    source_adversary_label = tf.ones([tf.shape(inputs_embedding_unsuper)[0], 1], tf.float32)
    target_adversary_label = tf.zeros([tf.shape(sque_1_unlabel_embedding)[0], 1], tf.float32)
    adversary_label = tf.concat([source_adversary_label, target_adversary_label], 0)
    adversary_ft = tf.concat([inputs_embedding_unsuper, sque_1_unlabel_embedding], 0)
    adver_layer_out = domain_classifier_embedding(adversary_ft)
    adversary_loss_embedding =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=adver_layer_out, labels=adversary_label))
    #####################################

    sque_1 = tf.nn.dropout(sque_1, rate=0.1)
    sque_1 = fcn_fusion(sque_1)
    sque_1 = tf.nn.dropout(sque_1, rate=0.1)
    sque_1_unlabel = tf.nn.dropout(sque_1_unlabel, rate=0.1)
    sque_1_unlabel = fcn_fusion(sque_1_unlabel)
    sque_1_unlabel = tf.nn.dropout(sque_1_unlabel, rate=0.1)
    sque_1_add = tf.nn.dropout(sque_1_add, rate=0.1)
    sque_1_add = fcn_fusion(sque_1_add)
    sque_1_add = tf.nn.dropout(sque_1_add, rate=0.1)

    CUI_embedding = fcn_CUIs(inputs_CUI_embedding)
    CUI_embedding = tf.nn.dropout(CUI_embedding, rate=0.1)

    CUI_embedding_add = fcn_CUIs(inputs_CUI_embedding_add)
    CUI_embedding_add = tf.nn.dropout(CUI_embedding_add, rate=0.1)

    sque_fused_in = layers.concatenate([sque_1, CUI_embedding],axis=-1)
    sque_fused_unlabel_in = layers.concatenate([sque_1_unlabel, CUI_embedding],axis=-1)
    sque_fused_add_in = layers.concatenate([sque_1_add, CUI_embedding_add],axis=-1)
    # sque_fused = tf.nn.dropout(sque_fused, rate=0.2)
    sque_fused_total = []
    sque_fused_unlabel_total = []
    sque_fused_add_total = []
    prediction_total = []
    prediction_add_total = []
    for interi in range(3):
        sque_fused = fcn_classifier(sque_fused_in)
        sque_fused = tf.nn.dropout(sque_fused, rate=0.1)
        sque_fused_total.append(sque_fused)

        sque_fused_unlabel = fcn_classifier(sque_fused_unlabel_in)
        sque_fused_unlabel = tf.nn.dropout(sque_fused_unlabel, rate=0.1)

        sque_fused_unlabel_total.append(sque_fused_unlabel)


        sque_fused_add = fcn_classifier_add(sque_fused_add_in)
        sque_fused_add = tf.nn.dropout(sque_fused_add, rate=0.1)

        sque_fused_add_total.append(sque_fused_add)

        prediction = layers.Dense(num_out, activation=None)(sque_fused)
        prediction_add = layers.Dense(num_out, activation=None)(sque_fused_add)
        prediction_total.append(prediction)
        prediction_add_total.append(prediction_add)

    sque_fused = tf.reduce_mean(sque_fused_total, axis=0)
    sque_fused_add = tf.reduce_mean(sque_fused_add_total, axis=0)
    sque_fused_unlabel = tf.reduce_mean(sque_fused_unlabel_total, axis=0)

    prediction = tf.reduce_mean(prediction_total, axis=0)
    prediction_add = tf.reduce_mean(prediction_add_total, axis=0)

    ########################fusion adversarial traiing
    source_adversary_label = tf.ones([tf.shape(sque_fused)[0], 1], tf.float32)
    target_adversary_label = tf.zeros([tf.shape(sque_fused_unlabel)[0], 1], tf.float32)
    adversary_label = tf.concat([source_adversary_label, target_adversary_label], 0)
    adversary_ft = tf.concat([sque_fused, sque_fused_unlabel], 0)
    adver_layer_out = domain_classifier_fusion(adversary_ft)
    adversary_loss_fusion = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=adver_layer_out, labels=adversary_label))
    #####################################

    model = models.Model(inputs=[inputs_smile, inputs_smile_embedding, inputs_F_embedding, inputs_CUI_embedding,
                                 inputs_smile_add,inputs_CUI_embedding_add,
                                 inputs_embedding_unsuper,inputs_smile_unlabel,inputs_embedding_phecodes],
                         outputs=[prediction, distance_similarity,prediction_add,
                                  adversary_loss_embedding, adversary_loss_fusion,sque_1_embedding])

    return model


def train(model, ds_train, ds_test, train_epoches):
    epoch=0
    epoch_val=3
    while (epoch<train_epoches):
        epoch += 1
        print ("----train_epoches total: ",train_epoches)
        similarity_train=[]
        gan_weights_input=weight_emb_gan
        similarity_test=[]
        label_name_total=[]
        prediction_total=[]
        label_total=[]
        auc_class = []
        Prc_class = []
        if flag_gan_progress==True:
            p_value = 1.0 * epoch / (train_epoches+1) * 1.0
            gan_weights_input = weight_emb_gan / (1.0 + math.exp(-10.0 * p_value)) - 1.0
      
        for features_seq1,labels,ID_ID_batch,embedding_seq,embedding_feature,cui_embedding,labelname,\
                    feature_seq1_add,embedding_add,label_add, embedding_unsuper,smile_unlabeled,embedding_phecodes in ds_train:
            loss_out,predictions,distance_similarity=train_step(model,features_seq1,labels,embedding_seq,
                    embedding_feature,cui_embedding,feature_seq1_add,embedding_add,label_add,embedding_unsuper,smile_unlabeled,
                               gan_weights_input,embedding_phecodes)
            similarity_train.append(distance_similarity)
                
            if epoch%epoch_val==1:
                i_number = 0
                for labels, ID_ID_batch, cui_embedding, labelname in ds_test:
                    loss_out, prediction, distance_similarity,predictions_add, embedding_pred = valid_step(model,features_seq1, labels, embedding_seq,
                                                                           embedding_feature, cui_embedding,
                                                                           feature_seq1_add, embedding_add, label_add)
                    similarity_test.append(distance_similarity)
                    label_name_total.extend(list(labelname.numpy()))
                    if i_number == 0:
                        prediction_total = np.array(prediction)
                        label_total = np.array(labels.numpy())
                    else:
                        prediction_total = np.concatenate((prediction_total, np.array(prediction)), axis=0)
                        label_total = np.concatenate((label_total, np.array(labels.numpy())), axis=0)
                    i_number += 1
                label_list = list(np.array(label_total).reshape((-1, 1)))
                prediction_list = list(np.array(prediction_total).reshape((-1, 1)))
                #################compute metrics by class sepaterely############
                if True:
                    dic_name_prob = {}
                    dic_name_label = {}
                    for prob, label, name in zip(prediction_list, label_list, label_name_total):
                        dic_name_prob.setdefault(name, []).append(prob)
                        dic_name_label.setdefault(name, []).append(label)
                    label_name_save_class = []
                    for name in dic_name_prob.keys():
                        label_name_save_class.append(name)
                        label = np.array(dic_name_label[name])
                        score = np.array(dic_name_prob[name])
                        auc_area = roc_auc_score(y_true=label, y_score=score)
                        precision, recall, thresholds = precision_recall_curve(label, score)
                        Prc = auc(recall, precision)
                        auc_class.append(auc_area)
                        Prc_class.append(Prc)

                    y_true_get = np.array(label_total).reshape((-1, 1))
                    score_get = np.array(prediction_total).reshape((-1, 1))
                    y_true_get = np.array(y_true_get)
                    score_get = np.array(score_get)
                    auc_overall = roc_auc_score(y_true=y_true_get,y_score=score_get)  ###
                    precision, recall, thresholds = precision_recall_curve(y_true_get, score_get)
                    prc_overall = auc(recall, precision)

                    logs = '--Epoch={},Loss:{}, Val_Loss:{},auc_train:{}, auc_overall:{},' \
                           'prc_overall:{},auc_class:{}' \
                           ',Prc_class:{} '
                    tf.print(tf.strings.format(logs, (epoch, train_loss.result(), valid_loss.result(), train_metric.result(),auc_overall,
                                              prc_overall,np.mean(auc_class),np.mean(Prc_class))))
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()
        train_loss_gan_embedding.reset_states()
        train_loss_gan_fusion.reset_states()
        test_loss_gan_embedding.reset_states()
        test_loss_gan_fusion.reset_states()
        train_loss_distance.reset_states()
        test_loss_distance.reset_states()

if __name__ == '__main__':
    model = Model_interaction()
    print("model training begin....")
    train(model, ds_train, ds_test, train_epoches)
    print("model training end....")
    model.save(ARGS.model_savename)
    print("model saving end....")

    
    
    




