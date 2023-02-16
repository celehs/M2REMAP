import pickle
import numpy as np
import pandas as pd
from random import choice
import random
import os
from sklearn.metrics.pairwise import cosine_similarity

def cosine(feature_1,feauture2):
    feature1_L1 = np.sqrt(np.sum(np.square(feature_1)))
    feature2_L1 = np.sqrt(np.sum(np.square(feauture2)))
    cosine_similarity = abs(np.sum(feature_1 * feauture2) / np.sum(feature1_L1 * feature2_L1))
    return cosine_similarity

def loaddata(dirr_main,file_train,file_test,file_embedding_disease,file_embedding_drugs,
             file_ID_token,file_DB_embedding,
             cosine_T,train_ratio):
    with open(dirr_main + file_ID_token, 'rb') as fid:
        (dic_charactor, dic_ID_SMILES) = pickle.load(fid)
    df = pd.read_csv(dirr_main + file_embedding_disease)
    CUIs_has_embedding = list(df["CUIs"])
    sidenames = list(df["sidenames"])
    dic_CUIs_feature = {}
    dic_CUIs_names = {}
    CUIs_feature = np.array(df[df.columns[2:2 + 500]] * 1.0)
    for i in range(len(CUIs_has_embedding)):
        dic_CUIs_feature[str(CUIs_has_embedding[i])] = CUIs_feature[i, :]
        dic_CUIs_names[str(CUIs_has_embedding[i])] = sidenames[i]
    if not os.path.exists(dirr_main + file_DB_embedding):
        SMILE_embedding = []
        feature_embedding = []
        df = pd.read_csv(dirr_main + file_embedding_drugs)
        drugbank_ID_availble = list(df["DrugBank ID"])
        dic_bankID_feature = {}
        drugbank_feature = np.array(df[df.columns[4:]])
        for i in range(len(drugbank_ID_availble)):
            drug = str(drugbank_ID_availble[i])
            if drug in dic_ID_SMILES:
                SMILE_embedding.append(dic_ID_SMILES[str(drug)])
                feature_embedding.append(drugbank_feature[i, :])
        print("---saving VA_DB_embedding.pkl")
        with open(dirr_main + file_DB_embedding, 'wb') as fid:
            pickle.dump((SMILE_embedding, np.array(feature_embedding)), fid)
    #######################
    smile1_train=[]
    label_train=[]
    cui_train=[]
    ID_ID_train=[]
    label_category_train=[]
    label_name_train=[]
    smile1_test=[]
    label_test=[]
    cui_test=[]
    ID_ID_test=[]
    label_name_test=[]
    label_category_test=[]

    IDs_total=list(dic_ID_SMILES.keys())
    print ("begining sampling data...")
    df_sider=pd.read_csv(dirr_main+file_train)
    Drug1_ID=list(df_sider["drugbank_id"])
    Polypharmacy_CUI=list(df_sider["umls_cui_from_meddra"])
    Polypharmacy_CUI_unique_train=list({}.fromkeys(Polypharmacy_CUI).keys())
    dic_drug_cuis={}
    for id,cui in zip(Drug1_ID,Polypharmacy_CUI):
        dic_drug_cuis.setdefault(id,[]).append(str(cui))
    print ("dic_drug_cuis: ",len(dic_drug_cuis))
    Polypharmacy_CUI_unique=list({}.fromkeys(Polypharmacy_CUI).keys())
    Polypharmacy_CUI_unique_set_train=set(Polypharmacy_CUI_unique)
    print ("Polypharmacy_CUI_unique len: ",len(Polypharmacy_CUI_unique))
    drugs_train={}
    drugs_test={}
    pairs_train_positive=0
    pairs_train_negative=0
    pairs_test_positive=0
    pairs_test_negative=0

    drug_train_valid={}
    drug_test_valid={}
    drug_total_valid={}
    pair_total_valid={}
    CUIs_total_valid={}

    for id1 in dic_drug_cuis:
        if str(id1) in dic_ID_SMILES:
            if len(drugs_train) % 5 == 1:
                print("train_drugs len: ", len(drugs_train))
            cui_list=dic_drug_cuis[id1]
            not_labeled_sides_overlapped = list(Polypharmacy_CUI_unique_set_train - set(cui_list))
            not_labeled_sides=list(Polypharmacy_CUI_unique_set_train-set(cui_list))
            #print ("not_labeled_sides: ",not_labeled_sides)
            smile1 = dic_ID_SMILES[str(id1)]
            id_negative = choice(IDs_total)
            smile_negative = dic_ID_SMILES[id_negative]
            id_id_neg = str(smile_negative)
            ##################### the negative should not close to any of the positive CUIs
            positive_CUI_embedding = []
            negative_CUI_embedding = []
            for cui in cui_list:
                cui_embedding = dic_CUIs_feature[str(cui)]
                positive_CUI_embedding.append(cui_embedding)
            if len(positive_CUI_embedding) > 0:
                positive_CUI_embedding = np.array(positive_CUI_embedding)
                # print ("len(cui_list): ",len(cui_list))
                # print("len(not_labeled_sides): ", len(not_labeled_sides))
                for cui in not_labeled_sides_overlapped:

                    cui_embedding = dic_CUIs_feature[str(cui)]
                    negative_CUI_embedding.append(cui_embedding)

                negative_CUI_embedding = np.array(negative_CUI_embedding)
                similarity_N_P = cosine_similarity(negative_CUI_embedding, positive_CUI_embedding)
                similarity_N_P_min = np.min(similarity_N_P, axis=-1)
                # print ("similarity_P_N_min: ",similarity_N_P_min.shape)
                #########################################training drugs
                CUI_positive_valid_num = 0
                for cui in cui_list:

                    drug_train_valid[id1] = 1
                    drug_total_valid[id1] = 1
                    pair_total_valid[str(id1) + str(cui)] = 1
                    CUIs_total_valid[str(cui)] = 1


                    CUI_positive_valid_num+=1
                    pairs_train_positive+=1
                    cui_embedding = dic_CUIs_feature[str(cui)]
                    label_name = dic_CUIs_names[str(cui)]
                    label_name = str(id1)
                    drugs_train[id1]=1
                    ###########positive
                    smile1_train.append(smile1)
                    label_train.append([1])
                    cui_train.append(cui_embedding)
                    ID_ID_train.append(id1)
                    label_name_train.append(label_name)

                nagetive_sample_num = CUI_positive_valid_num * 5
                cosine_T = cosine_T
                valid_negative_num = 0
                negative_valid_flag=False
                for neg_cui_i in range(len(not_labeled_sides_overlapped)):
                    if similarity_N_P_min[neg_cui_i] < cosine_T and valid_negative_num < nagetive_sample_num:
                        pairs_train_negative+=1
                        negative_valid_flag=True
                        valid_negative_num += 1
                        cui = not_labeled_sides_overlapped[neg_cui_i]  # not_labeled_sides_overlapped not_labeled_sides
                        cui_embedding_neg = dic_CUIs_feature[str(cui)]
                        label_name = dic_CUIs_names[str(cui)]
                        label_name = str(id1)
                        smile1_train.append(smile1)
                        label_train.append([0])
                        cui_train.append(cui_embedding_neg)
                        ID_ID_train.append(id1)
                        label_name_train.append(label_name)
                if negative_valid_flag == False:
                    pairs_train_negative += 1
                    cui = choice(not_labeled_sides)
                    cui_embedding_neg = dic_CUIs_feature[str(cui)]
                    label_name = dic_CUIs_names[str(cui)]
                    label_name = str(id1)
                    smile1_train.append(smile1)
                    label_train.append([0])
                    cui_train.append(cui_embedding_neg)
                    ID_ID_train.append(id1)
                    label_name_train.append(label_name)

    drug_ID_save=[]
    side_CUI_save=[]
    Y_save=[]
    df_sider = pd.read_csv(dirr_main + file_test)
    Drug1_ID = list(df_sider["drugbank_id"])
    Polypharmacy_CUI = list(df_sider["umls_cui_from_meddra"])
    Polypharmacy_CUI_unique=list({}.fromkeys(Polypharmacy_CUI).keys())
    Polypharmacy_CUI_unique_set=set(Polypharmacy_CUI_unique)
    dic_drug_cuis={}
    for id,cui in zip(Drug1_ID,Polypharmacy_CUI):
        dic_drug_cuis.setdefault(id,[]).append(str(cui))
    for id1 in dic_drug_cuis:
        if len(drugs_test) % 5 == 1:
            print("drugs_test len: ", len(drugs_test))
        if str(id1) in dic_ID_SMILES:
            cui_list = dic_drug_cuis[id1]
            not_labeled_sides = list(Polypharmacy_CUI_unique_set - set(cui_list))
            not_labeled_sides_overlapped = list(Polypharmacy_CUI_unique_set_train - set(cui_list))

            # print ("not_labeled_sides: ",not_labeled_sides)
            smile1 = dic_ID_SMILES[str(id1)]
            id_negative = choice(IDs_total)
            smile_negative = dic_ID_SMILES[id_negative]
            id_id_neg = str(smile_negative)
            dic_CUI_negative_test = {}

            ##################### the negative should not close to any of the positive CUIs
            positive_CUI_embedding = []
            negative_CUI_embedding = []
            for cui in cui_list:
                cui_embedding = dic_CUIs_feature[str(cui)]
                positive_CUI_embedding.append(cui_embedding)
            if len(positive_CUI_embedding) > 0:
                positive_CUI_embedding = np.array(positive_CUI_embedding)
                # print ("len(cui_list): ",len(cui_list))
                # print("len(not_labeled_sides): ", len(not_labeled_sides))
                for cui in not_labeled_sides_overlapped:
                    cui_embedding = dic_CUIs_feature[str(cui)]
                    negative_CUI_embedding.append(cui_embedding)

                negative_CUI_embedding = np.array(negative_CUI_embedding)
                similarity_N_P = cosine_similarity(negative_CUI_embedding, positive_CUI_embedding)
                similarity_N_P_min = np.min(similarity_N_P, axis=-1)

                CUI_positive_valid_num = 0
                for cui in cui_list:

                    drug_test_valid[id1] = 1
                    drug_total_valid[id1] = 1
                    pair_total_valid[str(id1) + str(cui)] = 1
                    CUIs_total_valid[str(cui)] = 1


                    pairs_test_positive += 1
                    CUI_positive_valid_num+=1
                    cui_embedding = dic_CUIs_feature[str(cui)]
                    label_name = dic_CUIs_names[str(cui)]
                    label_name = str(id1)
                    drugs_test[id1] = 1
                    ###########positive
                    smile1_test.append(smile1)
                    label_test.append([1])
                    cui_test.append(cui_embedding)
                    ID_ID_test.append(id1)
                    label_name_test.append(label_name)

                    drug_ID_save.append(id1)
                    side_CUI_save.append(cui)
                    Y_save.append(1)
                valid_negative_num=0
                nagetive_sample_num = CUI_positive_valid_num * 200
                if CUI_positive_valid_num>0:
                    for neg_cui_i in range(len(not_labeled_sides_overlapped)):
                        if similarity_N_P_min[neg_cui_i] < cosine_T and valid_negative_num < nagetive_sample_num:
                            pairs_test_negative += 1
                            negative_valid_flag = True
                            valid_negative_num += 1
                            cui = not_labeled_sides_overlapped[neg_cui_i]  # not_labeled_sides_overlapped not_labeled_sides
                            cui_embedding_neg = dic_CUIs_feature[str(cui)]
                            label_name = dic_CUIs_names[str(cui)]
                            label_name = str(id1)
                            smile1_test.append(smile1)
                            label_test.append([0])
                            cui_test.append(cui_embedding_neg)
                            ID_ID_test.append(id1)
                            label_name_test.append(label_name)

                            drug_ID_save.append(id1)
                            side_CUI_save.append(cui)
                            Y_save.append(0)

                    if valid_negative_num==0:
                        pairs_test_negative += 1
                        cui = choice(not_labeled_sides)
                        cui_embedding_neg = dic_CUIs_feature[str(cui)]
                        label_name = dic_CUIs_names[str(cui)]
                        label_name = str(id1)
                        cui_embedding = dic_CUIs_feature[str(cui)]
                        ###########negative
                        smile1_test.append(smile1)
                        label_test.append([0])
                        cui_test.append(cui_embedding_neg)
                        ID_ID_test.append(id1)
                        label_name_test.append(label_name)

                        drug_ID_save.append(id1)
                        side_CUI_save.append(cui)
                        Y_save.append(0)
    label_test = np.array(label_test)
    label_train = np.array(label_train)
    cui_train = np.array(cui_train)
    cui_test = np.array(cui_test)
    return smile1_train,ID_ID_train,label_train,cui_train,label_name_train,\
           smile1_test,ID_ID_test,label_test,cui_test,label_name_test


