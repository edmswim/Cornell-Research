import random
import numpy as np
import keras

# for participant independent
# assign feature data into train, validation, or testing
def dependent_assign(isOneHotLabel, cutoff_train, cutoff_val, train_x, train_y, val_x, val_y, test_x, test_y, x_vec, label, leave_one_patient_id, user_id):
	# doesn't do leave one patient out
    if leave_one_patient_id is None:
        p = random.random()

        if p <= cutoff_train:
            #training
            if len(train_x) == 0:
                train_x = x_vec
            else:
                train_x = np.concatenate((train_x, x_vec), axis=0)

            if isOneHotLabel:
                one_hot_labels = [keras.utils.to_categorical(
                    int(label), num_classes=4)]
                if len(train_y) == 0:
                    train_y = one_hot_labels
                else:
                    train_y = np.concatenate((train_y, one_hot_labels), axis=0)
            else:
                train_y = np.append(train_y, int(label))

        elif p > cutoff_train and p <= cutoff_val:
            # validation
            if len(val_x) == 0:
                val_x = x_vec
            else:
                val_x = np.concatenate((val_x, x_vec), axis=0)

            if isOneHotLabel:
                one_hot_labels = [keras.utils.to_categorical(
                    int(label), num_classes=4)]
                if len(val_y) == 0:
                    val_y = one_hot_labels
                else:
                    val_y = np.concatenate((val_y, one_hot_labels), axis=0)
            else:
                val_y = np.append(val_y, int(label))
        else:
            # testing
            if len(test_x) == 0:
                test_x = x_vec
            else:
                test_x = np.concatenate((test_x, x_vec), axis=0)

            #the truth values
            test_y = np.append(test_y, int(label))

    else:
        # this is for leave one patient out
        if leave_one_patient_id == user_id:
            if len(test_x) == 0:
                test_x = x_vec
            else:
                test_x = np.concatenate((test_x, x_vec), axis=0)

            #the truth values
            test_y = np.append(test_y, int(label))
        else:
            # put everything into training that is not the target userid
            if len(train_x) == 0:
                train_x = x_vec
            else:
                train_x = np.concatenate((train_x, x_vec), axis=0)

            if isOneHotLabel:
                one_hot_labels = [keras.utils.to_categorical(
                    int(label), num_classes=4)]
                if len(train_y) == 0:
                    train_y = one_hot_labels
                else:
                    train_y = np.concatenate((train_y, one_hot_labels), axis=0)
            else:
                train_y = np.append(train_y, int(label))



    return train_x, train_y, val_x, val_y, test_x, test_y




# for participant dependent
# assign feature data into train, validation, or testing
def independent_assign(isOneHotLabel, userid, trainingid, validationid, testingid, train_x, train_y, val_x, val_y, test_x, test_y, x_vec, label):
    #training
    if userid in trainingid:
        if len(train_x) == 0:
            train_x = x_vec
        else:
            train_x = np.concatenate((train_x, x_vec), axis=0)

        if isOneHotLabel:
            one_hot_labels = [keras.utils.to_categorical(
                int(label), num_classes=4)]
            if len(train_y) == 0:
                train_y = one_hot_labels
            else:
                train_y = np.concatenate((train_y, one_hot_labels), axis=0)
        else:
            train_y = np.append(train_y, int(label))

    # validation
    elif userid in validationid:
        if len(val_x) == 0:
            val_x = x_vec
        else:
            val_x = np.concatenate((val_x, x_vec), axis=0)

        if isOneHotLabel:
            one_hot_labels = [keras.utils.to_categorical(
                int(label), num_classes=4)]
            if len(val_y) == 0:
                val_y = one_hot_labels
            else:
                val_y = np.concatenate((val_y, one_hot_labels), axis=0)
        else:
            val_y = np.append(val_y, int(label))

    # testing
    elif userid in testingid:
        if len(test_x) == 0:
            test_x = x_vec
        else:
            test_x = np.concatenate((test_x, x_vec), axis=0)

        #the truth values
        test_y = np.append(test_y, int(label))

    return train_x, train_y, val_x, val_y, test_x, test_y

