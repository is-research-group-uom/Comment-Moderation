
def eval_dictionary(dict):
    # print("Dictionary: ", dict[0])

    matrixes = []
    for label in ['S', 'H', 'V', 'HR', 'SH', 'H2', 'S3', 'V2']:

        True_Positive = 0
        True_Negative = 0
        False_Positive = 0
        False_Negative = 0

        for data in dict:
            if label in data['Ai Labels: '] and label in data['Human Labels: ']:
                True_Positive += 1
            elif label not in data['Ai Labels: '] and label not in data['Human Labels: ']:
                True_Negative += 1
            elif label in data['Ai Labels: '] and label not in data['Human Labels: ']:
                False_Positive += 1
            elif label not in data['Ai Labels: '] and label in data['Human Labels: ']:
                False_Negative += 1

        confusion_matrix = {
            label:{
                "True Positive": True_Positive,
                "True Negative": True_Negative,
                "False Positive": False_Positive,
                "False Negative": False_Negative
            }
        }

        matrixes.append(confusion_matrix)

    for matrix in matrixes:
        for label in matrix:
            values = matrix[label]
            print(f'For {label}\nTrue Positive: {values['True Positive']} | True Negative: {values['True Negative']} | False Positive: {values['False Positive']} | False Negative: {values['False Negative']}')
            print('Precision: ', values['True Positive']/(values['True Positive']+values['False Positive']),'\nRecall: ', values['True Positive']/(values['True Positive']+values['False Negative']))