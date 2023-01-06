import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def slices(file_path):
    with open(file_path, 'r') as file:
        slice_up = file.read()
    __slices = slice_up.split('\n')
    return __slices


def print_results(y_test, predictions, success, baseline, total_times):
    print("Last score + report: " + "{:.2%}".format(accuracy_score(y_test, predictions)))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, target_names=["0", "1"]))
    print("\n")
    print('Average accuracy over last ' + str(total_times) + ' epochs is ' + "{:.2%}".format(success))
    print('Difference over baseline (guessing success every time) is '
          + "{:.2%}".format(success - baseline))


def testing(X, y, baseline, num_epochs):
    total_accuracy, total_times, start_count = 0.0, 0, num_epochs - 60

    for epoch in range(0, num_epochs):
        if start_count < 0: start_count = 0

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,shuffle=True)

        rfc = RandomForestClassifier(warm_start=True, random_state=100,n_estimators=1500)

        model_final = rfc.fit(X_train, y_train)
        predictions = model_final.predict(X_test)

        if epoch >= start_count:
            total_accuracy += accuracy_score(y_test, predictions)
            total_times += 1

    print_results(y_test, predictions, total_accuracy/total_times, baseline, total_times)


if __name__ == "__main__":
    df = pd.read_csv('CAX_Startup_Data.csv', encoding='latin1')
    num, total, ones = 0, 0, 0.0
    cols, params = slices('columns.txt'), slices('parameters.txt')

    df = df[cols]
    df = df.replace({'No Info': None, 'Success': 1, 'Failed': 0, 'No': 0, 'Yes': 1,
                        'Bachelors': 1, 'Masters': 2, 'PhD': 3, 'Both': 3, 'Tier_1': 1,
                        'Tier_2': 2, 'Low': 0, 'Medium': 1, 'High': 2, 'None': 0,
                        'unknown amount': 0}, inplace=False).fillna(0).astype(float)

    X = df[params]
    y = df['Dependent-Company Status']

    for i in y:
        num += 1
        total += i
    ones = total / num

    testing(X, y, ones, 120)
