import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tkinter import *
from tkinter import font
from tkinter import ttk
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import canvas
import graphviz
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import plotly.express as px
import csv
from sklearn import model_selection
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Prediction_model
def prediction_model():
    df = pd.read_csv("heart.csv")
    string_col = df.select_dtypes(include="object").columns
    df[string_col] = df[string_col].astype("string")
    string_col = df.select_dtypes("string").columns.to_list()
    num_col = df.columns.to_list()
    for col in string_col:
        num_col.remove(col)
    num_col.remove("HeartDisease")

    # Tree Based Algorithms
    # df_tree = df.apply(LabelEncoder().fit_transform)
    df_tree = pd.get_dummies(df, columns=string_col, drop_first=False)
    target = "HeartDisease"
    y = df_tree[target].values
    df_tree.drop("HeartDisease", axis=1, inplace=True)
    df_tree = pd.concat([df_tree, df[target]], axis=1)
    feature_col_tree = df_tree.columns.to_list()
    feature_col_tree.remove(target)
    ###
    acc_RandF = []
    kf = model_selection.StratifiedKFold(n_splits=5)
    for fold, (trn_, val_) in enumerate(kf.split(X=df_tree, y=y)):
        X_train = df_tree.loc[trn_, feature_col_tree]
        y_train = df_tree.loc[trn_, target]

        X_valid = df_tree.loc[val_, feature_col_tree]
        y_valid = df_tree.loc[val_, target]

        clf = RandomForestClassifier(n_estimators=200, criterion="entropy")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)
        acc = roc_auc_score(y_valid, y_pred)
        acc_RandF.append(acc)

    acc_XGB = []
    kf = model_selection.StratifiedKFold(n_splits=5)
    for fold, (trn_, val_) in enumerate(kf.split(X=df_tree, y=y)):
        X_train = df_tree.loc[trn_, feature_col_tree]
        y_train = df_tree.loc[trn_, target]

        X_valid = df_tree.loc[val_, feature_col_tree]
        y_valid = df_tree.loc[val_, target]

        clf = XGBClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)
        acc = roc_auc_score(y_valid, y_pred)
        acc_XGB.append(acc)

    df_tree.drop("HeartDisease", axis=1, inplace=True)
    result1 = clf.predict(df_tree)
    result2 = clf.predict_proba(df_tree)
    diagnoz = np.take(result1, -1)
    predict = np.take(result2, -4) # проблемы с выводом вероятности
    print(diagnoz, predict)
    return diagnoz, predict

# Create an instance of Tkinter frame or window


root = Tk()
root.title('Диагностирование сердечных заболеваний')
root.iconbitmap(default="Medicine.ico")
font1 = font.Font(size=12, slant='italic')
font2 = font.Font(size=11, slant='italic')
# Set the geometry of tkinter frame
root.geometry("600x650")
# создаем набор вкладок
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill=BOTH)
# создаем пару фреймвов
frame1 = ttk.Frame(notebook)
frame2 = ttk.Frame(notebook)
frame1.pack(fill=BOTH, expand=True)
frame2.pack(fill=BOTH, expand=False)
# добавляем фреймы в качестве вкладок
notebook.add(frame1, text="Диагнстирование пациента")
notebook.add(frame2, text="Мониторинг")

label1 = ttk.Label(frame1, text="Введите данные пациента:", font=font1)
label1.place(x=20, y=20)
# Column 0
name = ttk.Label(frame1, text="Имя")
name.place(x=40, y=70)
name_var = StringVar()
name_entrybox = ttk.Entry(frame1, width=50, textvariable=name_var, justify='center')
name_entrybox.place(x=120, y=70)
# Column 1
age = ttk.Label(frame1, text="Возраст")
age.place(x=40, y=100)
age_spinbox = ttk.Spinbox(frame1, from_=1.0, to=100.0)
age_spinbox.place(x=120, y=100)
# Column 2
sex = ttk.Label(frame1, text="Пол")
sex.place(x=40, y=130)
sex_combobox = ttk.Combobox(frame1, values=["M", "F"])
sex_combobox.place(x=120, y=130)
# Column 3
chestpain = ttk.Label(frame1, text="Вид грудной\n      боли")
chestpain.place(x=25, y=160)
chestpain_combobox = ttk.Combobox(frame1, values=["TA", "ATA", "NAP", "ASY"])
chestpain_combobox.place(x=120, y=165)
# Column 4
restBP = ttk.Label(frame1, text="Давление")
restBP.place(x=30, y=195)
restBP_entrybox = ttk.Entry(frame1, width=8, textvariable=IntVar(), justify='center')
restBP_entrybox.place(x=120, y=195)
# Column 5
cholesterol = ttk.Label(frame1, text="Холестерин")
cholesterol.place(x=30, y=225)
cholesterol_entrybox = ttk.Entry(frame1, width=8, textvariable=IntVar(), justify='center')
cholesterol_entrybox.place(x=120, y=225)
# Column 6
enabled = IntVar()
sugar = ttk.Label(frame1, text="Уровень сахара")
sugar.place(x=20, y=255)
sugar_checkbutton = ttk.Checkbutton(frame1, text="Больше 120 мг/дл", variable=enabled)
sugar_checkbutton.place(x=120, y=255)
# Column 7
ecg = ttk.Label(frame1, text="ЭКГ")
ecg.place(x=30, y=285)
ecg_combobox = ttk.Combobox(frame1, values=["Normal", "ST", "LVH"])
ecg_combobox.place(x=120, y=285)
# Column 8
maxhr = ttk.Label(frame1, text="Макс. ЧСС")
maxhr.place(x=30, y=315)
maxhr_entrybox = ttk.Entry(frame1, width=8, textvariable=IntVar(), justify='center')
maxhr_entrybox.place(x=120, y=315)
# Column 9
angina = ttk.Label(frame1, text="Стенокардия \n (в нагрузке)")
angina.place(x=20, y=345)
angina_combobox = ttk.Combobox(frame1, values=["Y", "N"])
angina_combobox.place(x=120, y=350)
# Column 11
oldpeak = ttk.Label(frame1, text="Мин. сегмент ST")
oldpeak.place(x=20, y=385)
oldpeak_entrybox = ttk.Entry(frame1, width=8, textvariable=StringVar(), justify='center')
oldpeak_entrybox.place(x=120, y=385)
# Column 12
stslope = ttk.Label(frame1, text="Макс. сегмент ST")
stslope.place(x=15, y=415)
stslope_combobox = ttk.Combobox(frame1, values=["Flat", "Up", "Down"])
stslope_combobox.place(x=120, y=415)

img = PhotoImage(file='heart_red1.png')
Label(frame1, image=img).place(x=300, y=130)

result = list(prediction_model())
diagnoz = result[0]


def writeToFile():
    with open('heart.csv', 'a') as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow([age_spinbox.get(), sex_combobox.get(), chestpain_combobox.get(), restBP_entrybox.get(),
                    cholesterol_entrybox.get(), enabled.get(), ecg_combobox.get(),
                    maxhr_entrybox.get(), angina_combobox.get(), oldpeak_entrybox.get(),
                    stslope_combobox.get(), diagnoz])
        label_predict = ttk.Label(frame1, text="Запись в базу данных\nпроизошла успешно", font=font2)
        label_predict.place(x=65, y=500)


bd_button = Button(frame1, text='Внести данные', fg='black', bg='lightblue', command=writeToFile)
bd_button.place(x=85, y=470)


def predict_print():
    value = prediction_model()
    value = list(value)
    if value[0] == 0:
        if float(value[1]) > 0.5:
            value[1] = 1 - float(value[1])
        predict = f'Сердечных заболеваний не обнаружено \nВероятность заболевания: {round(value[1]*100, 2)}%'
    else:
        if value[1] < 0.5:
            value[1] = 1 - value[1]
        predict = f'Обнаружено сердечное заболевание!\nВероятность заболевания: {round(value[1]*100, 2)}%'
    label_predict = ttk.Label(frame1, text=f"Диагноз:\n{predict}", font=font2)
    label_predict.place(x=255, y=515)


predict_button = Button(frame1, text='Диагностировать',fg='white', bg='red', command=predict_print)
predict_button.place(x=305,y=470)

# second tab
df = pd.read_csv("heart.csv")
string_col = df.select_dtypes(include="object").columns
df[string_col] = df[string_col].astype("string")
string_col = df.select_dtypes("string").columns.to_list()
num_col = df.columns.to_list()
for col in string_col:
    num_col.remove(col)
num_col.remove("HeartDisease")

symptomChoice = ttk.Label(frame2, text="Выберите симптом для отображения: ", font=font2)
symptomChoice.place(x=10, y=15)
choices = ["Age", "Cholesterol", "RestingBP", "MaxHR", "Oldpeak"]
choices_var = StringVar(value=choices[0])
symptomChoice_combobox = ttk.Combobox(frame2, textvariable=choices_var, values=choices)
symptomChoice_combobox.place(x=290, y=17)


def draw_graph(symptom1):
    graph = df[symptom1].values
    fig1 = plt.figure(figsize=(5, 3), num=1, dpi=120)
    fig1.add_subplot().plot(graph)
    fig1.suptitle('Мониторинг симптомов')
    return fig1


def create_widget(fig):
    plot1 = FigureCanvasTkAgg(fig, frame2)
    plot1.get_tk_widget().place(x=0, y=50)


def callback():
    symptom = symptomChoice_combobox.get()
    graph = draw_graph(symptom)
    create_widget(graph)


draw_button = Button(frame2, bg='lightblue', text='Изобразить график', command=callback)
draw_button.place(x=455, y=15)

root.mainloop()