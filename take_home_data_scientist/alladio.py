#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:30:03 2023

@author: juani.alladio
"""

#%% Importacion de librerias
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from statistics import descriptive_statistics, type_and_missings

#%% Seteo el directorio 
os.chdir('/Users/juani.alladio/Desktop/Fligoo/datasets')

#%% Part I 

hotels = pd.read_csv('https://s3-us-west-2.amazonaws.com/fligoo.data-science/TechInterviews/HotelBookings/hotels.csv')

'''We carry out a first inspection of the database, evaluating the format of the 
variables and the existence of missing values'''

initial_inspection = type_and_missings(hotels)

print(initial_inspection['Tipos de datos']) 
'''There is some variables that are categoric and has object format. We need to
modify the format, to be able to use them in posterior analysis'''
print(initial_inspection['Valores faltantes']) 
''' Country column has 289 missing values. Depending on the importance of this
column, we could impute these values ​ or remove the rows if the amount of missing
data is manageable. Since I consider is not relevant in our analysis, we don't 
care about it.'''

###############################################################################
#CATEGORICAL VARIABLES 
###############################################################################
# Consistency check on categorical data. Specifically, we review unique categories
#in some key categorical columns. In this way, we check if there are variations
# in spelling
unique_values={}
for variable in hotels.columns[:-1:]:
    if hotels[variable].dtype== 'object' and (not variable=='arrival date'):
        unique_values[variable]=hotels[variable].unique()
        
print(unique_values)
        
# We express the categorical variables numerically, in case we (potentially) need 
#them in the subsequent analysis.
hotels['hotel'] = hotels['hotel'].replace({'City_Hotel': 1, 'Resort_Hotel': 0})
hotels['children'] = hotels['children'].replace({'children': 1, 'none': 0})
hotels['required_car_parking_spaces'] = hotels['required_car_parking_spaces'].replace({'parking': 1, 'none': 0})

print(hotels.dtypes)

###############################################################################
#DESCRIPTIVE STATISTICS
###############################################################################
#We perform descriptive statistics for continuous variables and binary variables
variables=['hotel', 'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
           'adults', 'children', 'is_repeated_guest', 'previous_cancellations',
           'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
           'average_daily_rate', 'required_car_parking_spaces', 
           'total_of_special_requests']

column_labels = ['City', 'Lead Time', 'Weekend Nights', 'Week Nights',
                 'Adults', 'Children', 'Repeated Guest', 'Prev. Cancellations',
                 'Prev. Bookings Not Canceled', 'Changes', 'Waiting List (Days)',
                 'Average Daily Rate', 'Parking', 
                 'Special Requests']


descriptive_statistics(hotels, variables, column_labels)


'''Descriptive statistics provide an overview of the distributions of numerical 
variables and the frequency of categories in categorical variables. First, we 
can observe that 61% of the reservations in the sample are made at the city 
hotel. In turn, this analysis shows that the average advance time with which 
reservations are made is 80 days. The daily rate is on average 100. We can also
point out that, on average, reservations include stays of 1 day on the weekend
and 2 or 3 days a week. It is also important to note that 4% of reservations 
are made by clients who have already visited the hotel previously, approximately
10% of reservations request parking space. Regarding the age of the clients, 
we can see that reservations involve, on average, approximately 2 adults 
(1.83 more precisely) and only 8% of the sample correspond to reservations for
stays of adults with children. In that sense, it is important to note that there
is an imbalance in the number of stages that involve children and those that 
do not. We will have to consider this in our subsequent analysis of part II.
Furthermore, this table of descriptive statistics allows us to identify some 
irregularities in the data. For instance, the minimum value of the "average_daily_rate"
variable is approximately -6. This implies that there are observations associated
with a negative average cost, which is illogical. I propose removing these 
observations from the sample. Similarly, the minimum value for the number of 
adults included in the stay is 0. Moreover, some of these stays involve children.
Children cannot make reservations or travel without their parents, so these 
observations are also uncommon. I suggest removing all observations from the 
sample where the quantity of adults is equal to 0.'''

#As indicated previously, I drop observations for which adults is equal to zero 
#and average_daily_rate is negative.
hotels = hotels[(hotels['adults'] != 0) & (hotels['average_daily_rate'] >= 0)]
'''This eliminates only 195 observations.''' 

#We create two new columns to calculate total number of days stayed and total cost
hotels['stays_nights_total']= hotels['stays_in_weekend_nights']+hotels['stays_in_week_nights']
hotels['total_cost']= hotels['stays_nights_total']*hotels['average_daily_rate']

###############################################################################
#Outliers
###############################################################################
#We delve deeper into the analysis of the existence of Outliers
#I select some key numeric variables for the outlier analysis
numeric_variables= ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 
                    'average_daily_rate', 'stays_nights_total','total_cost']

# We create a box plots for these variables
plt.figure(figsize=(15, 10))
for index, variable in enumerate(numeric_variables):
    plt.subplot(2, 3, index+1)
    sns.boxplot(y=hotels[variable])
    plt.title(variable)

plt.tight_layout()
plt.savefig('boxplots.png', dpi=300)
plt.show()

'''The outlier analysis involve generating box plots to visually identify 
outliers in some key variables. As we can see, the boxplots suggest the existence 
of reservations made with an extremely long lead time, unusually long stays, and
atypically high average and total costs. The first panel, for example, shows that
bookings made more than 300 days in advance are atypical values. These outliers
could be legitimate and represent real cases, or they could be data errors. 
Since we will be using this data for predictive models, outliers can have a 
significant impact on model performance. We could choose to remove outliers, cap
them at a certain value, or keep them if they make sense in your analysis.

Let's consider the particular case of the advance time with which bookings are 
made. Intuitively, outliers look like measurement errors. Work trips, on the one
hand, are planned a few days in advance. Vacations, on the other hand, although
they are planned further in advance, can hardly be projected over a horizon longer
than a year. A more in-depth analysis can investigate how these atypical data are
distributed among the reservations that involve stays with children and without
children. In this case, for the reason stated above, I would suggest setting the
lead_time variable to 365.'''

###############################################################################
#Correlations
###############################################################################
#Add children  to the numeric variables list
variables.append('stays_nights_total')
variables.append('total_cost')
# Calculate correlation matrix for all variables in  the list called "variables"
#and graph the correlation heath map
correlation_matrix = hotels[variables].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation map")
plt.savefig('correlation_map.png', dpi=300)
plt.show()

'''The correlation heat map shows the relationships between numerical variables. 
Warmer and cooler hues indicate the strength and direction of the compensation 
between the variables. This analysis helps us understand how different booking
characteristics may be related to each other.

These analyzes provide valuable insight into your data set and will allow us 
to make informed decisions about how to proceed with additional data analysis 
or cleaning. Most variables do not show a strong correlation with each other, 
indicating that they are quite independent. Most variables do not show a strong
correlation with each other, indicating that they are quite independent. This is
important for a predictive analysis, as it suggests that each variable captures 
different information. The highest correlations occur between the variables that
were generated as transformations of other base variables and these generating
variables.'''

#separting arrival day, month and year
hotels[['year', 'month', 'day']] = hotels['arrival_date'].str.split('-', expand=True)

hotels['year'] = hotels['year'].astype(int)
hotels['month'] = hotels['month'].astype(int)
hotels['day'] = hotels['day'].astype(int)

'''We consider important to investigate on how hotel income varies depending on
whether reservations involve stays with or without children. For this purpose, 
we make a bar graph.'''

# We calculate the average income per reservation for each type of hotel and 
#depending on the presence of children
average_income_by_hotel_and_children = hotels.groupby(['hotel',
    'children'])['total_cost'].mean().reset_index() #reset_index() converts
#the result to a new DataFrame average_income_by_hotel_and_children, where each
#row is a unique combination of 'hotel' and 'has_children', with the corresponding
#average income.

#We create the bar graph
plt.figure(figsize=(10, 6))
sns.barplot(x='hotel', y='total_cost', hue='children', data=average_income_by_hotel_and_children)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ['Without children', 'With children'])

plt.title('Average income per booking')
plt.xlabel('Hotel type')
plt.ylabel('Average income per booking')
plt.xticks(ticks=[0, 1], labels=['Resort Hotel', 'City Hotel'])
plt.savefig('income_barplot.png')
plt.show()

'''The graph illustrates that both the Resort Hotel and the City Hotel experience
 higher average revenue per reservation when children are included in the bookings
 compared to those without children. There can be several reasons for this 
 phenomenon. Firstly, family trips often involve more individuals, resulting in
 higher overall expenditures. Additionally, during holidays, families tend to 
 spend more on supplementary amenities such as dining, entertainment, and upgraded 
 accommodations, leading to an increase in the average revenue per booking. Also,
 family trips usually take place at weekends or some special dates, when demand
 an prices are higher. This trend holds true for both types of hotels, suggesting 
 that families could represent a profitable market for the hotel industry as a 
 whole. However, it's worth noting that the difference is less pronounced for 
 City Hotels, which are typically associated with business trips.
 
 In order to analyze the reason for this difference more thoroughly, we present
 graphs to evaluate whether there are variations in the distribution of some
 characteristics in reservations with and without children.'''


categoric_characteristics = ['meal', 'required_car_parking_spaces', 'distribution_channel', 
                             'customer_type', 'month', 'reserved_room_type']

# Configuramos la visualización de los gráficos
fig, axes = plt.subplots(3, 2, figsize=(15, 15))  # Ajustamos la cantidad de subplots necesarios
axes = axes.flatten()  # Aplanamos el array de ejes para facilitar su uso

# Creamos una variable para manejar la leyenda
handles, labels = None, None

for index, characteristic in enumerate(categoric_characteristics):
    # Obtenemos los datos para cada característica, separados por reservas con hijos y sin hijos
    data = hotels.groupby([characteristic, 'children']).size().unstack(fill_value=0)
    
    # Calculamos el porcentaje para cada grupo
    data_percentage = data.divide(data.sum(axis=1), axis=0) * 100
    
    # Creamos el gráfico de barras para cada categoría
    ax = data_percentage.plot(kind='bar', ax=axes[index], stacked=True)
    
    # Establecemos el título y las etiquetas de los ejes para cada subplot
    axes[index].set_title(f'Percentage of Bookings with/without Children by {characteristic}')
    axes[index].set_ylabel('Percentage')
    axes[index].set_xlabel(characteristic)
    
    # Guardamos los manejadores de la leyenda del último gráfico para usarlos globalmente
    if index == len(categoric_characteristics) - 1:
        handles, labels = ax.get_legend_handles_labels()

# Ajustamos el layout para evitar la superposición de subplots
plt.tight_layout()

# Eliminamos las leyendas individuales de cada subplot
for ax in axes:
    ax.get_legend().remove()

# Colocamos una leyenda global para toda la figura
fig.legend(handles, labels, loc='lower center', title='Children Included', bbox_to_anchor=(0.5, 0.95), ncol=len(labels))
plt.savefig('categoric_characteristics_barplot.png')
# Muestra la figura con todos los subplots
plt.show()

'''Based on the bar charts, here is a concise interpretation highlighting key 
insights. The first panel indicates that bookings involving families with children
have a higher incidence in the 'Full Board' category, which offers a complete 
meal service. This suggests that families may value the convenience and 
inclusivity of having all meals provided at the hotel, even if it comes at a
higher cost.

The second panel reveals that bookings that include car parking requirements see
a higher incidence of families with children. This could imply that traveling 
with children is often associated with the need for personal transportation, leading
to a demand for parking spaces.

The fifth panel highlights a pronounced increase in bookings with children during
July and August, aligning with summer vacation periods in Europe. This seasonal
trend indicates a potential opportunity for targeted marketing and promotions 
aimed at families during these peak months.

The sixth panel shows a tendency for families with children to book room types 
C, G, and H. If these room types are characterized by features that make them 
more expensive, such as additional space or amenities conducive to family stays,
this could explain the higher expenditure associated with bookings that include
children.

These insights reflect the specific needs and preferences of families when booking
hotel stays, such as comprehensive meal plans, the necessity of car parking, and
the desire for room types that cater to the comfort of a family. Hotels can leverage
this data to enhance service offerings and tailor pricing strategies to attract
and accommodate families, ultimately driving revenue during key seasonal periods'''

# Lista de variables para las cuales queremos hacer histogramas de densidad
continuous_characteristics = ['stays_in_weekend_nights', 'stays_nights_total', 'total_of_special_requests']

# Configura el tamaño de la figura que contendrá todos los subplots
plt.figure(figsize=(20, 15))

# Crea subplots para cada una de las variables
for index, variable in enumerate(continuous_characteristics, 1):
    # Crea un subplot en la posición correspondiente
    ax = plt.subplot(1, 3, index)
    
    # Filtra los datos para reservas con hijos y sin hijos
    data_with_children = hotels[hotels['children'] > 0][variable]
    data_without_children = hotels[hotels['children'] == 0][variable]
    
    # Encuentra los valores máximos y mínimos para establecer los bins
    combined_data = pd.concat([data_with_children, data_without_children])
    min_val, max_val = combined_data.min(), combined_data.max()
    bins = np.linspace(min_val, max_val, 16)  # Asegúrate de tener 15 bins como en tu configuración original
    
    # Dibuja los histogramas para cada grupo en el mismo subplot
    sns.histplot(data_without_children, bins=bins, color='orange', kde=False, stat='density', 
                 label='Sin Hijos', element='step', ax=ax, alpha=0.5)
    sns.histplot(data_with_children, bins=bins, color='skyblue', kde=False, stat='density', 
                 label='Con Hijos', element='step', ax=ax, alpha=0.5)
    
    # Establece el título del subplot y los nombres de los ejes
    ax.set_title(f'Histograma de {variable}')
    ax.set_xlabel(variable)
    ax.set_ylabel('Density')

    # Muestra la leyenda
    ax.legend()

# Ajusta el layout para evitar la superposición de subplots
plt.tight_layout()
plt.savefig('continuous_characteristics_histogram.png')
# Muestra la figura con todos los subplots
plt.show()

'''From the first histogram we can see that reservations without children 
show a higher concentration of short stays in weekends, with the majority
not exceeding one weekend night. On the other hand, bookings with children 
display a more uniform distribution, with a significant proportion extending
to two and three nights. If weekend stays are more expensive, this may explain
the differences that exist in the income of each of the types of reservations.
It also implies tha promoting weekend packages could be an effective strategy 
to attract families, while short-stay offers may be more appealing to guests
without children.
    
The second panel shows a clear trend towards longer stays among bookings with
children compared to those without, where stays tend to be shorter. Stays i
nvolving children often extend beyond a week, which may indicate planned family 
holidays. A leisure and pleasure trip, in turn, can explain a greater 
willingness to pay/spend.
    
Finally, from the third panel we can see bookings without children have a
lower frequency special requests, which might indicate a more utilitarian or
pragmatic approach to their stay.  Bookings with children often include at 
least one special request, which may reflect additional needs such as cribs,
larger rooms, or special meal services. Since each of these requests are paid,
they may help explain the higher average income generated by family trips
    
In this way, it may be interesting to develop and train a model to predict
whether hotel stays have children or not. We do this in the next part'''

#%% Part II

'''In the descriptive statistics table, we can observe that there are reservations
 where the reported length of stay (stays_nights_total) is 0 days. We will assume
 that these bookings were cancelled. Since our interest lies in predicting whether
 a stay will include children or not, we will focus exclusively on non-cancelled bookings.
 
 To define and train a prediction model we first must identify which variables (
 or transformations of them) could be useful to predict whether a stay included 
 children/infants. We included variables following the logic that trips taken 
 with children are less likely to be work trips.

 1) Lead Time: The time in advance with which the reservation is made could influence
the probability of traveling with children. The trips that people take with 
children, in general, are related to leisure time. At the same time, the time 
in advance with which leisure trips are planned differs from business trips, 
for example. For this reason, although the correlation (see correlation map) is
low, we include this variable

 2) Hotel Type (City or Resort): There could be a different trend in terms of traveling
with children between these two types of hotels. In line with the previous 
argument, we could expect that reservations at the city hotel are more related 
to work and, therefore, the probability that it includes children is lower.

 3) Number of Adults: The number of adults on the reservation could be correlated
with the presence of children since it is more likely that children travel 
with both parents, for example.

 4) Length of Stay: Both weekdays and weekends could have an influence. Work 
trips are typically made on weekdays. Weekend reservations are more likely to 
be family getaways.

 5) Type of Meal: Some types of meal plans may be more common in families with 
children.

 6) Customer Type: The type of customer could give us an idea about the probability of 
traveling with children.

 7) Market segment: We can also know the probability of traveling with children
from the information on the market segment of the person who made the reservation.

 8) Required car parking spaces 
 
 9) Total of special Request

A binary classification model would be suitable for this purpose. We could consider
models such as Logistic Regression, Decision Trees or Random Forest. We will use
a logistic regression model, since it is the one I am most familiar with.'''

###############################################################################
#Logistic regression
###############################################################################

'''The main issue lies in the imbalance that exists in the sample between the 
number of bookings that include children and those that do not. There are different
alternatives to address this inconvenience. In this case, we analyze two 
possibilities: (1) stratified sampling, (2) class weighting.'''

# Filtrar observaciones con stays_nights_total > 0
hotels = hotels[hotels['stays_nights_total'] > 0]

# Definir tus features y target
features = ['lead_time', 'hotel', 'adults', 'stays_in_weekend_nights', 'stays_in_week_nights',
            'stays_nights_total', 'total_cost', 'meal', 'customer_type', 'month']
target = 'children'

X = hotels[features]
y = hotels[target]

# Codificar variables categóricas usando One-Hot Encoding
encoder = OneHotEncoder(sparse=False, drop='first')
X_encoded = encoder.fit_transform(X[['hotel', 'meal', 'customer_type', 'month']])

# Escalar las características numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['lead_time', 'adults', 'stays_in_weekend_nights',
                                   'stays_in_week_nights', 'stays_nights_total', 'total_cost']])

# Combinar las características codificadas y escaladas
X_final = np.concatenate((X_encoded, X_scaled), axis=1)

# Dividir el conjunto de datos en entrenamiento y pruebaa. Stratifico para corregir
# desbalances
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Entrenar un modelo de regresión logística
model = LogisticRegression(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Imprimir informe de clasificación
print(classification_report(y_test, y_pred))

# Calcular la probabilidad de pertenecer a la clase positiva
y_prob = model.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) -  Class weighting')
plt.legend(loc='lower right')
plt.show()

# Dividir el conjunto de datos en entrenamiento y pruebaa. Stratifico para corregir
# desbalances
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

# Entrenar un modelo de regresión logística
model2 = LogisticRegression(random_state=42)
model2.fit(X_train2, y_train2)

# Realizar predicciones en el conjunto de prueba
y_pred2 = model2.predict(X_test2)

# Imprimir informe de clasificación
print(classification_report(y_test2, y_pred2))

# Calcular la probabilidad de pertenecer a la clase positiva
y_prob2 = model2.predict_proba(X_test2)[:, 1]

# Calcular la curva ROC
fpr2, tpr2, thresholds2 = roc_curve(y_test2, y_prob2)
roc_auc2 = auc(fpr2, tpr2)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr2, tpr2, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Stratified sampling')
plt.legend(loc='lower right')
plt.show()

'''The choice of the best model depends on our goals and the relative importance
 of precision, recall, and F1-score for our specific problem. Here is an analysis
 of both options:

Model (1):

- Precision for class 0 is high (0.92), indicating that when it predicts class 0,
it is often correct.
- Recall for class 1 is very low (0.04), meaning the model identifies very few
true positive cases.
- F1-score for class 1 is very low (0.08), indicating a low balance between 
precision and recall for class 1.
-The overall accuracy of the model is high (0.92), but this is due to a high 
number of true negatives (class 0).
-The weighted average precision is high (0.89) due to class imbalance, as most 
predictions fall into class 0.
-The macro average precision is moderate (0.70), but the macro average recall is low (0.52).

Model (2):

-Precision for class 0 is high (0.97), indicating that when it predicts class 0,
it is often correct.
-Recall for class 1 is high (0.75), meaning the model identifies a good proportion
of true positive cases.
-F1-score for class 1 is moderate (0.30), indicating a reasonable balance between
precision and recall for class 1.
-The overall accuracy of the model is moderate (0.73), and it is more balanced in terms of precision and recall between both classes.
-The weighted average precision is moderate (0.79).
-The macro average precision is moderate (0.58), and the macro average recall is high (0.73).

In summary, Model (2) appears to be more balanced in terms of precision and r
recall for both classes, and its F1-score for class 1 is higher compared to Model (1).
Since, based on the exploratory analysis of Part I, we are interested in identifying
stays that include families with children (as they are willing to spend more) to offer
them a more suitable package, we want to minimize the number of false negatives.
In this sense, we want a high recall. For this reason, I consider that the most 
appropriate alternative is the one that uses class weighting.'''





