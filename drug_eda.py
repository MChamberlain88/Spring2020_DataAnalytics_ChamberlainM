# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 00:11:08 2020

@author: chambm6
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Reading in transformed categorical data and original data
df = pd.read_csv("C:/Users/chambm6/Desktop/Data_Analytics/eda.csv")
df2 = pd.read_csv("C:/Users/chambm6/Desktop/Data_Analytics/drug_consumption.data.csv")

plt.figure()

#Charts 1-10 consist of all by barplots, boxplots, and biolin plots I included in my report
sns.set(style="darkgrid")
#AgeBucket countplot
chart1 = sns.countplot(x="age", data=df,order=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'], palette = "rocket")
#Gender distribution plot
chart2 = sns.countplot(x="gender", data = df, palette = 'rocket')
#Level of Education distribution
chart3 = sns.countplot(x="education", data=df, order = ['Left School Before 16', 'Left School at 16', 'Left School at 17', 'Left School at 18', 'Some College, No Degree', 'Professional Diploma', 'University Degree', 'Masters Degree', 'Doctorate Degree'], palette = "rocket")
#Country distribution
chart4 = sns.countplot(x="country", data=df, order = ['UK', 'USA', 'Other', 'Canada', 'Austrailia', 'Republic of Ireland', 'New Zealand'], palette = "rocket")
#Ethnicity distribution
chart5 = sns.countplot(x="ethnicity", data=df,order = ['White', 'Other', 'Black', 'Aisian', 'Mixed-White/Black', 'Mixed-Black/Asian'], palette = 'rocket')
#List of all personality scores
pers = ['nscore', 'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss']
#Boxplot of all personality scores
chart6 = sns.boxplot(data = df2[pers], orient = 'h', palette = 'rocket')
#Violin plot of age vs nscore
chart7 = sns.violinplot(x= 'age', y='nscore', order = ['18-24', '25-34','35-44', '45-54', '55-64', '65+'], data = df, palette = 'rocket')
#Violin plot og age vs impulsivity score
chart8 = sns.violinplot(x= 'age', y='impulsive', order = ['18-24', '25-34','35-44', '45-54', '55-64', '65+'], data = df, palette = 'rocket')
#Violin plot of age vs escore
chart9 = sns.violinplot(x= 'age', y='escore',order = ['18-24', '25-34','35-44', '45-54', '55-64', '65+'], data = df, palette = 'rocket')
#list of all categorical variables
cat = ['age', 'gender', 'education', 'country', 'ethnicity']
#Boxplot of all categorical variables
chart10 = sns.boxplot(data = df2[cat], orient = 'h', palette = 'rocket')

#Adding titles to the aove plots and fixing xaxis label positions
chart1.set_xticklabels(chart1.get_xticklabels(),rotation=0)
chart1.set_title('Age Distribution')
chart2.set_xticklabels(chart2.get_xticklabels(), rotation=0)
chart3.set_xticklabels(chart3.get_xticklabels(), rotation = 90)
chart3.set_title('Education Distribution')
chart4.set_xticklabels(chart4.get_xticklabels(), rotation=45)
chart4.set_title('Country Distribution')
chart5.set_xticklabels(chart5.get_xticklabels(), rotation = 90)
chart5.set_title('Ethnicity Distribution')
chart6.set_title('Box Plot of Personality Scores')
chart7.set_title('Violin Plot of Age by Nscore')
chart8.set_title('Violin Plot of Age by Impulsivity')
chart9.set_title('Violin Plot of Age by Escore')
chart10.set_title('Boxplot of Categorical Features')

#Code to produce grouped par plot of drug use patterns of all 18 drugs
#The bar numbers go through and count the instances of different drug use frequencies in each drug
fig, ax = plt.subplots(figsize=(20,15))
bars1 = [len(df[df['alcohol'] == 'Never Used']),
         len(df[df['alcohol'] == 'Used over a Decade Ago']),
         len(df[df['alcohol'] == 'Used in Last Decade']),
         len(df[df['alcohol'] == 'Used in Last Year']),
         len(df[df['alcohol'] == 'Used in Last Month']),
         len(df[df['alcohol'] == 'Used in Last Week']),
         len(df[df['alcohol'] == 'Used in Last Day'])]
bars2 = [len(df[df['amphet'] == 'Never Used']),
         len(df[df['amphet'] == 'Used over a Decade Ago']),
         len(df[df['amphet'] == 'Used in Last Decade']),
         len(df[df['amphet'] == 'Used in Last Year']),
         len(df[df['amphet'] == 'Used in Last Month']),
         len(df[df['amphet'] == 'Used in Last Week']),
         len(df[df['amphet'] == 'Used in Last Day'])]
bars3 = [len(df[df['amyl'] == 'Never Used']),
         len(df[df['amyl'] == 'Used over a Decade Ago']),
         len(df[df['amyl'] == 'Used in Last Decade']),
         len(df[df['amyl'] == 'Used in Last Year']),
         len(df[df['amyl'] == 'Used in Last Month']),
         len(df[df['amyl'] == 'Used in Last Week']),
         len(df[df['amyl'] == 'Used in Last Day'])]

bars4 = [len(df[df['benzos'] == 'Never Used']),
         len(df[df['benzos'] == 'Used over a Decade Ago']),
         len(df[df['benzos'] == 'Used in Last Decade']),
         len(df[df['benzos'] == 'Used in Last Year']),
         len(df[df['benzos'] == 'Used in Last Month']),
         len(df[df['benzos'] == 'Used in Last Week']),
         len(df[df['benzos'] == 'Used in Last Day'])]
bars5 = [len(df[df['caff'] == 'Never Used']),
         len(df[df['caff'] == 'Used over a Decade Ago']),
         len(df[df['caff'] == 'Used in Last Decade']),
         len(df[df['caff'] == 'Used in Last Year']),
         len(df[df['caff'] == 'Used in Last Month']),
         len(df[df['caff'] == 'Used in Last Week']),
         len(df[df['caff'] == 'Used in Last Day'])]
bars6 = [len(df[df['cannabis'] == 'Never Used']),
         len(df[df['cannabis'] == 'Used over a Decade Ago']),
         len(df[df['cannabis'] == 'Used in Last Decade']),
         len(df[df['cannabis'] == 'Used in Last Year']),
         len(df[df['cannabis'] == 'Used in Last Month']),
         len(df[df['cannabis'] == 'Used in Last Week']),
         len(df[df['cannabis'] == 'Used in Last Day'])]
bars7 = [len(df[df['choc'] == 'Never Used']),
         len(df[df['choc'] == 'Used over a Decade Ago']),
         len(df[df['choc'] == 'Used in Last Decade']),
         len(df[df['choc'] == 'Used in Last Year']),
         len(df[df['choc'] == 'Used in Last Month']),
         len(df[df['choc'] == 'Used in Last Week']),
         len(df[df['choc'] == 'Used in Last Day'])]
bars8 = [len(df[df['coke'] == 'Never Used']),
         len(df[df['coke'] == 'Used over a Decade Ago']),
         len(df[df['coke'] == 'Used in Last Decade']),
         len(df[df['coke'] == 'Used in Last Year']),
         len(df[df['coke'] == 'Used in Last Month']),
         len(df[df['coke'] == 'Used in Last Week']),
         len(df[df['coke'] == 'Used in Last Day'])]
bars9 = [len(df[df['crack'] == 'Never Used']),
         len(df[df['crack'] == 'Used over a Decade Ago']),
         len(df[df['crack'] == 'Used in Last Decade']),
         len(df[df['crack'] == 'Used in Last Year']),
         len(df[df['crack'] == 'Used in Last Month']),
         len(df[df['crack'] == 'Used in Last Week']),
         len(df[df['crack'] == 'Used in Last Day'])]
bars10 = [len(df[df['ecstasy'] == 'Never Used']),
         len(df[df['ecstasy'] == 'Used over a Decade Ago']),
         len(df[df['ecstasy'] == 'Used in Last Decade']),
         len(df[df['ecstasy'] == 'Used in Last Year']),
         len(df[df['ecstasy'] == 'Used in Last Month']),
         len(df[df['ecstasy'] == 'Used in Last Week']),
         len(df[df['ecstasy'] == 'Used in Last Day'])]

bars11= [len(df[df['heroin'] == 'Never Used']),
         len(df[df['heroin'] == 'Used over a Decade Ago']),
         len(df[df['heroin'] == 'Used in Last Decade']),
         len(df[df['heroin'] == 'Used in Last Year']),
         len(df[df['heroin'] == 'Used in Last Month']),
         len(df[df['heroin'] == 'Used in Last Week']),
         len(df[df['heroin'] == 'Used in Last Day'])]
bars12= [len(df[df['ketamine'] == 'Never Used']),
         len(df[df['ketamine'] == 'Used over a Decade Ago']),
         len(df[df['ketamine'] == 'Used in Last Decade']),
         len(df[df['ketamine'] == 'Used in Last Year']),
         len(df[df['ketamine'] == 'Used in Last Month']),
         len(df[df['ketamine'] == 'Used in Last Week']),
         len(df[df['ketamine'] == 'Used in Last Day'])]
bars13= [len(df[df['legalh'] == 'Never Used']),
         len(df[df['legalh'] == 'Used over a Decade Ago']),
         len(df[df['legalh'] == 'Used in Last Decade']),
         len(df[df['legalh'] == 'Used in Last Year']),
         len(df[df['legalh'] == 'Used in Last Month']),
         len(df[df['legalh'] == 'Used in Last Week']),
         len(df[df['legalh'] == 'Used in Last Day'])]
bars14= [len(df[df['LSD'] == 'Never Used']),
         len(df[df['LSD'] == 'Used over a Decade Ago']),
         len(df[df['LSD'] == 'Used in Last Decade']),
         len(df[df['LSD'] == 'Used in Last Year']),
         len(df[df['LSD'] == 'Used in Last Month']),
         len(df[df['LSD'] == 'Used in Last Week']),
         len(df[df['LSD'] == 'Used in Last Day'])]
bars15= [len(df[df['meth'] == 'Never Used']),
         len(df[df['meth'] == 'Used over a Decade Ago']),
         len(df[df['meth'] == 'Used in Last Decade']),
         len(df[df['meth'] == 'Used in Last Year']),
         len(df[df['meth'] == 'Used in Last Month']),
         len(df[df['meth'] == 'Used in Last Week']),
         len(df[df['meth'] == 'Used in Last Day'])]
bars16= [len(df[df['mushrooms'] == 'Never Used']),
         len(df[df['mushrooms'] == 'Used over a Decade Ago']),
         len(df[df['mushrooms'] == 'Used in Last Decade']),
         len(df[df['mushrooms'] == 'Used in Last Year']),
         len(df[df['mushrooms'] == 'Used in Last Month']),
         len(df[df['mushrooms'] == 'Used in Last Week']),
         len(df[df['mushrooms'] == 'Used in Last Day'])]
bars17= [len(df[df['nicotine'] == 'Never Used']),
         len(df[df['nicotine'] == 'Used over a Decade Ago']),
         len(df[df['nicotine'] == 'Used in Last Decade']),
         len(df[df['nicotine'] == 'Used in Last Year']),
         len(df[df['nicotine'] == 'Used in Last Month']),
         len(df[df['nicotine'] == 'Used in Last Week']),
         len(df[df['nicotine'] == 'Used in Last Day'])]
bars18= [len(df[df['semer'] == 'Never Used']),
         len(df[df['semer'] == 'Used over a Decade Ago']),
         len(df[df['semer'] == 'Used in Last Decade']),
         len(df[df['semer'] == 'Used in Last Year']),
         len(df[df['semer'] == 'Used in Last Month']),
         len(df[df['semer'] == 'Used in Last Week']),
         len(df[df['semer'] == 'Used in Last Day'])]
bars19= [len(df[df['VSA'] == 'Never Used']),
         len(df[df['VSA'] == 'Used over a Decade Ago']),
         len(df[df['VSA'] == 'Used in Last Decade']),
         len(df[df['VSA'] == 'Used in Last Year']),
         len(df[df['VSA'] == 'Used in Last Month']),
         len(df[df['VSA'] == 'Used in Last Week']),
         len(df[df['VSA'] == 'Used in Last Day'])]

barWidth = .04
r1 = np.arange(len(bars1))

#Plotting each of the bars
plt.bar(r1, bars1, color = '#FF0000',width=barWidth, edgecolor='white', label='alcohol')
plt.bar(r1 + .04, bars2, color = '#FF6A00', width=barWidth, edgecolor='white', label='amphet')
plt.bar(r1 + .08, bars3, color= '#FF9400',  width=barWidth, edgecolor='white', label='amyl')
plt.bar(r1 + .12, bars4,color = '#FFBF00', width=barWidth, edgecolor='white', label='benzos')
plt.bar(r1 + .16, bars5, color = '#FFF600', width=barWidth, edgecolor='white', label='caff')
plt.bar(r1 + .20, bars6,color = '#B6FF00',  width=barWidth, edgecolor='white', label='cannabis')
plt.bar(r1 + .24, bars7,color = '#00FF15', width=barWidth, edgecolor='white', label='choc')
plt.bar(r1 + .28, bars8, color = '#00FFBB', width=barWidth, edgecolor='white', label='coke')
plt.bar(r1 + .32, bars9,color = '#00FAFF', width=barWidth, edgecolor='white', label='crack')
plt.bar(r1 + .36, bars10,color = '#00D4FF', width=barWidth, edgecolor='white', label='ecstasy')
plt.bar(r1 + .40, bars11,color = '#0088FF', width=barWidth, edgecolor='white', label='heroin')
plt.bar(r1 + .44, bars12,color= '#0008FF',  width=barWidth, edgecolor='white', label='ketamine')
plt.bar(r1 + .48, bars13, color = '#AA00FF', width=barWidth, edgecolor='white', label='legalh')
plt.bar(r1 + .52, bars14,color = '#DD00FF', width=barWidth, edgecolor='white', label='LSD')
plt.bar(r1 + .56, bars15, color = '#FF00FF', width=barWidth, edgecolor='white', label='meth')
plt.bar(r1 + .60, bars16,color = '#FF00B2', width=barWidth, edgecolor='white', label='mushrooms')
plt.bar(r1 + .64, bars17, color = '#FF006A', width=barWidth, edgecolor='white', label='nicotine')
plt.bar(r1 + .68, bars18,color = '#BA0053',  width=barWidth, edgecolor='white', label='semer')
plt.bar(r1 + .72, bars19,color = '#770039', width=barWidth, edgecolor='white', label='VSA')

ax.set_xticks(r1 + barWidth)
ax.legend()
plt.xticks([r + .08 for r in range(len(bars1))], ['Never Used', 'Used over a Decade Ago', 'Used in Last Decade', 'Used in Last Year', 'Used in Last Month', 'Used in Last Week', 'Used in Last Day'])
plt.title('Drug Use Patterns')
plt.show()

