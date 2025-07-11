#before looking at the code, look at the excel sheet and properly understand what data he gave
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# storing .csv(Comma separated values which basically mean data is not in a tabular form, it is like this:- state,hyderabad,mumbai,chennai,quantity,4,6,8)
data= pd.read_csv('salesData.csv')

#creating a data frame(converting csv values into tabular format)
df=pd.DataFrame(data)

#things to analyze from the dataset
# 1. each month, how much sales is happening, ex- month 2 has least sales
# 2. product wise, how much sale is happening ex-smartwatch is sold the least
# 3. region wise ,how much sales in happening. example- delhi has more sales
# 4. what is the return rate of each product


#1. monthly sales
monthly_sales=df.groupby('Month')['TotalSales'].sum() #chatgpt what df.groupby function does
print("Monthly Sales Trends:")
print(monthly_sales)
sns.barplot(data=df, x="Month", y="TotalSales", estimator="mean",errorbar=None) #using seaborn to plot the bargraph
plt.show()

#2. product sales
product_sales = df.groupby('Product')['TotalSales'].sum().sort_values(ascending=False).head(5)
print("\nTop 5 Profitable Products:")
print(product_sales)
sns.barplot(data=df, x="Product", y="TotalSales", estimator="mean",errorbar=None)
plt.show()

#3. region sales
region_sales = df.groupby('City')['TotalSales'].sum().sort_values(ascending=False)
print("\nRegion-wise Performance:")
print(region_sales)
sns.barplot(data=df, x="City", y="TotalSales", estimator="mean",errorbar=None)
plt.show()

#4. return rates vs product category
# Total orders per category
total_orders = df['Category'].value_counts()

# Returned orders per category (only 'Returning' customers)
returned_orders = df[df['CustomerType'] == 'Returning']['Category'].value_counts()

# Return rate per category
return_rate = (returned_orders / total_orders * 100).round(2)

# Convert to DataFrame for Seaborn
return_df = return_rate.reset_index()
return_df.columns = ['Category', 'ReturnRate']

#plot using seaborn
sns.barplot(data=return_df, x='Category', y='ReturnRate')
plt.title('Return Rates by Product Category (%)')
plt.ylabel('Return Rate (%)') 
plt.xlabel('Product Category')
plt.tight_layout()
plt.show()
