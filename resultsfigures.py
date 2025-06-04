import pandas as pd
import matplotlib.pyplot as plt

#creatinine and ejection fraction linear separability scatterplot
data = pd.read_csv("./Final_Project/Dataset/filtered_data.csv", delimiter=",")

# map death status to marker and color
markers = {0: 'o', 1: '^'}  # circle for survived, triangle for dead
colors = {0: 'red', 1: 'cyan'}

#making scatterplot
plt.figure(figsize=(10, 6))

survived = data[data['death'] == 0]
plt.scatter(survived['creatinine.enzymatic.method'], survived['LVEF'], color='red', marker='o', label='survived')
dead = data[data['death'] == 1]
plt.scatter(dead['creatinine.enzymatic.method'], dead['LVEF'], color='blue', marker='x', label='dead')

plt.xlabel('Creatinine')
plt.ylabel('LVEF')
plt.title('LVEF vs Creatinine')
plt.legend(title='Patient Mortality')
plt.grid(True)
plt.show()


#death by month barchart
within_28_days = data['death.within.28.days'] == 1
within_3_months = (data['death.within.3.months'] == 1) & ~within_28_days
within_6_months = (data['death.within.6.months'] == 1) & ~(within_28_days | within_3_months)
deaths = ['within 28 days', 'within 3 months', 'within 6 months']
values = [
    within_28_days.sum(),
    within_3_months.sum(),
    within_6_months.sum()
]
plt.bar(deaths, values)
plt.title('Deaths Over Time')
plt.xlabel('Months')
plt.ylabel('Deaths')
plt.show()
