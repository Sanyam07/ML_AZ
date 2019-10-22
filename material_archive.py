#plotting
plt.scatter(housing['longitude'], housing['latitude'], alpha=0.4, s=housing['population'] / 100,
           c=housing['median_house_value'], cmap='jet')
fig.set_size_inches(5, 5, forward=True)