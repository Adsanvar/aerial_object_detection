import pandas as pd
import numpy as np

property_data = pd.read_csv('data/data.csv')
property_area = pd.read_csv('data/original_area.csv')
detected_area = pd.read_csv('data/detected_area.csv')

house_percentage = {}
roof_area = {}
#gets property contour area, roof contour area, and calculates the percentages
for name in detected_area['Name']:
    
    prop = float(property_area.loc[property_area['Name'] == name, 'Area'])
    roof = float(detected_area.loc[detected_area['Name'] == name, 'Area'])
    name = str(name).split('.')[0]
    house_percentage[name] = roof / prop

#gets the area of the property and multiplies by the percentage
for house in house_percentage:
    r_area = float(property_data.loc[property_data['PIN'] == house, 'Shape.area']) * np.round(house_percentage[house], 3)
    roof_area[house] = np.round(r_area, 3)
    

df = pd.DataFrame(roof_area.items(), columns=['Property PIN', 'Roof Area'])

df.to_csv('data/predicted_area.csv')