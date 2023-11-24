import seaborn as sns

def centimeter_to_inch(centimeters):
    return centimeters * 1/2.54

def get_default_colors_categorical_seaborn(n=5):
    colors_custom = ['#FEC4DC', '#3F80B3', '#610099', '#CCA86C', '#91E693', 'lightgray']
    pallet_custom = sns.color_palette(colors_custom[:n])
    return pallet_custom